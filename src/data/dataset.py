"""Torch datasets: negative sampling for training, fixed candidates for eval.

Training uses popularity^alpha negatives resampled every epoch. Evaluation uses
a frozen 1-positive + N-negative candidate set so metrics are comparable across
epochs and runs.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config, default_config
from ..utils import get_logger
from .encoders import Encoders
from .features import FeatureStore
from .iqp import IQPLookup, build_iqp_table

LOGGER = get_logger()


class NegativeSampler:
    """Draws items from a popularity^alpha distribution, skipping seen items."""

    def __init__(
        self,
        popularity: np.ndarray,
        alpha: float,
        seen: Dict[int, set],
        seed: int = 42,
    ):
        probs = np.power(np.maximum(popularity, 0.0), alpha)
        probs[0] = 0.0  # never sample the padding slot
        total = probs.sum()
        if total <= 0:  # degenerate fallback: uniform over real items
            probs = np.ones_like(probs)
            probs[0] = 0.0
            total = probs.sum()
        self.probs = probs / total
        self.n_items = len(probs)
        self.seen = seen
        self.rng = np.random.default_rng(seed)

    def sample(self, user_id: int, k: int, max_tries: int = 10) -> np.ndarray:
        """Rejection-sample k items the user has never interacted with."""
        seen = self.seen.get(user_id, ())
        out: List[int] = []
        for _ in range(max_tries):
            need = k - len(out)
            if need <= 0:
                break
            draws = self.rng.choice(self.n_items, size=need * 2, p=self.probs)
            for it in draws:
                if it not in seen:
                    out.append(int(it))
                    if len(out) == k:
                        break
        while len(out) < k:  # give up on the constraint rather than loop forever
            out.append(int(self.rng.integers(1, self.n_items)))
        return np.array(out[:k], dtype=np.int64)


def build_seen_map(samples: pd.DataFrame) -> Dict[int, set]:
    """All items each user ever touched, used to keep negatives clean."""
    return (
        samples.groupby("user_id")["item_id"]
        .apply(lambda s: set(s.tolist()))
        .to_dict()
    )


@dataclass
class SplitData:
    """One split's rows, already sliced out of the global feature store."""

    user_ids: np.ndarray
    item_ids: np.ndarray
    hist_items: np.ndarray
    hist_lens: np.ndarray
    dense: np.ndarray
    top_cat_ids: np.ndarray
    top_brand_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.user_ids)


def slice_split(
    store: FeatureStore, encoders: Encoders, split: str, positives_only: bool = True
) -> SplitData:
    frame = store.frame
    mask = (frame["split"] == split).to_numpy()
    if positives_only:
        # Only rating >= threshold events act as positives for ranking.
        mask = mask & (frame["label"].to_numpy() > 0.5)
    idx = np.where(mask)[0]
    dense = encoders.user_dense_scaler.transform(store.dense[idx])
    return SplitData(
        user_ids=frame["user_id"].to_numpy()[idx],
        item_ids=frame["item_id"].to_numpy()[idx],
        hist_items=store.hist_items[idx],
        hist_lens=store.hist_lens[idx],
        dense=dense,
        top_cat_ids=encoders.cat_leaf_vocab.encode_many(
            frame["top_cat"].to_numpy()[idx]
        ),
        top_brand_ids=encoders.brand_vocab.encode_many(
            frame["top_brand"].to_numpy()[idx]
        ),
    )


class TrainDataset(Dataset):
    """Yields 1 positive + K negatives per row, resampled each epoch."""

    def __init__(
        self,
        data: SplitData,
        sampler: NegativeSampler,
        n_negatives: int,
        iqp: Optional[IQPLookup] = None,
        is_train_split: bool = True,
    ):
        self.data = data
        self.sampler = sampler
        self.n_negatives = n_negatives
        self.iqp = iqp
        self.is_train_split = is_train_split

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        d = self.data
        user_id = int(d.user_ids[idx])
        negatives = self.sampler.sample(user_id, self.n_negatives)
        items = np.concatenate([[d.item_ids[idx]], negatives])
        labels = np.zeros(len(items), dtype=np.float32)
        labels[0] = 1.0
        out = {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "hist_items": torch.from_numpy(d.hist_items[idx]),
            "hist_len": torch.tensor(d.hist_lens[idx], dtype=torch.long),
            "dense": torch.from_numpy(d.dense[idx]),
            "top_cat": torch.tensor(d.top_cat_ids[idx], dtype=torch.long),
            "top_brand": torch.tensor(d.top_brand_ids[idx], dtype=torch.long),
            "items": torch.from_numpy(items),
            "labels": torch.from_numpy(labels),
        }
        if self.iqp is not None:
            out["iqp"] = torch.from_numpy(
                self.iqp(items, d.top_cat_ids[idx], d.top_brand_ids[idx],
                         self.is_train_split)
            )
        return out


class EvalDataset(Dataset):
    """Frozen candidate sets so every epoch is scored on identical data."""

    def __init__(
        self,
        data: SplitData,
        sampler: NegativeSampler,
        n_negatives: int,
        seed: int,
        iqp: Optional[IQPLookup] = None,
    ):
        self.data = data
        self.n_negatives = n_negatives
        self.iqp = iqp
        rng_state = sampler.rng
        sampler.rng = np.random.default_rng(seed)
        self.candidates = np.stack(
            [
                np.concatenate(
                    [
                        [data.item_ids[i]],
                        sampler.sample(int(data.user_ids[i]), n_negatives),
                    ]
                )
                for i in range(len(data))
            ]
        ).astype(np.int64)
        sampler.rng = rng_state

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        d = self.data
        labels = np.zeros(self.n_negatives + 1, dtype=np.float32)
        labels[0] = 1.0
        out = {
            "user_id": torch.tensor(int(d.user_ids[idx]), dtype=torch.long),
            "hist_items": torch.from_numpy(d.hist_items[idx]),
            "hist_len": torch.tensor(d.hist_lens[idx], dtype=torch.long),
            "dense": torch.from_numpy(d.dense[idx]),
            "top_cat": torch.tensor(d.top_cat_ids[idx], dtype=torch.long),
            "top_brand": torch.tensor(d.top_brand_ids[idx], dtype=torch.long),
            "items": torch.from_numpy(self.candidates[idx]),
            "labels": torch.from_numpy(labels),
        }
        if self.iqp is not None:
            # valid/test rows never entered the table, so nothing to exclude.
            out["iqp"] = torch.from_numpy(
                self.iqp(self.candidates[idx], d.top_cat_ids[idx],
                         d.top_brand_ids[idx], False)
            )
        return out


def build_dataloaders(
    store: FeatureStore, encoders: Encoders, cfg: Optional[Config] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    cfg = cfg or default_config()
    seen = build_seen_map(store.frame)
    sampler = NegativeSampler(
        encoders.item_table.popularity,
        cfg.data.neg_sampling_alpha,
        seen,
        cfg.data.seed,
    )

    # IQP lookup is only built for InteractRank; the other models never see the
    # extra batch key and their code paths stay byte-identical.
    iqp = None
    if cfg.model.use_interactrank:
        table = build_iqp_table(
            store.frame,
            # item_id -> leaf category id, straight off the encoded item table.
            {
                int(i): int(c)
                for i, c in enumerate(encoders.item_table.cat_leaf_ids)
            },
        )
        iqp = IQPLookup(table, encoders.cat_leaf_vocab, encoders.brand_vocab)

    train = TrainDataset(
        slice_split(store, encoders, "train"),
        sampler,
        cfg.data.train_negatives,
        iqp=iqp,
        is_train_split=True,
    )
    valid = EvalDataset(
        slice_split(store, encoders, "valid"),
        sampler,
        cfg.data.eval_negatives,
        cfg.data.seed + 1,
        iqp=iqp,
    )
    test = EvalDataset(
        slice_split(store, encoders, "test"),
        sampler,
        cfg.data.eval_negatives,
        cfg.data.seed + 2,
        iqp=iqp,
    )
    LOGGER.info(
        "dataset sizes | train=%d valid=%d test=%d", len(train), len(valid), len(test)
    )

    common = {"num_workers": cfg.train.num_workers, "pin_memory": False}
    return (
        DataLoader(
            train, batch_size=cfg.train.batch_size, shuffle=True, drop_last=False, **common
        ),
        DataLoader(valid, batch_size=cfg.train.eval_batch_size, shuffle=False, **common),
        DataLoader(test, batch_size=cfg.train.eval_batch_size, shuffle=False, **common),
    )
