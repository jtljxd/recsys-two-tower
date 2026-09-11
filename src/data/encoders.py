"""Vocabularies, item feature tables and dense normalisation.

Every statistic here is fitted on the *training* split only and then applied
unchanged to valid/test. That includes item popularity and item average rating,
which are the two easiest ways to leak the future into a recommender.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config, default_config
from ..utils import ensure_dir, get_logger

LOGGER = get_logger()

PAD = 0  # index 0 is padding / unknown everywhere

ITEM_DENSE_FEATURES: Tuple[str, ...] = (
    "price_log",
    "price_missing",
    "sales_rank_log",
    "sales_rank_missing",
    "title_len_log",
    "also_bought_log",
    "also_viewed_log",
    "item_pop_log",
    "item_avg_rating",
    "item_pos_ratio",
)


@dataclass
class Vocabulary:
    """String -> contiguous index, with 0 reserved for unknown."""

    mapping: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def fit(cls, values, min_freq: int = 1) -> "Vocabulary":
        counts: Dict[str, int] = {}
        for v in values:
            v = str(v or "")
            if not v:
                continue
            counts[v] = counts.get(v, 0) + 1
        kept = sorted(k for k, c in counts.items() if c >= min_freq)
        return cls({k: i + 1 for i, k in enumerate(kept)})

    def encode(self, value) -> int:
        return self.mapping.get(str(value or ""), PAD)

    def encode_many(self, values) -> np.ndarray:
        return np.array([self.encode(v) for v in values], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.mapping) + 1  # +1 for the unknown slot


@dataclass
class StandardScaler:
    """z-score with a guard against zero-variance columns."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "StandardScaler":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-6] = 1.0
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = (x - self.mean) / self.std
        return np.clip(out, -10.0, 10.0).astype(np.float32)


@dataclass
class ItemTable:
    """Dense per-item tensors, indexed directly by item_id."""

    dense: np.ndarray  # (n_items + 1, len(ITEM_DENSE_FEATURES))
    brand_ids: np.ndarray  # (n_items + 1,)
    cat_l1_ids: np.ndarray
    cat_leaf_ids: np.ndarray
    popularity: np.ndarray  # raw train counts, used for negative sampling

    @property
    def n_items(self) -> int:
        return len(self.brand_ids)


@dataclass
class Encoders:
    brand_vocab: Vocabulary
    cat_l1_vocab: Vocabulary
    cat_leaf_vocab: Vocabulary
    user_dense_scaler: StandardScaler
    item_table: ItemTable
    n_users: int

    @property
    def n_brands(self) -> int:
        return len(self.brand_vocab)

    @property
    def n_cat_l1(self) -> int:
        return len(self.cat_l1_vocab)

    @property
    def n_cat_leaf(self) -> int:
        return len(self.cat_leaf_vocab)

    @property
    def n_items(self) -> int:
        return self.item_table.n_items


def _log1p(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    values = np.where(np.isnan(values), 0.0, np.maximum(values, 0.0))
    return np.log1p(values)


def _build_item_table(
    items: pd.DataFrame,
    train_interactions: pd.DataFrame,
    brand_vocab: Vocabulary,
    cat_l1_vocab: Vocabulary,
    cat_leaf_vocab: Vocabulary,
    positive_threshold: float,
    n_items: int,
) -> ItemTable:
    # --- statistics computed on the training split ONLY -------------------
    pop = train_interactions.groupby("item_id").size()
    avg_rating = train_interactions.groupby("item_id")["rating"].mean()
    pos_ratio = (
        train_interactions.assign(
            _pos=(train_interactions["rating"] >= positive_threshold).astype(float)
        )
        .groupby("item_id")["_pos"]
        .mean()
    )

    size = n_items + 1
    dense = np.zeros((size, len(ITEM_DENSE_FEATURES)), dtype=np.float32)
    brand_ids = np.zeros(size, dtype=np.int64)
    cat_l1_ids = np.zeros(size, dtype=np.int64)
    cat_leaf_ids = np.zeros(size, dtype=np.int64)
    popularity = np.zeros(size, dtype=np.float64)

    price_raw = pd.to_numeric(items.get("price"), errors="coerce")
    rank_raw = pd.to_numeric(items.get("sales_rank"), errors="coerce")
    price_log = np.log1p(price_raw.fillna(0.0).clip(lower=0.0).to_numpy())
    rank_log = np.log1p(rank_raw.fillna(0.0).clip(lower=0.0).to_numpy())
    title_log = _log1p(items["title_len"])
    ab_log = _log1p(items["n_also_bought"])
    av_log = _log1p(items["n_also_viewed"])

    idx = items["item_id"].to_numpy(dtype=np.int64)
    dense[idx, 0] = price_log
    dense[idx, 1] = price_raw.isna().to_numpy(dtype=np.float32)
    dense[idx, 2] = rank_log
    dense[idx, 3] = rank_raw.isna().to_numpy(dtype=np.float32)
    dense[idx, 4] = title_log
    dense[idx, 5] = ab_log
    dense[idx, 6] = av_log

    pop_vec = pop.reindex(range(size), fill_value=0).to_numpy(dtype=np.float64)
    popularity[:] = pop_vec
    popularity[PAD] = 0.0
    dense[:, 7] = np.log1p(pop_vec)
    # Global means are the sane fallback for items unseen during training.
    global_rating = float(train_interactions["rating"].mean())
    global_pos = float(
        (train_interactions["rating"] >= positive_threshold).astype(float).mean()
    )
    dense[:, 8] = (
        avg_rating.reindex(range(size)).fillna(global_rating).to_numpy(dtype=np.float32)
    )
    dense[:, 9] = (
        pos_ratio.reindex(range(size)).fillna(global_pos).to_numpy(dtype=np.float32)
    )

    brand_ids[idx] = brand_vocab.encode_many(items["brand"])
    cat_l1_ids[idx] = cat_l1_vocab.encode_many(items["cat_l1"])
    cat_leaf_ids[idx] = cat_leaf_vocab.encode_many(items["cat_leaf"])

    # Padding row must stay neutral.
    dense[PAD] = 0.0
    return ItemTable(
        dense=dense,
        brand_ids=brand_ids,
        cat_l1_ids=cat_l1_ids,
        cat_leaf_ids=cat_leaf_ids,
        popularity=popularity,
    )


def fit(
    samples: pd.DataFrame,
    user_dense: np.ndarray,
    items: pd.DataFrame,
    interactions: pd.DataFrame,
    cfg: Optional[Config] = None,
) -> Encoders:
    cfg = cfg or default_config()
    train_mask = (samples["split"] == "train").to_numpy()
    train_interactions = interactions.merge(
        samples.loc[train_mask, ["user_id", "item_id"]], on=["user_id", "item_id"]
    )
    if train_interactions.empty:
        raise RuntimeError("training split is empty; check the leave-one-out split")

    brand_vocab = Vocabulary.fit(items["brand"], cfg.data.min_brand_freq)
    cat_l1_vocab = Vocabulary.fit(items["cat_l1"], cfg.data.min_category_freq)
    cat_leaf_vocab = Vocabulary.fit(items["cat_leaf"], cfg.data.min_category_freq)

    n_items = int(items["item_id"].max())
    n_users = int(samples["user_id"].max())

    item_table = _build_item_table(
        items,
        train_interactions,
        brand_vocab,
        cat_l1_vocab,
        cat_leaf_vocab,
        cfg.data.positive_threshold,
        n_items,
    )

    # Item dense columns are already log-scaled; normalise them too so the
    # tower sees comparable magnitudes.
    item_scaler = StandardScaler.fit(item_table.dense[1:])
    item_table.dense[1:] = item_scaler.transform(item_table.dense[1:])
    item_table.dense[PAD] = 0.0

    scaler = StandardScaler.fit(user_dense[train_mask])
    LOGGER.info(
        "vocab sizes | brands=%d cat_l1=%d cat_leaf=%d items=%d users=%d",
        len(brand_vocab),
        len(cat_l1_vocab),
        len(cat_leaf_vocab),
        n_items,
        n_users,
    )
    return Encoders(
        brand_vocab=brand_vocab,
        cat_l1_vocab=cat_l1_vocab,
        cat_leaf_vocab=cat_leaf_vocab,
        user_dense_scaler=scaler,
        item_table=item_table,
        n_users=n_users,
    )


def save(encoders: Encoders, out_dir: Path) -> None:
    ensure_dir(out_dir)
    with open(out_dir / "vocabs.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "brand": encoders.brand_vocab.mapping,
                "cat_l1": encoders.cat_l1_vocab.mapping,
                "cat_leaf": encoders.cat_leaf_vocab.mapping,
                "n_users": encoders.n_users,
            },
            f,
        )
    np.savez_compressed(
        out_dir / "encoders.npz",
        dense_mean=encoders.user_dense_scaler.mean,
        dense_std=encoders.user_dense_scaler.std,
        item_dense=encoders.item_table.dense,
        item_brand=encoders.item_table.brand_ids,
        item_cat_l1=encoders.item_table.cat_l1_ids,
        item_cat_leaf=encoders.item_table.cat_leaf_ids,
        item_pop=encoders.item_table.popularity,
    )


def load(out_dir: Path) -> Encoders:
    with open(out_dir / "vocabs.json", "r", encoding="utf-8") as f:
        vocabs = json.load(f)
    arrays = np.load(out_dir / "encoders.npz")
    return Encoders(
        brand_vocab=Vocabulary(vocabs["brand"]),
        cat_l1_vocab=Vocabulary(vocabs["cat_l1"]),
        cat_leaf_vocab=Vocabulary(vocabs["cat_leaf"]),
        user_dense_scaler=StandardScaler(arrays["dense_mean"], arrays["dense_std"]),
        item_table=ItemTable(
            dense=arrays["item_dense"],
            brand_ids=arrays["item_brand"],
            cat_l1_ids=arrays["item_cat_l1"],
            cat_leaf_ids=arrays["item_cat_leaf"],
            popularity=arrays["item_pop"],
        ),
        n_users=int(vocabs["n_users"]),
    )
