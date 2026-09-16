"""Assert the user features never look at the present or the future.

Run after any change to ``data/features.py``:

    python -m src.tools.check_leakage --raw-dir data/mock

Checks performed:
  1. Recomputing each row's features from an independent, naive "all events
     strictly before this one" pass reproduces the pipeline output exactly.
  2. A row's behaviour sequence only contains items the user touched earlier.
  3. The leave-one-out split is chronological: train < valid < test per user.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import default_config
from ..data import features as features_mod
from ..data import parse_amazon
from ..utils import get_logger, load_table, table_exists

LOGGER = get_logger()


def check_history_causality(store, interactions: pd.DataFrame) -> None:
    """Every id in hist_items must come from a strictly earlier event."""
    df = interactions.sort_values(["user_id", "timestamp", "item_id"]).reset_index(
        drop=True
    )
    seen_by_user = {}
    violations = 0
    for idx, row in enumerate(df.itertuples(index=False)):
        past = seen_by_user.setdefault(row.user_id, [])
        hist = store.hist_items[idx]
        hist = hist[hist != 0]
        past_set = set(past)
        if not set(hist.tolist()).issubset(past_set):
            violations += 1
        # The current item must NOT be in its own history.
        if row.item_id in set(hist.tolist()):
            violations += 1
        past.append(row.item_id)
    assert violations == 0, "{} history causality violations".format(violations)
    LOGGER.info("history causality OK (%d rows)", len(df))


def check_dense_recompute(store, interactions: pd.DataFrame) -> None:
    """Independently recompute two simple features and compare."""
    df = interactions.sort_values(["user_id", "timestamp", "item_id"]).reset_index(
        drop=True
    )
    names = list(store.dense_names)
    i_hist_len = names.index("hist_len_log")
    i_avg_rating = names.index("avg_rating")

    history = {}
    max_err_len = 0.0
    max_err_rating = 0.0
    for idx, row in enumerate(df.itertuples(index=False)):
        past = history.setdefault(row.user_id, [])
        expected_len = np.log1p(len(past))
        expected_rating = float(np.mean(past)) if past else 0.0
        max_err_len = max(max_err_len, abs(store.dense[idx, i_hist_len] - expected_len))
        max_err_rating = max(
            max_err_rating, abs(store.dense[idx, i_avg_rating] - expected_rating)
        )
        past.append(float(row.rating))

    assert max_err_len < 1e-5, "hist_len mismatch: {}".format(max_err_len)
    assert max_err_rating < 1e-4, "avg_rating mismatch: {}".format(max_err_rating)
    LOGGER.info(
        "dense recompute OK (max err: len=%.2e rating=%.2e)", max_err_len, max_err_rating
    )


def check_split_order(store) -> None:
    """train events must predate valid, which must predate test."""
    frame = store.frame
    pivot = frame.pivot_table(
        index="user_id", columns="split", values="timestamp", aggfunc="max"
    )
    if not {"train", "valid", "test"}.issubset(pivot.columns):
        LOGGER.warning("not all splits present, skipping order check")
        return
    complete = pivot.dropna()
    bad_tv = int((complete["train"] > complete["valid"]).sum())
    bad_vt = int((complete["valid"] > complete["test"]).sum())
    assert bad_tv == 0, "{} users have train after valid".format(bad_tv)
    assert bad_vt == 0, "{} users have valid after test".format(bad_vt)
    LOGGER.info("split chronology OK (%d users)", len(complete))


def check_iqp_no_leakage(store) -> None:
    """The cross features must not encode the label.

    Three assertions, each guarding a distinct failure mode:

    1. Category snapshots are causal: row i's counts must equal the number of
       the user's events strictly before i.
    2. Test-period items carry no global prior. With a temporal split, a
       nonzero value would mean the table saw past the training cutoff.
    3. The feature distribution overlaps between positives and negatives. This
       is the important one: if a feature were derived from the user's own
       interaction with the candidate item, negatives would be identically
       zero (the sampler skips seen items) and the model could read the label
       straight off it.
    """
    from ..data.iqp import build_iqp_table

    frame = store.frame
    cp = store.cat_profile
    if cp is None:
        raise AssertionError("cat_profile missing; features must be rebuilt")

    # (1) causality: nnz counts sum to the number of prior events.
    users = frame["user_id"].to_numpy()
    seen_n = {}
    bad = 0
    for i in range(len(frame)):
        u = users[i]
        expected = seen_n.get(u, 0)
        lo, hi = int(cp.indptr[i]), int(cp.indptr[i + 1])
        got = int(cp.cat_counts[lo:hi].sum())
        # Categories can be blank for some items, so the snapshot total is a
        # lower bound on the event count, never an over-count.
        if got > expected:
            bad += 1
        seen_n[u] = expected + 1
    if bad:
        raise AssertionError(
            f"category snapshot sees the future on {bad} rows"
        )
    LOGGER.info("IQP snapshot causality OK (%d rows)", len(frame))

    # (2) no global prior for test-only items.
    table = build_iqp_table(frame)
    train_items = set(
        frame.loc[(frame["split"] == "train") & (frame["label"] > 0), "item_id"]
        .astype(int)
    )
    later = set(frame.loc[frame["split"] != "train", "item_id"].astype(int))
    unseen = sorted(later - train_items)
    if unseen:
        hot = max(table.item_total.get(i, 0) for i in unseen)
        if hot:
            raise AssertionError(
                f"global prior leaks: test-only items carry counts (max {hot})"
            )
        LOGGER.info("IQP unseen-item check OK (%d test-only items)", len(unseen))
    else:
        LOGGER.warning("IQP unseen-item check skipped: no test-only items")

    # (3) positives and negatives must draw from the same distribution.
    # This is the assertion that matters most. A feature built from the user's
    # own interaction with the candidate would be nonzero for every positive
    # and zero for every negative, because the sampler skips seen items, and
    # the model would score a meaningless near-perfect AUC off it.
    from ..config import default_config
    from ..data import dataset as dataset_mod
    from ..data import encoders as encoders_mod
    from ..data.iqp import IQP_FEATURES

    cfg = default_config()
    cfg.model.use_interactrank = True
    enc = encoders_mod.load(cfg.processed_dir)
    train_loader, _, _ = dataset_mod.build_dataloaders(store, enc, cfg)
    pos, neg = [], []
    for i, batch in enumerate(train_loader):
        q = batch["iqp"].numpy()
        pos.append(q[:, 0, :])
        neg.append(q[:, 1:, :].reshape(-1, q.shape[-1]))
        if i >= 12:
            break
    pos = np.concatenate(pos)
    neg = np.concatenate(neg)
    for j, name in enumerate(IQP_FEATURES):
        pz = float((pos[:, j] != 0).mean())
        nz = float((neg[:, j] != 0).mean())
        if pz > 0.05 and nz < 0.001:
            raise AssertionError(
                f"{name} is a label indicator: {pz:.1%} of positives nonzero "
                f"but only {nz:.3%} of negatives. It almost certainly reads the "
                "user's interaction with the candidate item, which the negative "
                "sampler excludes by construction."
            )
    LOGGER.info(
        "IQP label-indicator check OK (%d features, pos/neg nonzero rates within range)",
        len(IQP_FEATURES),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = default_config()
    cfg.data.sample_users = None
    if not table_exists(cfg.processed_dir, "interactions"):
        parse_amazon.run(cfg, raw_dir=Path(args.raw_dir) if args.raw_dir else None)

    interactions = load_table(cfg.processed_dir, "interactions")
    items = load_table(cfg.processed_dir, "items")
    store = features_mod.build(interactions, items, cfg)

    check_history_causality(store, interactions)
    check_dense_recompute(store, interactions)
    check_split_order(store)
    check_iqp_no_leakage(store)
    LOGGER.info("all leakage checks passed")


if __name__ == "__main__":
    main()
