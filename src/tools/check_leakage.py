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
    """IQP counts must not encode anything from valid/test.

    Two independent assertions:

    1. Items that appear *only* after the training cutoff must read as unseen.
       If a test-period item carried a nonzero popularity prior, the table
       would be telling the model something it could not know at serving time.
    2. Removing a training positive's own contribution must actually change
       its value. If it does not, the self-exclusion is silently broken and the
       model can read the label off the feature.
    """
    from ..data.iqp import build_iqp_table, lookup

    frame = store.frame
    cat_of = {}
    table = build_iqp_table(frame, cat_of)

    train_items = set(
        frame.loc[(frame["split"] == "train") & (frame["label"] > 0), "item_id"]
        .astype(int)
    )
    later_items = set(frame.loc[frame["split"] != "train", "item_id"].astype(int))
    unseen = sorted(later_items - train_items)
    if unseen:
        probe = np.array(unseen[:500], dtype=np.int64)
        vals = lookup(
            table,
            probe,
            np.full(len(probe), "", dtype=object),
            np.full(len(probe), "", dtype=object),
            np.zeros(len(probe), dtype=bool),
        )
        hot = float(vals[:, :3].max())
        if hot != 0.0:
            raise AssertionError(
                f"IQP leaks: {len(unseen)} test-only items have nonzero priors "
                f"(max {hot:.6f}); the table saw data past the training cutoff"
            )
        LOGGER.info("IQP unseen-item check OK (%d test-only items, all zero)", len(unseen))
    else:
        LOGGER.warning("IQP unseen-item check skipped: no test-only items")

    pos = frame[(frame["split"] == "train") & (frame["label"] > 0)]
    if len(pos):
        row = pos.iloc[0]
        item = np.array([int(row["item_id"])], dtype=np.int64)
        ctx_c = np.array([row["top_cat"]], dtype=object)
        ctx_b = np.array([row["top_brand"]], dtype=object)
        with_self = lookup(table, item, ctx_c, ctx_b, np.array([False]))
        without = lookup(table, item, ctx_c, ctx_b, np.array([True]))
        if float(np.abs(with_self - without).max()) == 0.0:
            raise AssertionError(
                "IQP self-exclusion is a no-op; a positive's own engagement is "
                "still inside its feature and the label is readable from it"
            )
        LOGGER.info(
            "IQP self-exclusion OK (delta %.6f)",
            float(np.abs(with_self - without).max()),
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
