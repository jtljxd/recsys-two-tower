"""Parse the raw Amazon-Electronics ``.json.gz`` dumps into tidy parquet files.

Two gotchas this module handles:

1. ``reviews_*.json.gz`` is strict JSON-per-line, but ``meta_*.json.gz`` is a
   dump of Python ``dict`` reprs (single quotes), so ``json.loads`` fails on it.
   We try ``json.loads`` first and fall back to ``ast.literal_eval``.
2. The review file is large, so we never hold raw text in memory: review length
   is computed while streaming and the text itself is dropped.

Outputs (under ``data/processed``):
    interactions.parquet : one row per (user, item) review event
    items.parquet        : one row per item with side information
"""

import argparse
import ast
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, Optional, Set

import numpy as np
import pandas as pd

from ..config import Config, default_config
from ..utils import ensure_dir, get_logger, save_table, set_seed, timed

LOGGER = get_logger()


def _iter_json_gz(path: Path) -> Iterator[dict]:
    """Yield one dict per line, tolerating Python-repr style records."""
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except (json.JSONDecodeError, ValueError):
                try:
                    yield ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    continue


# --------------------------------------------------------------------------
# reviews
# --------------------------------------------------------------------------
def _count_user_interactions(reviews_path: Path) -> Counter:
    """First streaming pass: how many reviews does each user have?"""
    counts: Counter = Counter()
    for rec in _iter_json_gz(reviews_path):
        uid = rec.get("reviewerID")
        if uid:
            counts[uid] += 1
    return counts


def _select_users(
    counts: Counter, sample_users: Optional[int], min_interactions: int, seed: int
) -> Set[str]:
    """Keep users with enough history, then optionally subsample."""
    eligible = [u for u, c in counts.items() if c >= min_interactions]
    LOGGER.info(
        "users with >= %d interactions: %d / %d",
        min_interactions,
        len(eligible),
        len(counts),
    )
    if sample_users is None or sample_users >= len(eligible):
        return set(eligible)
    rng = np.random.default_rng(seed)
    eligible.sort()  # stable order before sampling -> reproducible
    picked = rng.choice(len(eligible), size=sample_users, replace=False)
    return {eligible[i] for i in picked}


def _helpful_ratio(helpful) -> float:
    """helpful is [useful_votes, total_votes]; NaN when nobody voted."""
    try:
        up, total = helpful[0], helpful[1]
    except (TypeError, IndexError, KeyError):
        return np.nan
    if not total:
        return np.nan
    return float(up) / float(total)


def _load_reviews(reviews_path: Path, keep_users: Set[str]) -> pd.DataFrame:
    """Second streaming pass: materialise rows for the selected users only."""
    rows = []
    for rec in _iter_json_gz(reviews_path):
        uid = rec.get("reviewerID")
        if uid not in keep_users:
            continue
        asin = rec.get("asin")
        ts = rec.get("unixReviewTime")
        if not asin or ts is None:
            continue
        text = rec.get("reviewText") or ""
        summary = rec.get("summary") or ""
        rows.append(
            (
                uid,
                asin,
                float(rec.get("overall", 0.0)),
                int(ts),
                len(text.split()),
                len(summary.split()),
                _helpful_ratio(rec.get("helpful")),
            )
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "user_raw",
            "item_raw",
            "rating",
            "timestamp",
            "review_len",
            "summary_len",
            "helpful_ratio",
        ],
    )
    return df


# --------------------------------------------------------------------------
# meta
# --------------------------------------------------------------------------
def _first_category(categories) -> str:
    """categories looks like [[l1, l2, ...], [...]]; take the first chain."""
    if not isinstance(categories, (list, tuple)) or not categories:
        return ""
    chain = categories[0]
    if not isinstance(chain, (list, tuple)) or not chain:
        return ""
    return str(chain[0])


def _leaf_category(categories) -> str:
    if not isinstance(categories, (list, tuple)) or not categories:
        return ""
    chain = categories[0]
    if not isinstance(chain, (list, tuple)) or not chain:
        return ""
    return str(chain[-1])


def _main_sales_rank(sales_rank) -> float:
    """salesRank is {category: rank}; keep the best (smallest) rank."""
    if not isinstance(sales_rank, dict) or not sales_rank:
        return np.nan
    try:
        return float(min(sales_rank.values()))
    except (TypeError, ValueError):
        return np.nan


def _load_meta(meta_path: Path, keep_items: Set[str]) -> pd.DataFrame:
    rows = []
    seen: Set[str] = set()
    for rec in _iter_json_gz(meta_path):
        asin = rec.get("asin")
        if asin not in keep_items or asin in seen:
            continue
        seen.add(asin)
        price = rec.get("price")
        try:
            price = float(price) if price is not None else np.nan
        except (TypeError, ValueError):
            price = np.nan
        title = rec.get("title") or ""
        related = rec.get("related") or {}
        rows.append(
            (
                asin,
                str(rec.get("brand") or ""),
                _first_category(rec.get("categories")),
                _leaf_category(rec.get("categories")),
                price,
                _main_sales_rank(rec.get("salesRank")),
                len(title.split()),
                len(related.get("also_bought") or []),
                len(related.get("also_viewed") or []),
            )
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "item_raw",
            "brand",
            "cat_l1",
            "cat_leaf",
            "price",
            "sales_rank",
            "title_len",
            "n_also_bought",
            "n_also_viewed",
        ],
    )
    return df


# --------------------------------------------------------------------------
# filtering / id mapping
# --------------------------------------------------------------------------
def _iterative_core_filter(
    df: pd.DataFrame, min_user: int, min_item: int, max_rounds: int = 10
) -> pd.DataFrame:
    """Alternate user/item pruning until the frame stops shrinking."""
    for _ in range(max_rounds):
        before = len(df)
        item_counts = df["item_raw"].value_counts()
        df = df[df["item_raw"].map(item_counts) >= min_item]
        user_counts = df["user_raw"].value_counts()
        df = df[df["user_raw"].map(user_counts) >= min_user]
        if len(df) == before:
            break
    return df.reset_index(drop=True)


def _assign_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw string ids to contiguous ints. 0 is reserved for padding/unknown."""
    users = sorted(df["user_raw"].unique())
    items = sorted(df["item_raw"].unique())
    user_map = {u: i + 1 for i, u in enumerate(users)}
    item_map = {a: i + 1 for i, a in enumerate(items)}
    df = df.copy()
    df["user_id"] = df["user_raw"].map(user_map).astype(np.int64)
    df["item_id"] = df["item_raw"].map(item_map).astype(np.int64)
    return df


def run(cfg: Optional[Config] = None, raw_dir: Optional[Path] = None) -> Dict[str, Path]:
    cfg = cfg or default_config()
    set_seed(cfg.data.seed)
    raw_dir = Path(raw_dir) if raw_dir else cfg.raw_dir
    reviews_path = raw_dir / cfg.data.reviews_file
    meta_path = raw_dir / cfg.data.meta_file
    for p in (reviews_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(
                "missing raw file: {}\nPoint --raw-dir at the directory holding "
                "the Amazon .json.gz dumps.".format(p)
            )

    with timed("pass 1/3 counting user interactions", LOGGER):
        counts = _count_user_interactions(reviews_path)
    keep_users = _select_users(
        counts, cfg.data.sample_users, cfg.data.min_user_interactions, cfg.data.seed
    )
    LOGGER.info("sampled %d users", len(keep_users))

    with timed("pass 2/3 loading reviews", LOGGER):
        inter = _load_reviews(reviews_path, keep_users)
    LOGGER.info("raw interactions: %d", len(inter))

    # Drop duplicate (user, item) pairs, keeping the earliest event.
    inter = inter.sort_values(["user_raw", "timestamp"]).drop_duplicates(
        ["user_raw", "item_raw"], keep="first"
    )
    inter = _iterative_core_filter(
        inter, cfg.data.min_user_interactions, cfg.data.min_item_interactions
    )
    LOGGER.info(
        "after core filter: %d interactions, %d users, %d items",
        len(inter),
        inter["user_raw"].nunique(),
        inter["item_raw"].nunique(),
    )
    if inter.empty:
        raise RuntimeError("no interactions survived filtering; loosen the thresholds")

    inter = _assign_ids(inter)
    keep_items = set(inter["item_raw"].unique())

    with timed("pass 3/3 loading item metadata", LOGGER):
        meta = _load_meta(meta_path, keep_items)
    LOGGER.info("metadata found for %d / %d items", len(meta), len(keep_items))

    # Items without metadata still need a row so downstream joins never drop data.
    item_ids = (
        inter[["item_raw", "item_id"]].drop_duplicates().reset_index(drop=True)
    )
    items = item_ids.merge(meta, on="item_raw", how="left")
    items["brand"] = items["brand"].fillna("")
    items["cat_l1"] = items["cat_l1"].fillna("")
    items["cat_leaf"] = items["cat_leaf"].fillna("")
    for col in ("title_len", "n_also_bought", "n_also_viewed"):
        items[col] = items[col].fillna(0.0)

    out_dir = ensure_dir(cfg.processed_dir)
    inter_path = save_table(inter, out_dir, "interactions")
    items_path = save_table(items, out_dir, "items")
    LOGGER.info("wrote %s and %s", inter_path, items_path)
    return {"interactions": inter_path, "items": items_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--sample-users", type=int, default=None)
    args = parser.parse_args()

    cfg = default_config()
    if args.sample_users is not None:
        cfg.data.sample_users = None if args.sample_users <= 0 else args.sample_users
    run(cfg, raw_dir=args.raw_dir)


if __name__ == "__main__":
    main()
