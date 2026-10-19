"""Parse MovieLens-1M into the same interactions/items parquet schema as Amazon.

MovieLens 1M ships as three ``.dat`` files with ``::``-separated columns:

    ratings.dat : UserID::MovieID::Rating::Timestamp
    movies.dat  : MovieID::Title::Genres     (genres pipe-separated)
    users.dat   : UserID::Gender::Age::Occupation::Zip-code   (unused)

It maps onto the Amazon pipeline without changing the downstream code:

  - every rating is an interaction event; ordering by timestamp is native;
  - rating >= 4 (positive_threshold) is positive implicit feedback;
  - genres are treated as the item's category (``cat_leaf``), so the
    category-based cross features (top_cat, IQP cat features, codebook) still
    have real signal;
  - fields Amazon has that MovieLens does not are filled neutrally: no review
    text (lengths 0), no helpful votes (NaN), no price/sales rank (NaN),
    no brand (empty string).

The output schemas (column names, dtypes) are identical to parse_amazon's, so
``features.build`` and ``encoders.fit`` never know which dataset they came from.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import Config, default_config
from ..utils import ensure_dir, get_logger, save_table, set_seed, timed
from .parse_amazon import _adaptive_core_filter, _assign_ids

LOGGER = get_logger()


def _load_ratings(path: Path) -> pd.DataFrame:
    """UserID::MovieID::Rating::Timestamp -> tidy interactions frame."""
    rows = []
    with path.open("rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("::")
            if len(parts) != 4:
                continue
            uid, mid, rating, ts = parts
            try:
                rows.append(
                    (
                        uid.strip(),
                        mid.strip(),
                        float(rating),
                        int(ts),
                    )
                )
            except ValueError:
                continue
    df = pd.DataFrame(
        rows,
        columns=["user_raw", "item_raw", "rating", "timestamp"],
    )
    # MovieLens has no review text or helpful votes; downstream code reads
    # these columns, so emit them empty rather than omitting them.
    df["review_len"] = 0.0
    df["summary_len"] = 0.0
    df["helpful_ratio"] = np.nan
    return df


def _load_movies(path: Path) -> pd.DataFrame:
    """MovieID::Title::Genres -> items frame (no brand, no price)."""
    rows = []
    with path.open("rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("::")
            if len(parts) < 2 or not parts[0].strip().isdigit():
                continue
            mid = parts[0].strip()
            title = parts[1].strip()
            genres = parts[2].strip() if len(parts) > 2 else ""
            first_genre = genres.split("|")[0] if genres else ""
            rows.append(
                (
                    mid,
                    "",
                    first_genre,
                    first_genre,  # genres are flat; l1 == leaf here
                    np.nan,       # price
                    np.nan,       # sales_rank
                    len(title.split()),
                    0,            # n_also_bought
                    0,            # n_also_viewed
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


def run(cfg: Optional[Config] = None, raw_dir: Optional[Path] = None) -> Dict[str, Path]:
    cfg = cfg or default_config()
    set_seed(cfg.data.seed)
    raw_dir = Path(raw_dir) if raw_dir else cfg.raw_dir
    ratings_path = raw_dir / "ratings.dat"
    movies_path = raw_dir / "movies.dat"
    for p in (ratings_path, movies_path):
        if not p.exists():
            raise FileNotFoundError(
                "missing MovieLens-1M file: {}\nPoint --raw-dir at the directory "
                "containing ratings.dat and movies.dat.".format(p)
            )

    with timed("loading ratings.dat", LOGGER):
        inter = _load_ratings(ratings_path)
    with timed("loading movies.dat", LOGGER):
        meta = _load_movies(movies_path)
    LOGGER.info(
        "raw: %d interactions, %d users, %d movies",
        len(inter),
        inter["user_raw"].nunique(),
        inter["item_raw"].nunique(),
    )

    inter = inter.sort_values(["user_raw", "timestamp"]).drop_duplicates(
        ["user_raw", "item_raw"], keep="first"
    )
    inter = _adaptive_core_filter(
        inter,
        cfg.data.min_user_interactions,
        cfg.data.min_item_interactions,
        sampled=cfg.data.sample_users is not None,
    )
    LOGGER.info(
        "after core filter: %d interactions, %d users, %d movies",
        len(inter),
        inter["user_raw"].nunique(),
        inter["item_raw"].nunique(),
    )
    if inter.empty:
        raise RuntimeError(
            "no interactions survived filtering.\n"
            "Either use the full dataset (--sample-users 0) or lower "
            "min_item_interactions in config.py."
        )

    inter = _assign_ids(inter)
    movie_ids = inter[["item_raw", "item_id"]].drop_duplicates().reset_index(drop=True)
    items = movie_ids.merge(meta, on="item_raw", how="left")
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
    parser.add_argument("--min-item-interactions", type=int, default=None)
    args = parser.parse_args()

    cfg = default_config()
    if args.sample_users is not None:
        cfg.data.sample_users = None if args.sample_users <= 0 else args.sample_users
    if args.min_item_interactions is not None:
        cfg.data.min_item_interactions = args.min_item_interactions
    run(cfg, raw_dir=Path(args.raw_dir) if args.raw_dir else None)


if __name__ == "__main__":
    main()