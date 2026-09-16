"""Leak-free user-side temporal feature engineering.

The whole module exists to enforce one rule:

    features for the i-th event of a user are aggregated from events 0..i-1 only.

We do that structurally instead of with rolling windows: events are walked in
chronological order per user, features are emitted from the running accumulator
*before* the current event is folded in. There is no code path that can look at
the present or the future.

Amazon timestamps only have day resolution, so same-day events are ordered by
item id for determinism; an event never sees another event that sorts after it.
"""

import math
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config, default_config
from ..utils import ensure_dir, get_logger, load_table, save_table, timed

LOGGER = get_logger()

DAY = 86400.0

# Order matters: this is the column order of the dense matrix.
DENSE_FEATURES: Tuple[str, ...] = (
    "has_history",
    "hist_len_log",
    "avg_rating",
    "std_rating",
    "pos_ratio",
    "last_rating",
    "avg_review_len_log",
    "avg_summary_len_log",
    "avg_helpful_ratio",
    "helpful_missing_ratio",
    "days_since_last_log",
    "lifetime_span_log",
    "avg_gap_log",
    "recent_7d_cnt",
    "recent_30d_cnt",
    "recent_90d_cnt",
    "avg_price_log",
    "max_price_log",
    "std_price_log",
    "price_missing_ratio",
    "n_unique_cat_log",
    "n_unique_brand_log",
    "cat_entropy",
    "cat_repeat_ratio",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
)


@dataclass
class CatProfile:
    """Per-row snapshot of the user's category/brand history, in CSR form.

    Row i holds the counts accumulated over user u's events *strictly before*
    row i -- the same causal guarantee the dense features already carry, since
    both are read off the accumulator before ``update``.

    A dense (n_rows x n_categories) matrix would be far too large, and users
    touch only a handful of distinct categories each, so this stores the
    nonzeros only.
    """

    indptr: np.ndarray  # (N + 1,) int64
    cat_ids: np.ndarray  # (nnz,) int64 -- raw category strings, encoded later
    cat_counts: np.ndarray  # (nnz,) int32
    cat_ratings: np.ndarray  # (nnz,) float32, rating total for that category
    brand_indptr: np.ndarray  # (N + 1,) int64
    brand_ids: np.ndarray  # (nnz_b,) int64
    brand_counts: np.ndarray  # (nnz_b,) int32
    totals: np.ndarray  # (N,) int32 -- history length, the ratio denominator


@dataclass
class FeatureStore:
    """Everything the Dataset needs, already aligned row-by-row."""

    frame: pd.DataFrame  # user_id, item_id, timestamp, rating, label, split, top1_*
    hist_items: np.ndarray  # (N, max_seq_len) int64, left-padded with 0
    hist_lens: np.ndarray  # (N,) int64
    dense: np.ndarray  # (N, len(DENSE_FEATURES)) float32
    dense_names: Tuple[str, ...] = DENSE_FEATURES
    cat_profile: Optional[CatProfile] = None

    def __len__(self) -> int:
        return len(self.frame)


def _safe_log1p(x: float) -> float:
    """log1p that never blows up on negatives or NaN."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return math.log1p(max(0.0, float(x)))


def _entropy(counter: Counter, total: int) -> float:
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        p = c / total
        if p > 0:
            ent -= p * math.log(p)
    return ent


class _UserAccumulator:
    """Running aggregates over one user's *past* events."""

    __slots__ = (
        "n",
        "rating_sum",
        "rating_sq_sum",
        "pos_count",
        "last_rating",
        "review_len_sum",
        "summary_len_sum",
        "helpful_sum",
        "helpful_n",
        "first_ts",
        "last_ts",
        "ts_window",
        "price_sum",
        "price_sq_sum",
        "price_n",
        "price_max",
        "price_missing",
        "cat_counter",
        "brand_counter",
        "cat_rating_sum",
        "recent_items",
    )

    def __init__(self, max_seq_len: int):
        self.n = 0
        self.rating_sum = 0.0
        self.rating_sq_sum = 0.0
        self.pos_count = 0
        self.last_rating = 0.0
        self.review_len_sum = 0.0
        self.summary_len_sum = 0.0
        self.helpful_sum = 0.0
        self.helpful_n = 0
        self.first_ts: Optional[int] = None
        self.last_ts: Optional[int] = None
        self.ts_window: deque = deque()  # timestamps, for recency windows
        self.price_sum = 0.0
        self.price_sq_sum = 0.0
        self.price_n = 0
        self.price_max = 0.0
        self.price_missing = 0
        self.cat_counter: Counter = Counter()
        self.brand_counter: Counter = Counter()
        # Rating total per category, so the cross features can report how well
        # the user liked a category rather than only how often they touched it.
        self.cat_rating_sum: Counter = Counter()
        self.recent_items: deque = deque(maxlen=max_seq_len)

    # -- read side: only ever called before ``update`` for the same event ----
    def dense_vector(self, now_ts: int) -> List[float]:
        n = self.n
        if n == 0:
            # Cold start: everything zero, flagged by has_history=0.
            base = [0.0] * len(DENSE_FEATURES)
            dow, month = self._calendar(now_ts)
            base[DENSE_FEATURES.index("dow_sin")] = dow[0]
            base[DENSE_FEATURES.index("dow_cos")] = dow[1]
            base[DENSE_FEATURES.index("month_sin")] = month[0]
            base[DENSE_FEATURES.index("month_cos")] = month[1]
            return base

        avg_rating = self.rating_sum / n
        var_rating = max(0.0, self.rating_sq_sum / n - avg_rating ** 2)
        avg_helpful = self.helpful_sum / self.helpful_n if self.helpful_n else 0.0

        span_days = (self.last_ts - self.first_ts) / DAY if self.first_ts else 0.0
        gap_days = (now_ts - self.last_ts) / DAY if self.last_ts else 0.0
        avg_gap = span_days / n if n else 0.0

        r7 = self._window_count(now_ts, 7)
        r30 = self._window_count(now_ts, 30)
        r90 = self._window_count(now_ts, 90)

        if self.price_n:
            avg_price = self.price_sum / self.price_n
            var_price = max(0.0, self.price_sq_sum / self.price_n - avg_price ** 2)
        else:
            avg_price = 0.0
            var_price = 0.0

        total_cats = sum(self.cat_counter.values())
        top_cat_n = self.cat_counter.most_common(1)[0][1] if self.cat_counter else 0
        cat_repeat = top_cat_n / total_cats if total_cats else 0.0

        dow, month = self._calendar(now_ts)

        return [
            1.0,                                            # has_history
            _safe_log1p(n),                                 # hist_len_log
            avg_rating,                                     # avg_rating
            math.sqrt(var_rating),                          # std_rating
            self.pos_count / n,                             # pos_ratio
            self.last_rating,                               # last_rating
            _safe_log1p(self.review_len_sum / n),           # avg_review_len_log
            _safe_log1p(self.summary_len_sum / n),          # avg_summary_len_log
            avg_helpful,                                    # avg_helpful_ratio
            1.0 - self.helpful_n / n,                       # helpful_missing_ratio
            _safe_log1p(gap_days),                          # days_since_last_log
            _safe_log1p(span_days),                         # lifetime_span_log
            _safe_log1p(avg_gap),                           # avg_gap_log
            float(r7),                                      # recent_7d_cnt
            float(r30),                                     # recent_30d_cnt
            float(r90),                                     # recent_90d_cnt
            _safe_log1p(avg_price),                         # avg_price_log
            _safe_log1p(self.price_max),                    # max_price_log
            _safe_log1p(math.sqrt(var_price)),              # std_price_log
            self.price_missing / n,                         # price_missing_ratio
            _safe_log1p(len(self.cat_counter)),             # n_unique_cat_log
            _safe_log1p(len(self.brand_counter)),           # n_unique_brand_log
            _entropy(self.cat_counter, total_cats),         # cat_entropy
            cat_repeat,                                     # cat_repeat_ratio
            dow[0], dow[1], month[0], month[1],
        ]

    def top_category(self) -> str:
        return self.cat_counter.most_common(1)[0][0] if self.cat_counter else ""

    def top_brand(self) -> str:
        return self.brand_counter.most_common(1)[0][0] if self.brand_counter else ""

    def history(self) -> List[int]:
        return list(self.recent_items)

    # -- write side ---------------------------------------------------------
    def update(
        self,
        item_id: int,
        rating: float,
        ts: int,
        review_len: float,
        summary_len: float,
        helpful: float,
        price: float,
        category: str,
        brand: str,
        positive_threshold: float,
    ) -> None:
        self.n += 1
        self.rating_sum += rating
        self.rating_sq_sum += rating * rating
        self.last_rating = rating
        if rating >= positive_threshold:
            self.pos_count += 1
        self.review_len_sum += review_len
        self.summary_len_sum += summary_len
        if not (isinstance(helpful, float) and math.isnan(helpful)):
            self.helpful_sum += helpful
            self.helpful_n += 1

        if self.first_ts is None:
            self.first_ts = ts
        self.last_ts = ts
        self.ts_window.append(ts)

        if price is not None and not (isinstance(price, float) and math.isnan(price)):
            self.price_sum += price
            self.price_sq_sum += price * price
            self.price_n += 1
            self.price_max = max(self.price_max, price)
        else:
            self.price_missing += 1

        if category:
            self.cat_counter[category] += 1
            self.cat_rating_sum[category] += rating
        if brand:
            self.brand_counter[brand] += 1
        self.recent_items.append(item_id)

    # -- helpers ------------------------------------------------------------
    def _window_count(self, now_ts: int, days: int) -> int:
        cutoff = now_ts - days * DAY
        # ts_window is chronological; drop nothing, just count from the right.
        count = 0
        for ts in reversed(self.ts_window):
            if ts >= cutoff:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _calendar(ts: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Cyclical encoding of weekday and month."""
        dt = pd.Timestamp(ts, unit="s")
        dow = dt.dayofweek
        month = dt.month - 1
        return (
            (math.sin(2 * math.pi * dow / 7), math.cos(2 * math.pi * dow / 7)),
            (math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12)),
        )


def _item_lookup(items: pd.DataFrame) -> Dict[int, Tuple[float, str, str]]:
    """item_id -> (price, cat_leaf, brand) for history aggregation."""
    lookup = {}
    for row in items.itertuples(index=False):
        price = getattr(row, "price", np.nan)
        lookup[int(row.item_id)] = (
            float(price) if price is not None and not pd.isna(price) else float("nan"),
            str(getattr(row, "cat_leaf", "") or ""),
            str(getattr(row, "brand", "") or ""),
        )
    return lookup


def _assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out: last event -> test, second to last -> valid, rest -> train."""
    df = df.copy()
    # rank from the end: 0 == last event
    df["_rev_rank"] = df.groupby("user_id").cumcount(ascending=False)
    df["split"] = "train"
    df.loc[df["_rev_rank"] == 1, "split"] = "valid"
    df.loc[df["_rev_rank"] == 0, "split"] = "test"
    return df.drop(columns=["_rev_rank"])


def build(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    cfg: Optional[Config] = None,
) -> FeatureStore:
    cfg = cfg or default_config()
    max_seq_len = cfg.data.max_seq_len
    pos_threshold = cfg.data.positive_threshold

    # Deterministic chronological order; item_id breaks same-day ties.
    df = interactions.sort_values(["user_id", "timestamp", "item_id"]).reset_index(
        drop=True
    )
    df = _assign_splits(df)

    lookup = _item_lookup(items)
    n = len(df)
    hist_items = np.zeros((n, max_seq_len), dtype=np.int64)
    hist_lens = np.zeros(n, dtype=np.int64)
    dense = np.zeros((n, len(DENSE_FEATURES)), dtype=np.float32)
    top_cats: List[str] = [""] * n
    top_brands: List[str] = [""] * n

    # Per-row category/brand history snapshots for the InteractRank cross
    # features. Collected in the same place as ``dense``, i.e. before the
    # current event is folded in, so causality is inherited rather than
    # re-argued.
    cp_cats: List[str] = []
    cp_counts: List[int] = []
    cp_ratings: List[float] = []
    cp_indptr = np.zeros(n + 1, dtype=np.int64)
    cp_brands: List[str] = []
    cp_bcounts: List[int] = []
    cp_bindptr = np.zeros(n + 1, dtype=np.int64)
    cp_totals = np.zeros(n, dtype=np.int32)

    acc: Optional[_UserAccumulator] = None
    current_user = None
    log_every = max(100_000, n // 20)  # ~20 progress lines regardless of size
    start = time.time()

    for idx, row in enumerate(df.itertuples(index=False)):
        if row.user_id != current_user:
            current_user = row.user_id
            acc = _UserAccumulator(max_seq_len)

        if idx and idx % log_every == 0:
            elapsed = time.time() - start
            rate = idx / elapsed
            LOGGER.info(
                "  features %d/%d (%.1f%%) | %.0f rows/s | eta %.1f min",
                idx,
                n,
                100.0 * idx / n,
                rate,
                (n - idx) / rate / 60.0,
            )

        # ---- emit features from the PAST only -----------------------------
        dense[idx] = acc.dense_vector(int(row.timestamp))
        hist = acc.history()
        if hist:
            # right-aligned so the most recent item is always at the end
            hist_items[idx, max_seq_len - len(hist):] = hist
            hist_lens[idx] = len(hist)
        top_cats[idx] = acc.top_category()
        top_brands[idx] = acc.top_brand()

        for c, cnt in acc.cat_counter.items():
            cp_cats.append(c)
            cp_counts.append(cnt)
            cp_ratings.append(acc.cat_rating_sum[c])
        cp_indptr[idx + 1] = len(cp_cats)
        for b, cnt in acc.brand_counter.items():
            cp_brands.append(b)
            cp_bcounts.append(cnt)
        cp_bindptr[idx + 1] = len(cp_brands)
        cp_totals[idx] = acc.n

        # ---- only now does the current event become history ---------------
        price, cat, brand = lookup.get(int(row.item_id), (float("nan"), "", ""))
        acc.update(
            item_id=int(row.item_id),
            rating=float(row.rating),
            ts=int(row.timestamp),
            review_len=float(row.review_len),
            summary_len=float(row.summary_len),
            helpful=float(row.helpful_ratio)
            if not pd.isna(row.helpful_ratio)
            else float("nan"),
            price=price,
            category=cat,
            brand=brand,
            positive_threshold=pos_threshold,
        )

    frame = df[["user_id", "item_id", "timestamp", "rating", "split"]].copy()
    frame["label"] = (frame["rating"] >= pos_threshold).astype(np.float32)
    frame["top_cat"] = top_cats
    frame["top_brand"] = top_brands

    LOGGER.info(
        "built %d samples | train=%d valid=%d test=%d | positives=%.3f",
        n,
        int((frame["split"] == "train").sum()),
        int((frame["split"] == "valid").sum()),
        int((frame["split"] == "test").sum()),
        float(frame["label"].mean()),
    )
    return FeatureStore(
        frame=frame,
        hist_items=hist_items,
        hist_lens=hist_lens,
        dense=dense,
        cat_profile=CatProfile(
            indptr=cp_indptr,
            # Strings for now; encoders map them to ids once the vocab exists.
            cat_ids=np.array(cp_cats, dtype=object),
            cat_counts=np.array(cp_counts, dtype=np.int32),
            cat_ratings=np.array(cp_ratings, dtype=np.float32),
            brand_indptr=cp_bindptr,
            brand_ids=np.array(cp_brands, dtype=object),
            brand_counts=np.array(cp_bcounts, dtype=np.int32),
            totals=cp_totals,
        ),
    )


def save(store: FeatureStore, out_dir: Path) -> None:
    ensure_dir(out_dir)
    save_table(store.frame, out_dir, "samples")
    arrays = dict(
        hist_items=store.hist_items,
        hist_lens=store.hist_lens,
        dense=store.dense,
        dense_names=np.array(store.dense_names),
    )
    cp = store.cat_profile
    if cp is not None:
        arrays.update(
            cp_indptr=cp.indptr,
            cp_cat_ids=cp.cat_ids,
            cp_cat_counts=cp.cat_counts,
            cp_cat_ratings=cp.cat_ratings,
            cp_brand_indptr=cp.brand_indptr,
            cp_brand_ids=cp.brand_ids,
            cp_brand_counts=cp.brand_counts,
            cp_totals=cp.totals,
        )
    np.savez_compressed(out_dir / "user_features.npz", **arrays)


def load(out_dir: Path) -> FeatureStore:
    frame = load_table(out_dir, "samples")
    arrays = np.load(out_dir / "user_features.npz", allow_pickle=True)
    cp = None
    if "cp_indptr" in arrays:
        cp = CatProfile(
            indptr=arrays["cp_indptr"],
            cat_ids=arrays["cp_cat_ids"],
            cat_counts=arrays["cp_cat_counts"],
            cat_ratings=arrays["cp_cat_ratings"],
            brand_indptr=arrays["cp_brand_indptr"],
            brand_ids=arrays["cp_brand_ids"],
            brand_counts=arrays["cp_brand_counts"],
            totals=arrays["cp_totals"],
        )
    return FeatureStore(
        frame=frame,
        hist_items=arrays["hist_items"],
        hist_lens=arrays["hist_lens"],
        dense=arrays["dense"],
        dense_names=tuple(arrays["dense_names"].tolist()),
        cat_profile=cp,
    )


def run(cfg: Optional[Config] = None) -> FeatureStore:
    cfg = cfg or default_config()
    interactions = load_table(cfg.processed_dir, "interactions")
    items = load_table(cfg.processed_dir, "items")
    with timed("building user features", LOGGER):
        store = build(interactions, items, cfg)
    save(store, cfg.processed_dir)
    return store


if __name__ == "__main__":
    run()
