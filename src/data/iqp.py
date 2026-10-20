"""Cross-interaction features for InteractRank.

The paper (Khandagale et al., WWW'25) scores with an affine layer over the
tower dot product concatenated with cross-interaction features derived from
engagement logs:

    IQP_p(q) = P(p | q) = C(p, q) / C(q)

We have no search query. Section 2.3 of the paper allows the prior to be
conditioned on context, IQP_p(q, c) = P(p | q, c), and we push that to its
limit: the context is the individual user, and the "query" is the candidate
item's own category. So the features answer "how has this user behaved toward
this item's category in the past", plus one genuinely cross-user prior
(``xf_global``) to keep the paper's original flavour.

Two things this module must never do, both of which are enforced by tests:

1. Use the user's interaction with the candidate item itself. The negative
   sampler skips every item a user ever touched, so such a feature would be
   nonzero for positives and *identically* zero for negatives -- a perfect
   label indicator that would push AUC toward 1.0 while meaning nothing.
2. Read history that postdates the row. The per-row category snapshots are
   captured before the current event is folded into the accumulator, so this
   is inherited from the existing causality guarantee rather than re-argued.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .features import DENSE_FEATURES, USER_ATTR_COLS

LOGGER = logging.getLogger(__name__)

# Order is fixed: the model's affine layer indexes these positionally.
# First block: candidate-dependent cross behaviour.
# Second block: cohort priors, (user attribute value, candidate genre).
# Last block: the user's own statistics (identical for every candidate in a
# row), appended so the head can act as an IQP_p(q, c) prior conditioned on the
# user.
IQP_FEATURES = (
    "xf_cat_cnt",  # user's interactions across the candidate's genres
    "xf_cat_ratio",  # that as a share of the user's history
    "xf_cat_rating",  # mean rating the user gave across those genres
    "xf_brand_cnt",  # user's past interactions with the candidate's brand
    "xf_genre_iou",  # Jaccard overlap between user's and candidate's genres
    "xf_year_diff",  # |candidate year - user's mean watched year|, log1p
    "xf_global",  # item's global popularity prior (train split only)
    "xf_recent_cnt",  # last-≤max_seq_len events sharing a genre with candidate
    "xf_gender_genre",  # cohort: gender x candidate genre, train positive rate
    "xf_age_genre",  # cohort: age bucket x candidate genre
    "xf_occupation_genre",  # cohort: occupation x candidate genre
) + tuple("xf_user_" + n for n in DENSE_FEATURES)


@dataclass
class IQPTable:
    """Global item prior, accumulated over the training split only."""

    item_total: Dict[int, int]
    grand_total: int

    @property
    def n_items_seen(self) -> int:
        return len(self.item_total)


def build_iqp_table(frame: pd.DataFrame) -> IQPTable:
    """Accumulate the global popularity prior from training-split positives.

    Only positives count: the paper's C(p) is "how often item p was *engaged*",
    and our label already draws that line. Restricting to the training split is
    load-bearing -- with a temporal split, anything later would leak "this item
    became popular" backwards into the test features.
    """
    train = frame[(frame["split"] == "train") & (frame["label"] > 0)]
    items, counts = np.unique(train["item_id"].to_numpy(), return_counts=True)
    item_total = {int(i): int(c) for i, c in zip(items, counts)}
    grand_total = int(len(train))
    LOGGER.info(
        "IQP global prior | %d train positives over %d items",
        grand_total,
        len(item_total),
    )
    return IQPTable(item_total=item_total, grand_total=grand_total)


def _bits(genre_ids: np.ndarray) -> int:
    """Genre id set -> integer bitmask. Genres are few (<=3 per item, ~28 per
    user), so even Amazon's ~2000-deep vocab fits machine words; intersection
    is a single Python-int AND instead of numpy calls per candidate."""
    bits = 0
    for g in genre_ids:
        bits |= 1 << int(g)
    return bits


def build_attr_genre_table(frame: pd.DataFrame, encoders) -> Dict[str, np.ndarray]:
    """(attribute value, genre id) -> smoothed positive rate, TRAIN split only.

    Cohort priors for gender/age/occupation x candidate genre: for every
    attribute value and every genre in the vocab, the share of that (value,
    genre)'s training interactions that were positive, Laplace-smoothed
    (pos+1)/(tot+2). A female user seeing a drama candidate reads
    table[gender=2][drama]; the transfer across all users sharing the attribute
    is exactly what a per-user xf_cat_* cannot provide (no history in that
    genre yet). All stats come from the training split only, and the current
    sample's label is temporal-later than the training split by construction,
    so nothing leaks.

    Column 0 of each table holds the genre-agnostic positive rate of that
    attribute value (the model's fallback for OOV genres).
    """
    it = encoders.item_table
    if getattr(it, "genre_ids", None) is None or len(it.genre_ids) == 0:
        return {}
    indptr, ids = it.genre_indptr, it.genre_ids
    n_vocab = len(encoders.cat_leaf_vocab) + 1  # ids are 1-based; 0 = unknown
    train_ids = np.flatnonzero((frame["split"] == "train").to_numpy())
    item_ids = frame["item_id"].to_numpy()
    labels = frame["label"].to_numpy()

    def item_genres(item: int) -> np.ndarray:
        lo, hi = int(indptr[item]), int(indptr[item + 1])
        return ids[lo:hi]

    tables = {}
    for attr in USER_ATTR_COLS:
        if attr not in frame.columns:
            continue
        col = frame[attr].to_numpy()
        n_vals = int(np.nanmax(col)) + 1
        pos = np.zeros((n_vals + 1, n_vocab), dtype=np.float64)
        tot = np.zeros((n_vals + 1, n_vocab), dtype=np.float64)
        for i in train_ids:
            a = int(col[i])
            gs = item_genres(int(item_ids[i]))
            if a <= 0 or len(gs) == 0:
                continue
            pos[a, gs] += labels[i]
            tot[a, gs] += 1.0
        table = (pos + 1.0) / (tot + 2.0)
        # genre-agnostic prior per attribute value -> column 0.
        table[:, 0] = (pos.sum(axis=1) + 1.0) / (tot.sum(axis=1) + 2.0)
        tables[attr] = table.astype(np.float32)
        LOGGER.info(
            "IQP cohort prior | %s values=%d hit=%d of %d train rows",
            attr,
            n_vals,
            int((col[train_ids] > 0).sum()),
            len(train_ids),
        )
    return tables


class IQPLookup:
    """Per-candidate cross features, computed inside the Dataset.

    Negatives are resampled every epoch, so these cannot be precomputed onto
    the sample frame -- only the positive row would ever hold a value, and the
    model would read the label straight off that.
    """

    def __init__(
        self,
        table: IQPTable,
        cat_profile,
        item_cat_leaf_ids: np.ndarray,
        item_brand_ids: np.ndarray,
        cat_vocab,
        brand_vocab,
        item_genre_indptr=None,
        item_genre_ids=None,
        item_year=None,
        user_stats=None,
        attr_values=None,
        attr_tables=None,
    ):
        self.table = table
        self.cp = cat_profile
        # Candidate item -> its own category / brand id.
        self.item_cat = item_cat_leaf_ids
        self.item_brand = item_brand_ids
        # Multi-label genre sets (CSR over item ids). Datasets without them
        # (Amazon) store None and matching falls back to the single leaf id.
        self.genre_indptr = (
            item_genre_indptr if item_genre_indptr is not None else None
        )
        self.genre_ids = item_genre_ids
        self.item_year = item_year
        # (N, D) standardized user statistics; appended to the head's input so
        # the score can act as a user-conditioned prior.
        self.user_stats = (
            np.asarray(user_stats, dtype=np.float32)
            if user_stats is not None
            else None
        )
        # Per-row static user attributes and (attr value, genre) cohort tables.
        self.attr_values = attr_values or {}
        self.attr_tables = attr_tables or {}
        # The snapshots store raw strings; encode once here so the per-row
        # lookups compare integers.
        self.cat_ids = cat_vocab.encode_many(cat_profile.cat_ids)
        self.brand_ids = brand_vocab.encode_many(cat_profile.brand_ids)

    def _candidate_genres(self, item: int) -> np.ndarray:
        """Encoded id set of the candidate's genres; falls back to its leaf."""
        if self.genre_indptr is not None and item < len(self.genre_indptr) - 1:
            lo = int(self.genre_indptr[item])
            hi = int(self.genre_indptr[item + 1])
            if hi > lo:
                return np.asarray(self.genre_ids[lo:hi], dtype=np.int64)
        cat = int(self.item_cat[item]) if item < len(self.item_cat) else 0
        if cat > 0:
            return np.array([cat], dtype=np.int64)
        return np.zeros(0, dtype=np.int64)

    def __call__(self, row: int, items: np.ndarray) -> np.ndarray:
        """items (C,) -> (C, len(IQP_FEATURES)) float32.

        ``row`` indexes the *global* feature store, which is how the snapshots
        are keyed; SplitData carries that index so the mapping survives the
        split slicing.
        """
        cp = self.cp
        # 11 candidate-dependent cross features (8 behaviour + 3 cohort priors);
        # everything after is the user's own statistics, constant per row.
        n_cross = 11
        out = np.zeros((len(items), len(IQP_FEATURES)), dtype=np.float32)

        lo, hi = int(cp.indptr[row]), int(cp.indptr[row + 1])
        cats = self.cat_ids[lo:hi]
        counts = cp.cat_counts[lo:hi]
        ratings = cp.cat_ratings[lo:hi]
        blo, bhi = int(cp.brand_indptr[row]), int(cp.brand_indptr[row + 1])
        brands = self.brand_ids[blo:bhi]
        bcounts = cp.brand_counts[blo:bhi]
        total = int(cp.totals[row])
        year_n = int(cp.year_n[row])
        mean_year = float(cp.year_sum[row]) / year_n if year_n else 0.0
        rlo, rhi = int(cp.recent_indptr[row]), int(cp.recent_indptr[row + 1])
        # Bitmasks of each recent item's genres, computed once per row: every
        # candidate's recency test is then a bare Python-int AND, not a numpy
        # call per (candidate, recent item).
        recent_bits = []
        if rhi > rlo:
            for rid in cp.recent_ids[rlo:rhi]:
                rg = self._candidate_genres(int(rid))
                if len(rg):
                    recent_bits.append(_bits(rg))

        for j, item in enumerate(items):
            item = int(item)
            genre_ids = self._candidate_genres(item)
            brand = int(self.item_brand[item]) if item < len(self.item_brand) else 0

            # (0-2) category behaviour across ALL of the candidate's genres.
            match = np.flatnonzero(np.isin(cats, genre_ids)) if len(genre_ids) else np.zeros(0, dtype=np.int64)
            if len(match):
                c = int(counts[match].sum())
                r = float(ratings[match].sum())
                out[j, 0] = np.log1p(c)
                out[j, 1] = c / total if total else 0.0
                out[j, 2] = r / c if c else 0.0

            # (3) brand behaviour (structurally absent on MovieLens).
            bhit = np.flatnonzero(brands == brand)
            if len(bhit):
                out[j, 3] = np.log1p(int(bcounts[bhit[0]]))

            # (4) genre-set overlap: Jaccard(user genres, candidate genres).
            if len(genre_ids):
                user_set = len(np.unique(cats))
                cand_set = len(np.unique(genre_ids))
                inter = len(np.intersect1d(np.unique(cats), np.unique(genre_ids)))
                union = user_set + cand_set - inter
                out[j, 4] = inter / union if union else 0.0

            # (5) release-year preference gap.
            cand_year = self.item_year[item] if self.item_year is not None and item < len(self.item_year) else 0.0
            if year_n and cand_year > 0:
                out[j, 5] = float(np.log1p(abs(cand_year - mean_year)))

            # (6) global popularity prior.
            out[j, 6] = _global_prior(self.table, item)

            # (7) recency: how many of the user's last items share a genre.
            if recent_bits:
                cb = _bits(genre_ids)
                if cb:
                    hit = 0
                    for b in recent_bits:
                        if b & cb:
                            hit += 1
                    out[j, 7] = np.log1p(hit)

            # (8-10) cohort priors: (user attribute value, candidate genre).
            if len(genre_ids):
                for k, attr in enumerate(USER_ATTR_COLS):
                    t = self.attr_tables.get(attr)
                    av = self.attr_values.get(attr)
                    if t is None or av is None:
                        continue
                    a = int(av[row])
                    if a <= 0 or a >= t.shape[0]:
                        continue
                    out[j, 8 + k] = float(t[a, genre_ids].mean())

        # (8..) user statistics -- row-level, broadcast over all candidates.
        if self.user_stats is not None and row < len(self.user_stats):
            out[:, n_cross:] = self.user_stats[row]

        return out


def _global_prior(table: IQPTable, item: int) -> float:
    """C(p) / C(all) on a log scale.

    Raw values sit around 1e-5 while the tower dot product runs to +/-5, so a
    bare probability would be drowned by the dot product inside the affine
    layer. Scaled up before log1p to land on a usable range.
    """
    c = table.item_total.get(item, 0)
    if c <= 0 or table.grand_total <= 0:
        return 0.0
    return float(np.log1p(1e4 * c / table.grand_total))
