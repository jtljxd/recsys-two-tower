"""ItemQueryPerf (IQP) cross-interaction features for InteractRank.

The paper (Khandagale et al., WWW'25) estimates, from historical engagement
logs, the probability that an item is engaged in the context of a query:

    IQP_p(q) = P(p | q) = C(p, q) / C(q)

We have no search query. The role the query plays there -- a coarse marker of
the user's current intent -- is filled here by the user's top category, which
the feature pipeline already computes from *past* events only.

Two correctness constraints dominate this module, and both are enforced rather
than merely documented:

1. Statistics are accumulated from the training split only. A temporal split
   means anything later would leak "this item became popular" backwards into
   the test features.
2. A sample's own contribution is removed at lookup time. Without that, a
   positive's IQP is mechanically higher than a negative's and the model can
   score a near-perfect AUC by reading the label straight off the feature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Order is fixed: the model's affine layer indexes these positionally.
IQP_FEATURES = ("iqp_cat", "iqp_brand", "iqp_global", "iqp_cat_match")


@dataclass
class IQPTable:
    """Engagement counts gathered over the training split."""

    # C(item, top_cat) and C(item, top_brand), keyed by the pair.
    pair_cat: Dict[Tuple[int, str], int]
    pair_brand: Dict[Tuple[int, str], int]
    # C(top_cat) and C(top_brand): context totals.
    ctx_cat: Dict[str, int]
    ctx_brand: Dict[str, int]
    # C(item) and C(all): the global popularity prior.
    item_total: Dict[int, int]
    grand_total: int
    # item_id -> its own leaf-category id, for the explicit match signal.
    item_category: Dict[int, int]

    @property
    def n_items_seen(self) -> int:
        return len(self.item_total)


def build_iqp_table(
    frame: pd.DataFrame,
    item_category: Dict[int, int],
) -> IQPTable:
    """Accumulate IQP counts from the training split.

    Args:
        frame: the full sample frame; only ``split == "train"`` rows are read.
        item_category: item_id -> category string, for ``iqp_cat_match``.

    Only positive engagements count. The paper's C(p, q) is "how often item p
    was *engaged*", not how often it was shown, and our label already encodes
    exactly that distinction.
    """
    train = frame[(frame["split"] == "train") & (frame["label"] > 0)]

    pair_cat: Dict[Tuple[int, str], int] = {}
    pair_brand: Dict[Tuple[int, str], int] = {}
    ctx_cat: Dict[str, int] = {}
    ctx_brand: Dict[str, int] = {}
    item_total: Dict[int, int] = {}

    items = train["item_id"].to_numpy()
    cats = train["top_cat"].to_numpy()
    brands = train["top_brand"].to_numpy()

    for item, cat, brand in zip(items, cats, brands):
        item = int(item)
        pair_cat[(item, cat)] = pair_cat.get((item, cat), 0) + 1
        pair_brand[(item, brand)] = pair_brand.get((item, brand), 0) + 1
        ctx_cat[cat] = ctx_cat.get(cat, 0) + 1
        ctx_brand[brand] = ctx_brand.get(brand, 0) + 1
        item_total[item] = item_total.get(item, 0) + 1

    grand_total = int(len(train))

    LOGGER.info(
        "IQP table | %d train positives | %d items | %d cat contexts | %d brand contexts",
        grand_total,
        len(item_total),
        len(ctx_cat),
        len(ctx_brand),
    )
    return IQPTable(
        pair_cat=pair_cat,
        pair_brand=pair_brand,
        ctx_cat=ctx_cat,
        ctx_brand=ctx_brand,
        item_total=item_total,
        grand_total=grand_total,
        item_category=item_category,
    )


def lookup(
    table: IQPTable,
    item_ids: np.ndarray,
    top_cats: np.ndarray,
    top_brands: np.ndarray,
    exclude_self: np.ndarray,
    cat_id: int = 0,
) -> np.ndarray:
    """Compute the 4 IQP features for a batch of (item, context) pairs.

    Args:
        item_ids: (n,) candidate item ids.
        top_cats: (n,) user top category, the query stand-in.
        top_brands: (n,) user top brand.
        exclude_self: (n,) bool, True where this row contributed to the table
            (i.e. a training-split positive) and must be subtracted out.

    Returns:
        (n, 4) float32, matching ``IQP_FEATURES`` order.

    The ratios are ~1e-5, while the tower dot product runs to +/-5. Feeding raw
    probabilities into the affine layer would let the dot product drown them
    entirely, so the three count-based features are log1p-compressed onto a
    comparable scale. The match flag is already 0/1.
    """
    n = len(item_ids)
    out = np.zeros((n, len(IQP_FEATURES)), dtype=np.float32)

    for i in range(n):
        item = int(item_ids[i])
        cat = top_cats[i]
        brand = top_brands[i]
        # A training positive is inside every count it would look up, so remove
        # one from both numerator and denominator.
        self_n = 1 if exclude_self[i] else 0

        num = table.pair_cat.get((item, cat), 0) - self_n
        den = table.ctx_cat.get(cat, 0) - self_n
        out[i, 0] = _ratio(num, den)

        num = table.pair_brand.get((item, brand), 0) - self_n
        den = table.ctx_brand.get(brand, 0) - self_n
        out[i, 1] = _ratio(num, den)

        num = table.item_total.get(item, 0) - self_n
        den = table.grand_total - self_n
        out[i, 2] = _ratio(num, den)

        out[i, 3] = 1.0 if table.item_category.get(item, -1) == cat_id else 0.0

    return out


def _ratio(num: int, den: int) -> float:
    """C(p,q)/C(q) on a log scale.

    Both arguments can go negative once the self-contribution is removed from
    an otherwise-empty cell, which is the unseen case and must read as zero.
    Scaled by 1e4 before log1p so that typical values (~1e-5) land in a range
    the affine layer can actually use rather than being rounded to noise.
    """
    if num <= 0 or den <= 0:
        return 0.0
    return float(np.log1p(1e4 * num / den))


class IQPLookup:
    """Per-candidate IQP lookup, used inside the Dataset.

    Negatives are resampled every epoch, so these features cannot be
    precomputed onto the sample frame -- only the positive row would have a
    value. They are cheap enough to compute per batch instead.

    The context (top_cat / top_brand) comes from the *user's* row and is held
    fixed across that row's candidates; only the item varies. That mirrors the
    paper, where IQP is a lookup keyed by (item, query) with the query fixed
    for the request.
    """

    def __init__(self, table: IQPTable, cat_vocab, brand_vocab):
        self.table = table
        # The counts are keyed by the raw strings stored on the frame, but
        # SplitData hands us encoded ids, so keep a decode map.
        self.cat_decode = {int(v): k for k, v in cat_vocab.mapping.items()}
        self.brand_decode = {int(v): k for k, v in brand_vocab.mapping.items()}

    def __call__(
        self,
        items: np.ndarray,
        top_cat_id: int,
        top_brand_id: int,
        positive_in_table: bool,
    ) -> np.ndarray:
        """items (C,) -> (C, 4) float32.

        Args:
            positive_in_table: True when this row is a training positive, i.e.
                it contributed to the counts. Only the first candidate (the
                positive itself) gets its contribution removed; the sampled
                negatives were never part of this row's counts.
        """
        cat_id = int(top_cat_id)
        cat = self.cat_decode.get(cat_id, "")
        brand = self.brand_decode.get(int(top_brand_id), "")
        exclude = np.zeros(len(items), dtype=bool)
        if positive_in_table:
            exclude[0] = True
        return lookup(
            self.table,
            items,
            np.full(len(items), cat, dtype=object),
            np.full(len(items), brand, dtype=object),
            exclude,
            cat_id=cat_id,
        )

