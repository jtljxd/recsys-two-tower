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

LOGGER = logging.getLogger(__name__)

# Order is fixed: the model's affine layer indexes these positionally.
IQP_FEATURES = (
    "xf_cat_cnt",  # user's past interactions in the candidate's category
    "xf_cat_ratio",  # that count as a share of the user's history
    "xf_cat_rating",  # mean rating the user gave in that category
    "xf_brand_cnt",  # user's past interactions with the candidate's brand
    "xf_global",  # item's global popularity prior (train split only)
)


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
    ):
        self.table = table
        self.cp = cat_profile
        # Candidate item -> its own category / brand id.
        self.item_cat = item_cat_leaf_ids
        self.item_brand = item_brand_ids
        # The snapshots store raw strings; encode once here so the per-row
        # lookups compare integers.
        self.cat_ids = cat_vocab.encode_many(cat_profile.cat_ids)
        self.brand_ids = brand_vocab.encode_many(cat_profile.brand_ids)

    def __call__(self, row: int, items: np.ndarray) -> np.ndarray:
        """items (C,) -> (C, 5) float32.

        ``row`` indexes the *global* feature store, which is how the snapshots
        are keyed; SplitData carries that index so the mapping survives the
        split slicing.
        """
        cp = self.cp
        out = np.zeros((len(items), len(IQP_FEATURES)), dtype=np.float32)

        lo, hi = int(cp.indptr[row]), int(cp.indptr[row + 1])
        cats = self.cat_ids[lo:hi]
        counts = cp.cat_counts[lo:hi]
        ratings = cp.cat_ratings[lo:hi]
        blo, bhi = int(cp.brand_indptr[row]), int(cp.brand_indptr[row + 1])
        brands = self.brand_ids[blo:bhi]
        bcounts = cp.brand_counts[blo:bhi]
        total = int(cp.totals[row])

        for j, item in enumerate(items):
            item = int(item)
            cat = int(self.item_cat[item]) if item < len(self.item_cat) else 0
            brand = int(self.item_brand[item]) if item < len(self.item_brand) else 0

            hit = np.flatnonzero(cats == cat)
            if len(hit):
                k = hit[0]
                c = int(counts[k])
                out[j, 0] = np.log1p(c)
                out[j, 1] = c / total if total else 0.0
                out[j, 2] = float(ratings[k]) / c if c else 0.0

            bhit = np.flatnonzero(brands == brand)
            if len(bhit):
                out[j, 3] = np.log1p(int(bcounts[bhit[0]]))

            out[j, 4] = _global_prior(self.table, item)

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
