"""The two-tower network.

Towers never see each other's inputs; they only meet at the final dot product.
That constraint is what makes item embeddings precomputable and ANN-servable,
so it is enforced structurally rather than by convention.

Scoring:
    logit = (u . v) / temperature + bias

L2-normalised vectors give a dot product in [-1, 1], which is far too narrow for
BCE to produce confident probabilities. Dividing by a learnable temperature
widens the range, and the scalar bias lets the model calibrate the base rate.

``FITModel`` (SIGIR'25) keeps the decoupling but relaxes *how* the towers meet:
a learnable meta query stands in for the candidate inside the user tower, and
the final comparison is a small learned network over a multi-head similarity
matrix rather than a bare dot product. Item vectors are still precomputable.
"""

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .fit import DINAttention, LightweightSimilarityScorer, MetaQueryModule


def _mlp(in_dim: int, hidden: Sequence[int], dropout: float) -> nn.Sequential:
    layers = []
    prev = in_dim
    for i, h in enumerate(hidden):
        layers.append(nn.Linear(prev, h))
        # No BN/activation after the final projection: the output feeds
        # straight into L2-normalisation.
        if i < len(hidden) - 1:
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        prev = h
    return nn.Sequential(*layers)


class DenseEncoder(nn.Module):
    """Lift a handful of scalars to embedding-comparable width."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UserTower(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_brands: int,
        n_cats: int,
        dense_dim: int,
        cfg: ModelConfig,
    ):
        super().__init__()
        d_id = cfg.id_embedding_dim
        d_side = cfg.side_embedding_dim

        self.user_emb = nn.Embedding(n_users + 1, d_id, padding_idx=0)
        # Behaviour sequence embedding; kept separate from the item tower's
        # table on purpose so the two towers stay independent.
        self.hist_item_emb = nn.Embedding(n_items + 1, d_id, padding_idx=0)
        self.top_cat_emb = nn.Embedding(n_cats + 1, d_side, padding_idx=0)
        self.top_brand_emb = nn.Embedding(n_brands + 1, d_side, padding_idx=0)
        self.dense_encoder = DenseEncoder(dense_dim, cfg.dense_hidden, cfg.dropout)

        in_dim = d_id * 2 + d_side * 2 + cfg.dense_hidden
        self.din = None
        if cfg.use_fit:
            # FIT: the meta query is concatenated as a feature *and* used as the
            # DIN target, replacing mean-pooling over the sequence.
            meta_dim = d_id + d_side * 2
            in_dim += meta_dim
            self.din = DINAttention(d_id, meta_dim, cfg.din_hidden)
        self.mlp = _mlp(in_dim, cfg.tower_hidden, cfg.dropout)
        self.normalize = cfg.normalize
        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (
            self.user_emb,
            self.hist_item_emb,
            self.top_cat_emb,
            self.top_brand_emb,
        ):
            nn.init.normal_(emb.weight, std=0.01)
            with torch.no_grad():
                emb.weight[0].fill_(0.0)

    def forward(
        self,
        user_id: torch.Tensor,
        hist_items: torch.Tensor,
        hist_len: torch.Tensor,
        dense: torch.Tensor,
        top_cat: torch.Tensor,
        top_brand: torch.Tensor,
        meta_query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raw_mask = hist_items != 0  # (B, L)

        if self.din is not None and meta_query is not None:
            # DIN weighted pooling against the meta query. This is the whole
            # reason MQM exists: a target-aware pooling that needs no candidate.
            seq_vec = self.din(self.hist_item_emb(hist_items), meta_query, raw_mask)
        else:
            # Masked mean-pooling over the behaviour sequence.
            mask = raw_mask.float().unsqueeze(-1)  # (B, L, 1)
            seq = self.hist_item_emb(hist_items) * mask
            denom = mask.sum(dim=1).clamp(min=1.0)
            seq_vec = seq.sum(dim=1) / denom

        parts = [
            self.user_emb(user_id),
            seq_vec,
            self.top_cat_emb(top_cat),
            self.top_brand_emb(top_brand),
            self.dense_encoder(dense),
        ]
        if meta_query is not None:
            parts.append(meta_query)
        out = self.mlp(torch.cat(parts, dim=-1))
        return F.normalize(out, p=2, dim=-1) if self.normalize else out


class ItemTower(nn.Module):
    def __init__(
        self,
        n_items: int,
        n_brands: int,
        n_cat_l1: int,
        n_cat_leaf: int,
        dense_dim: int,
        cfg: ModelConfig,
    ):
        super().__init__()
        d_id = cfg.id_embedding_dim
        d_side = cfg.side_embedding_dim

        self.item_emb = nn.Embedding(n_items + 1, d_id, padding_idx=0)
        self.brand_emb = nn.Embedding(n_brands + 1, d_side, padding_idx=0)
        self.cat_l1_emb = nn.Embedding(n_cat_l1 + 1, d_side, padding_idx=0)
        self.cat_leaf_emb = nn.Embedding(n_cat_leaf + 1, d_side, padding_idx=0)
        self.dense_encoder = DenseEncoder(dense_dim, cfg.dense_hidden, cfg.dropout)

        in_dim = d_id + d_side * 3 + cfg.dense_hidden
        self.mlp = _mlp(in_dim, cfg.tower_hidden, cfg.dropout)
        self.normalize = cfg.normalize
        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.item_emb, self.brand_emb, self.cat_l1_emb, self.cat_leaf_emb):
            nn.init.normal_(emb.weight, std=0.01)
            with torch.no_grad():
                emb.weight[0].fill_(0.0)

    def attribute_features(
        self, item_id: torch.Tensor, brand: torch.Tensor, cat_l1: torch.Tensor
    ) -> torch.Tensor:
        """e_c: concatenated item attribute embeddings feeding MQM (Eq. 4).

        Training-only. At inference the query index is read from the item store,
        so these embeddings are not needed to serve a request -- which is what
        keeps the decoupling intact.
        """
        return torch.cat(
            [self.item_emb(item_id), self.brand_emb(brand), self.cat_l1_emb(cat_l1)],
            dim=-1,
        )

    def forward(
        self,
        item_id: torch.Tensor,
        brand: torch.Tensor,
        cat_l1: torch.Tensor,
        cat_leaf: torch.Tensor,
        dense: torch.Tensor,
    ) -> torch.Tensor:
        parts = [
            self.item_emb(item_id),
            self.brand_emb(brand),
            self.cat_l1_emb(cat_l1),
            self.cat_leaf_emb(cat_leaf),
            self.dense_encoder(dense),
        ]
        out = self.mlp(torch.cat(parts, dim=-1))
        return F.normalize(out, p=2, dim=-1) if self.normalize else out


class TwoTowerModel(nn.Module):
    """Owns both towers plus the item side-feature lookup tables."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_brands: int,
        n_cat_l1: int,
        n_cat_leaf: int,
        user_dense_dim: int,
        item_dense: np.ndarray,
        item_brand_ids: np.ndarray,
        item_cat_l1_ids: np.ndarray,
        item_cat_leaf_ids: np.ndarray,
        cfg: Optional[ModelConfig] = None,
    ):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg

        self.user_tower = UserTower(
            n_users, n_items, n_brands, n_cat_leaf, user_dense_dim, cfg
        )
        self.item_tower = ItemTower(
            n_items, n_brands, n_cat_l1, n_cat_leaf, item_dense.shape[1], cfg
        )

        # Static item side features live as buffers so .to(device) moves them.
        self.register_buffer(
            "item_dense", torch.from_numpy(item_dense.astype(np.float32))
        )
        self.register_buffer(
            "item_brand", torch.from_numpy(item_brand_ids.astype(np.int64))
        )
        self.register_buffer(
            "item_cat_l1", torch.from_numpy(item_cat_l1_ids.astype(np.int64))
        )
        self.register_buffer(
            "item_cat_leaf", torch.from_numpy(item_cat_leaf_ids.astype(np.int64))
        )

        # Parameterised in log space so temperature can never hit zero.
        log_t = math.log(cfg.temperature)
        if cfg.learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(log_t))
        else:
            self.register_buffer("log_temperature", torch.tensor(log_t))
        self.bias = (
            nn.Parameter(torch.zeros(1)) if cfg.use_global_bias else None
        )

    @property
    def temperature(self) -> torch.Tensor:
        # Clamped so the logit scale stays in a trainable range.
        return self.log_temperature.clamp(math.log(0.01), math.log(1.0)).exp()

    def encode_user(self, batch) -> torch.Tensor:
        return self.user_tower(
            batch["user_id"],
            batch["hist_items"],
            batch["hist_len"],
            batch["dense"],
            batch["top_cat"],
            batch["top_brand"],
        )

    def encode_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        """item_ids may be any shape; side features are gathered by index."""
        flat = item_ids.reshape(-1)
        vec = self.item_tower(
            flat,
            self.item_brand[flat],
            self.item_cat_l1[flat],
            self.item_cat_leaf[flat],
            self.item_dense[flat],
        )
        return vec.view(*item_ids.shape, -1)

    def forward(self, batch) -> torch.Tensor:
        """Returns logits shaped like batch['items']: (B, n_candidates)."""
        u = self.encode_user(batch)  # (B, D)
        v = self.encode_item(batch["items"])  # (B, C, D)
        scores = torch.einsum("bd,bcd->bc", u, v)
        logits = scores / self.temperature
        if self.bias is not None:
            logits = logits + self.bias
        return logits


class FITModel(TwoTowerModel):
    """Fully Interacted Two-tower model (SIGIR'25).

    Two changes over the base model, both preserving tower decoupling:

    * MQM feeds the user tower a meta query standing in for the candidate item,
      enabling DIN pooling over the behaviour sequence.
    * LSS replaces the dot product with row/column-wise FC layers over the
      multi-head similarity matrix.

    Because the user representation now depends on which meta query is selected,
    the user tower runs once per candidate. That is exactly what the paper
    describes: at serving time all N hard queries are evaluated in parallel and
    the stored query index picks the right one, so per-request cost stays flat in
    the candidate count.

    Temperature and the global bias are unused here: LSS emits an unbounded
    scalar and calibrates itself.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.cfg
        d_id = cfg.id_embedding_dim
        d_side = cfg.side_embedding_dim

        # e_c = [item_id || brand || cat_l1]; must match attribute_features().
        meta_dim = d_id + d_side * 2
        self.mqm = MetaQueryModule(
            item_feature_dim=meta_dim,
            meta_size=cfg.meta_size,
            temp_threshold=cfg.meta_temp_threshold,
            temp_min=cfg.meta_temp_min,
        )
        tower_out = cfg.tower_hidden[-1]
        self.lss = LightweightSimilarityScorer(
            user_dim=tower_out,
            item_dim=tower_out,
            n_heads_user=cfg.lss_heads_user,
            n_heads_item=cfg.lss_heads_item,
            head_dim=cfg.lss_head_dim,
            out_dim=cfg.lss_out_dim,
        )

    def item_attributes(self, item_ids: torch.Tensor) -> torch.Tensor:
        flat = item_ids.reshape(-1)
        feats = self.item_tower.attribute_features(
            flat, self.item_brand[flat], self.item_cat_l1[flat]
        )
        return feats.view(*item_ids.shape, -1)

    def encode_user_per_candidate(
        self, batch, meta_query: torch.Tensor
    ) -> torch.Tensor:
        """Run the user tower once per candidate. meta_query: (B, C, M)."""
        b, c = meta_query.shape[:2]

        def rep(t: torch.Tensor) -> torch.Tensor:
            # (B, ...) -> (B*C, ...) matching the flattened candidate axis.
            return t.unsqueeze(1).expand(b, c, *t.shape[1:]).reshape(b * c, *t.shape[1:])

        u = self.user_tower(
            rep(batch["user_id"]),
            rep(batch["hist_items"]),
            rep(batch["hist_len"]),
            rep(batch["dense"]),
            rep(batch["top_cat"]),
            rep(batch["top_brand"]),
            meta_query.reshape(b * c, -1),
        )
        return u.view(b, c, -1)

    def forward(self, batch, hard: Optional[bool] = None) -> torch.Tensor:
        items = batch["items"]  # (B, C)

        # Soft query while training, hard query at inference -- the annealed
        # temperature is what makes the two agree by the end of training.
        #
        # Defaulting off self.training rather than to a literal matters: the
        # shared evaluate() calls model(batch) positionally for every model, so
        # if the default were False the evaluation would silently score with the
        # soft query and the whole annealing mechanism would be wasted.
        if hard is None:
            hard = not self.training
        query, _ = self.mqm(self.item_attributes(items), hard=hard)

        u = self.encode_user_per_candidate(batch, query)  # (B, C, D)
        v = self.encode_item(items)  # (B, C, D)

        z_u = self.lss.user_heads(u)  # (B, C, H_u, p)
        z_v = self.lss.item_heads(v)  # (B, C, H_v, p)
        return self.lss(z_u, z_v)

    def query_similarity(self, batch) -> torch.Tensor:
        """Soft/hard query cosine similarity -- the paper's QS diagnostic."""
        return self.mqm.query_similarity(self.item_attributes(batch["items"]))
