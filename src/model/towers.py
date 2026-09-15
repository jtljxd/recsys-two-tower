"""The two-tower network.

Towers never see each other's inputs; they only meet at the final dot product.
That constraint is what makes item embeddings precomputable and ANN-servable,
so it is enforced structurally rather than by convention.

Scoring:
    logit = (u . v) / temperature + bias

L2-normalised vectors give a dot product in [-1, 1], which is far too narrow for
BCE to produce confident probabilities. Dividing by a learnable temperature
widens the range, and the scalar bias lets the model calibrate the base rate.

With ``use_dat`` the model additionally learns one augmented vector per user and
per item (DAT, Yu et al. RecSys'21). Each is trained to mimic the *opposite*
tower's output on positive pairs, but is fed into its *own* tower as a plain
lookup. Information about the other side therefore reaches a tower through a
static parameter rather than a runtime dependency, so the constraint above is
untouched.
"""

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


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
        # DAT: the augmented vector is just another input feature.
        if cfg.use_dat:
            in_dim += cfg.tower_hidden[-1]
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
        aug: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Masked mean-pooling over the behaviour sequence.
        mask = (hist_items != 0).float().unsqueeze(-1)  # (B, L, 1)
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
        if aug is not None:
            parts.append(aug)
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
        if cfg.use_dat:
            in_dim += cfg.tower_hidden[-1]
        self.mlp = _mlp(in_dim, cfg.tower_hidden, cfg.dropout)
        self.normalize = cfg.normalize
        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.item_emb, self.brand_emb, self.cat_l1_emb, self.cat_leaf_emb):
            nn.init.normal_(emb.weight, std=0.01)
            with torch.no_grad():
                emb.weight[0].fill_(0.0)

    def forward(
        self,
        item_id: torch.Tensor,
        brand: torch.Tensor,
        cat_l1: torch.Tensor,
        cat_leaf: torch.Tensor,
        dense: torch.Tensor,
        aug: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [
            self.item_emb(item_id),
            self.brand_emb(brand),
            self.cat_l1_emb(cat_l1),
            self.cat_leaf_emb(cat_leaf),
            self.dense_encoder(dense),
        ]
        if aug is not None:
            parts.append(aug)
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

        # --- DAT augmented vectors -------------------------------------
        # a_u is looked up by user_id, a_v by item_id, and each is trained to
        # mimic the *opposite* tower's output on positive pairs. At inference
        # they are static parameters, so tower independence is preserved and
        # item vectors stay precomputable for ANN serving.
        if cfg.use_dat:
            d_aug = cfg.tower_hidden[-1]
            self.aug_user_emb = nn.Embedding(n_users + 1, d_aug, padding_idx=0)
            self.aug_item_emb = nn.Embedding(n_items + 1, d_aug, padding_idx=0)
            for emb in (self.aug_user_emb, self.aug_item_emb):
                nn.init.normal_(emb.weight, std=0.01)
                with torch.no_grad():
                    emb.weight[0].fill_(0.0)
        else:
            self.aug_user_emb = None
            self.aug_item_emb = None

    @property
    def use_dat(self) -> bool:
        return self.aug_user_emb is not None

    @property
    def temperature(self) -> torch.Tensor:
        # Clamped so the logit scale stays in a trainable range.
        return self.log_temperature.clamp(math.log(0.01), math.log(1.0)).exp()

    def encode_user(self, batch) -> torch.Tensor:
        aug = self.aug_user_emb(batch["user_id"]) if self.use_dat else None
        return self.user_tower(
            batch["user_id"],
            batch["hist_items"],
            batch["hist_len"],
            batch["dense"],
            batch["top_cat"],
            batch["top_brand"],
            aug,
        )

    def encode_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        """item_ids may be any shape; side features are gathered by index."""
        flat = item_ids.reshape(-1)
        aug = self.aug_item_emb(flat) if self.use_dat else None
        vec = self.item_tower(
            flat,
            self.item_brand[flat],
            self.item_cat_l1[flat],
            self.item_cat_leaf[flat],
            self.item_dense[flat],
            aug,
        )
        return vec.view(*item_ids.shape, -1)

    def forward(self, batch) -> torch.Tensor:
        """Returns logits shaped like batch['items']: (B, n_candidates)."""
        logits, _ = self.forward_with_aux(batch)
        return logits

    def forward_with_aux(self, batch):
        """Logits plus the tensors the DAT mimic loss needs.

        ``aux`` is empty unless DAT is enabled. Column 0 of ``batch['items']``
        is always the positive candidate (guaranteed by both TrainDataset and
        EvalDataset), which is what makes the positives-only mimic target a
        simple slice.
        """
        u = self.encode_user(batch)  # (B, D)
        v = self.encode_item(batch["items"])  # (B, C, D)
        scores = torch.einsum("bd,bcd->bc", u, v)
        logits = scores / self.temperature
        if self.bias is not None:
            logits = logits + self.bias

        if not self.use_dat:
            return logits, {}

        aux = {
            "u": u,
            "v_pos": v[:, 0, :],
            "a_u": self.aug_user_emb(batch["user_id"]),
            "a_v_pos": self.aug_item_emb(batch["items"][:, 0]),
        }
        return logits, aux

    def dat_loss(self, aux) -> torch.Tensor:
        """Mean squared error between each augmented vector and the opposite
        tower's positive-pair output.

        The target is detached on purpose: without it the mimic term also pulls
        the towers toward the augmented vectors, and since both are free
        parameters the pair can collapse onto a trivial solution. The augmented
        vector chases the tower, never the other way round.
        """
        if not aux:
            return torch.zeros((), device=self.item_dense.device)

        def target(t: torch.Tensor) -> torch.Tensor:
            return t.detach() if self.cfg.dat_detach else t

        loss_u = F.mse_loss(aux["a_u"], target(aux["v_pos"]))
        loss_v = F.mse_loss(aux["a_v_pos"], target(aux["u"]))
        return loss_u + loss_v
