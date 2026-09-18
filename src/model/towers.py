"""The two-tower network.

Towers never see each other's inputs; they only meet at the final dot product.
That constraint is what makes item embeddings precomputable and ANN-servable,
so it is enforced structurally rather than by convention.

Scoring:
    logit = (u . v) / temperature + bias

L2-normalised vectors give a dot product in [-1, 1], which is far too narrow for
BCE to produce confident probabilities. Dividing by a learnable temperature
widens the range, and the scalar bias lets the model calibrate the base rate.
"""

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..data.iqp import IQP_FEATURES
from .codebook import ResidualCodebook


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
        self.mlp = _mlp(in_dim, cfg.tower_hidden, cfg.dropout)
        # The code is injected after the MLP rather than concatenated onto its
        # input. Concatenating forces the whole tower to run one row per
        # (user, candidate) pair, and the dropout inside it then draws a fresh
        # mask for each -- so one user's positive and negatives get different
        # user vectors. Projecting the code onto the output instead lets the
        # shared part keep a single dropout draw per user.
        self.code_fc = (
            nn.Linear(cfg.n_code_levels * cfg.code_dim, cfg.tower_hidden[-1])
            if cfg.use_codebook
            else None
        )
        if self.code_fc is not None:
            # Default init makes this branch 1.7x the trunk's norm, which drowns
            # the tower. Start it as a small perturbation and let the gain in
            # ResidualCodebook grow it if the code earns the room.
            nn.init.normal_(self.code_fc.weight, std=0.01)
            nn.init.zeros_(self.code_fc.bias)
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
        code: Optional[torch.Tensor] = None,
        expand: int = 1,
    ) -> torch.Tensor:
        """expand > 1 repeats each row that many times *after* the
        candidate-independent features are computed.

        The code varies per candidate, so the tower has to emit one row per
        (user, candidate) pair. Feeding it a pre-expanded batch instead is not
        equivalent: dropout samples a fresh mask per row, so the same user ends
        up with a different vector for their positive than for each negative.
        Measured spread across the 5 copies of one user was 0.5265, against
        components of order 0.1 in the L2-normalised output. BCE compares one
        user's scores across candidates, so that noise lands directly on the
        quantity being ranked -- which is why AUC suffered while logloss did
        not.

        Computing the shared part once and expanding afterwards keeps a single
        dropout draw per user, exactly as the non-codebook models see.
        """
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
        out = self.mlp(torch.cat(parts, dim=-1))
        if expand > 1:
            out = out.repeat_interleave(expand, dim=0)
        if code is not None:
            out = out + self.code_fc(code)
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
        # Symmetric with the user side; see UserTower.
        self.code_fc = (
            nn.Linear(cfg.n_code_levels * cfg.code_dim, cfg.tower_hidden[-1])
            if cfg.use_codebook
            else None
        )
        if self.code_fc is not None:
            # Default init makes this branch 1.7x the trunk's norm, which drowns
            # the tower. Start it as a small perturbation and let the gain in
            # ResidualCodebook grow it if the code earns the room.
            nn.init.normal_(self.code_fc.weight, std=0.01)
            nn.init.zeros_(self.code_fc.bias)
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
        code: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [
            self.item_emb(item_id),
            self.brand_emb(brand),
            self.cat_l1_emb(cat_l1),
            self.cat_leaf_emb(cat_leaf),
            self.dense_encoder(dense),
        ]
        out = self.mlp(torch.cat(parts, dim=-1))
        if code is not None:
            out = out + self.code_fc(code)
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


class InteractRankModel(TwoTowerModel):
    """Two tower + IQP cross-interaction features (Khandagale et al., WWW'25).

    The paper's whole contribution to the scoring function is an affine layer
    over the dot product concatenated with precomputed cross-interaction
    features:

        score = W . [u.v, IQP_1, ..., IQP_N] + b

    Tower independence is fully preserved -- the IQP values are an offline
    lookup keyed by (item, context), not a runtime function of the other tower,
    which is exactly why the paper can claim ~1.1x serving FLOPs against
    IntTower's 24x.

    The affine layer replaces the base model's temperature and bias, which were
    doing the same job (scaling and shifting the dot product) with fewer
    degrees of freedom.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_iqp = len(IQP_FEATURES)
        self.interact = nn.Linear(1 + n_iqp, 1)
        # Start at the base model's behaviour: dot/temperature passed through,
        # cross features contributing nothing. The model then learns how much
        # weight the IQP signals deserve, rather than being perturbed away from
        # a working solution at step 0.
        with torch.no_grad():
            self.interact.weight.zero_()
            self.interact.weight[0, 0] = 1.0 / float(self.cfg.temperature)
            self.interact.bias.zero_()

    def forward(self, batch) -> torch.Tensor:
        u = self.encode_user(batch)
        v = self.encode_item(batch["items"])
        dot = torch.einsum("bd,bcd->bc", u, v)

        iqp = batch.get("iqp")
        if iqp is None:
            raise KeyError(
                "InteractRank needs the 'iqp' batch key; build the dataloaders "
                "with cfg.model.use_interactrank=True"
            )
        feats = torch.cat([dot.unsqueeze(-1), iqp], dim=-1)  # (B, C, 1+N)
        return self.interact(feats).squeeze(-1)


class OursModel(InteractRankModel):
    """InteractRank plus a residual codebook synchronising the two towers.

    The affine layer over [dot product, cross features] is inherited unchanged,
    so anything this model gains over InteractRank is attributable to the
    codebook alone.

    The cross features are lifted to a wider space before quantisation. Five
    scalars, three of which are strongly correlated, do not carry enough
    structure for a 64-way partition to latch onto; the projection gives the
    centroids room to separate patterns.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.cfg
        n_cross = len(IQP_FEATURES)
        self.code_proj = nn.Sequential(
            nn.Linear(n_cross, cfg.code_proj_dim),
            nn.ReLU(),
        )
        self.codebook = ResidualCodebook(
            in_dim=cfg.code_proj_dim,
            n_levels=cfg.n_code_levels,
            codebook_size=cfg.codebook_size,
            code_dim=cfg.code_dim,
            dead_threshold=cfg.code_dead_threshold,
            code_gain=cfg.code_gain,
        )
        self._last_indices = None
        self._last_proj = None

    def forward(self, batch) -> torch.Tensor:
        iqp = batch.get("iqp")
        if iqp is None:
            raise KeyError(
                "OursModel needs the 'iqp' batch key; build the dataloaders "
                "with cfg.model.use_interactrank=True"
            )
        items = batch["items"]
        B, C = items.shape

        # One code per (row, candidate): the crossing varies with the candidate,
        # so a single code per row would throw away exactly the signal we want.
        proj = self.code_proj(iqp.reshape(B * C, -1))
        u_code, v_code, indices = self.codebook(proj)
        # Stashed for the training loop, which needs them for the codebook loss
        # and the occupancy log.
        self._last_proj = proj
        self._last_indices = indices

        u = self.user_tower(
            batch["user_id"],
            batch["hist_items"],
            batch["hist_len"],
            batch["dense"],
            batch["top_cat"],
            batch["top_brand"],
            code=u_code,
            expand=C,
        ).view(B, C, -1)

        flat = items.reshape(-1)
        v = self.item_tower(
            flat,
            self.item_brand[flat],
            self.item_cat_l1[flat],
            self.item_cat_leaf[flat],
            self.item_dense[flat],
            code=v_code,
        ).view(B, C, -1)

        dot = torch.einsum("bcd,bcd->bc", u, v)
        feats = torch.cat([dot.unsqueeze(-1), iqp], dim=-1)
        return self.interact(feats).squeeze(-1)

    def codebook_loss(self) -> torch.Tensor:
        if self._last_indices is None:
            return torch.zeros((), device=self.item_dense.device)
        return self.codebook.codebook_loss(self._last_proj, self._last_indices)
