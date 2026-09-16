"""FIT building blocks: Meta Query Module, DIN attention, Lightweight Similarity Scorer.

Reference: "A Learnable Fully Interacted Two-Tower Model for Pre-Ranking System"
(Xiong & Yu et al., SIGIR'25, arXiv:2509.12948).

The point of MQM is subtle and worth stating plainly, because it is what keeps
the architecture servable: we want the *user* tower to run a DIN-style attention
over the behaviour sequence, which normally requires the target item -- exactly
the dependency a two-tower model cannot afford. MQM sidesteps it by learning a
small matrix of item cluster centroids. A candidate item is matched to those
centroids by softmax, and the resulting vector (not the item itself) is what the
user tower consumes.

At inference the match collapses to an argmax, and the winning index is stored
alongside the precomputed item vector. The query group itself is derived from
the meta matrix alone, so no item features are needed at request time and item
vectors remain precomputable. Storage overhead is the index only.
"""

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaQueryModule(nn.Module):
    """Learnable item meta matrix + parameter-free self-attention.

    Produces a ``soft query`` (softmax-weighted, used for training) and a
    ``hard query`` (argmax-selected, used for inference) that stands in for the
    candidate item inside the user tower.
    """

    def __init__(
        self,
        item_feature_dim: int,
        meta_size: int,
        temp_threshold: int,
        temp_min: float = 0.001,
    ):
        super().__init__()
        self.meta_size = meta_size
        self.temp_threshold = max(int(temp_threshold), 1)
        self.temp_min = temp_min

        # Kaiming-uniform per the paper: better numerical stability than normal
        # init here, since the matrix is consumed by a dot product against
        # concatenated item embeddings rather than by a linear layer.
        self.meta = nn.Parameter(torch.empty(meta_size, item_feature_dim))
        nn.init.kaiming_uniform_(self.meta, a=math.sqrt(5))

        # Global step drives temperature annealing. Registered as a buffer so it
        # survives checkpointing and moves with .to(device).
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    @property
    def temperature(self) -> float:
        """Anneal 1.0 -> temp_min so training ends up consistent with argmax.

        High temperature early keeps the softmax flat, which updates many meta
        vectors per step and lets the matrix converge; low temperature late makes
        the soft query approach the hard query actually used at inference.
        """
        step = float(self.global_step.item())
        return max(min(1.0, 1.0 - step / self.temp_threshold), self.temp_min)

    def query_group(self) -> torch.Tensor:
        """Q* = (Q Qᵀ) Q -- parameter-free self-attention over the meta matrix.

        Computable from the meta matrix alone, which is precisely why the item
        tower is not needed at inference time.
        """
        return (self.meta @ self.meta.t()) @ self.meta

    def forward(
        self, item_features: torch.Tensor, hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map item features to a meta query.

        Args:
            item_features: (..., D) concatenated item attribute embeddings.
            hard: use argmax selection (inference) instead of softmax weighting.

        Returns:
            (query, index) where query is (..., D) and index is (...,).
        """
        flat = item_features.reshape(-1, item_features.size(-1))
        group = self.query_group()  # (N, D)

        # Similarity against the *raw* meta matrix, as in Eq. 6.
        logits = flat @ self.meta.t() / self.temperature  # (B, N)
        index = logits.argmax(dim=-1)

        if hard:
            query = group[index]
        else:
            query = F.softmax(logits, dim=-1) @ group

        out_shape = item_features.shape[:-1]
        return query.view(*out_shape, -1), index.view(*out_shape)

    def query_similarity(self, item_features: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between soft and hard query (paper's QS metric).

        Should approach 1.0 as the temperature decays; that is the diagnostic
        for train/inference consistency.
        """
        soft, _ = self.forward(item_features, hard=False)
        hard, _ = self.forward(item_features, hard=True)
        return F.cosine_similarity(soft, hard, dim=-1)


class DINAttention(nn.Module):
    """Target-attention weighted pooling over the behaviour sequence.

    The target is the meta query, never a real candidate item vector, so the user
    tower stays free of item-tower dependencies.
    """

    def __init__(self, emb_dim: int, target_dim: int, hidden: Sequence[int]):
        super().__init__()
        # DIN convention: [target, seq, target - seq, target * seq].
        # The difference and product terms are what let the net express
        # "how close is this behaviour to the target" cheaply.
        self.proj = (
            nn.Linear(target_dim, emb_dim) if target_dim != emb_dim else nn.Identity()
        )
        layers = []
        prev = emb_dim * 4
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, seq: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """seq (B, L, E), target (B, T), mask (B, L) -> (B, E)."""
        tgt = self.proj(target).unsqueeze(1).expand(-1, seq.size(1), -1)
        feats = torch.cat([tgt, seq, tgt - seq, tgt * seq], dim=-1)
        scores = self.mlp(feats).squeeze(-1)  # (B, L)

        # Padded positions must not receive weight. Using -inf before softmax
        # rather than zeroing after keeps the weights a true distribution.
        scores = scores.masked_fill(~mask, float("-inf"))
        # A row with no history at all would produce all -inf -> NaN; detect and
        # zero those rows instead.
        empty = ~mask.any(dim=1, keepdim=True)
        weights = F.softmax(scores, dim=-1)
        weights = torch.where(empty, torch.zeros_like(weights), weights)
        return torch.einsum("bl,ble->be", weights, seq)


class LightweightSimilarityScorer(nn.Module):
    """Row-wise then column-wise FC layers over the multi-head similarity matrix.

    Replaces the dot product. Proven (via LITE) to be a universal approximator of
    continuous scoring functions, unlike the hand-crafted sum-max reduction, and
    cheaper than flattening the matrix into one big FC layer.

    Note the towers still encode independently -- only the *comparison* changes,
    so item vectors remain precomputable.
    """

    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        n_heads_user: int,
        n_heads_item: int,
        head_dim: int,
        out_dim: int,
        scale: float = 0.07,
    ):
        super().__init__()
        self.n_heads_user = n_heads_user
        self.n_heads_item = n_heads_item
        self.head_dim = head_dim

        self.user_proj = nn.Linear(user_dim, n_heads_user * head_dim)
        self.item_proj = nn.Linear(item_dim, n_heads_item * head_dim)

        # The towers emit L2-normalised vectors, so every entry of the similarity
        # matrix lands in a narrow band around zero (std ~0.09 at init). Feeding
        # that straight into the FC stack leaves the logits with std ~0.007 --
        # sigmoid maps the whole batch into [0.39, 0.40] and there is almost no
        # gradient to learn from. The base model solves the identical problem by
        # dividing the dot product by a temperature; do the same here, on the
        # matrix, before the FC layers see it.
        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / scale)))

        # BatchNorm keeps the FC inputs centred, which matters because ReLU
        # otherwise zeroes the ~50% of entries that are negative.
        self.row_norm = nn.BatchNorm1d(n_heads_user * n_heads_item)

        self.row_fc = nn.Linear(n_heads_item, out_dim)
        self.col_fc = nn.Linear(n_heads_user, out_dim)
        self.out = nn.Linear(out_dim * out_dim, 1)

        # Default init scales weights by 1/sqrt(fan_in); with out_dim^2 inputs
        # that shrinks the logits back down by roughly the factor the scale above
        # just bought us. Initialise the projection to unit gain instead, so the
        # logit spread at step 0 is comparable to the base model's dot/temperature.
        nn.init.normal_(self.out.weight, std=1.0 / out_dim)
        nn.init.zeros_(self.out.bias)

    def user_heads(self, h_u: torch.Tensor) -> torch.Tensor:
        """(B, d) -> (B, H_u, p)"""
        return self.user_proj(h_u).view(*h_u.shape[:-1], self.n_heads_user, self.head_dim)

    def item_heads(self, h_v: torch.Tensor) -> torch.Tensor:
        """(..., d) -> (..., H_v, p)"""
        return self.item_proj(h_v).view(*h_v.shape[:-1], self.n_heads_item, self.head_dim)

    def forward(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        """z_u (B, C, H_u, p), z_v (B, C, H_v, p) -> logits (B, C).

        Broadcast to a candidate axis so the 1-positive + N-negative evaluation
        batches score in one pass.
        """
        # Similarity matrix S: (B, C, H_u, H_v)
        sim = torch.einsum("bcup,bcvp->bcuv", z_u, z_v)

        # Widen the range so the downstream sigmoid can express confidence, then
        # centre it so ReLU does not discard half the entries.
        sim = sim * self.log_scale.clamp(max=math.log(1e4)).exp()
        shape = sim.shape
        sim = self.row_norm(sim.reshape(-1, shape[-2] * shape[-1])).view(shape)

        # Row-wise FC: mixes over item heads.
        s1 = F.relu(self.row_fc(sim))  # (B, C, H_u, out)
        # Column-wise FC: mixes over user heads.
        s2 = F.relu(self.col_fc(s1.transpose(-1, -2)))  # (B, C, out, out)
        return self.out(s2.flatten(start_dim=-2)).squeeze(-1)
