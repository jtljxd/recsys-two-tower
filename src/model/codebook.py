"""Residual codebook that synchronises the two towers through a discrete index.

Adapted from a production pre-ranking model where the mechanism is called Cross
Tower Synchronization (CTS). The idea resolves the central tension of two tower
architectures: the towers must stay independent so item vectors can be computed
offline and served through ANN, yet the signal that matters most -- how a
particular user relates to a particular item -- is inherently a crossing.

CTS routes that crossing through a *discrete* medium. Cross features are
quantised to an index; both towers receive the same index but look it up in
their own codebook. They never exchange a continuous vector, so independence
survives: the index is a handful of bits and can be stored alongside the item
vector exactly like FIT's meta query index.

Three layers of residual quantisation give 4^3 = 64 distinct patterns from only
12 centroids. Quantising the residual at each level, rather than adding a fourth
independent codebook, is what buys the multiplicative rather than additive
capacity.

Two details are load-bearing and easy to get wrong:

1. Assignment is computed under stop-gradient. If distances were differentiable
   the encoder could move features toward whichever centroid is convenient,
   which collapses the partition rather than learning it.
2. Centroids are trained toward the mean of their assigned samples, also under
   stop-gradient. Without it the centroid and its samples chase each other and
   the codebook degenerates.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualCodebook(nn.Module):
    """Multi-level residual quantiser with per-side output embeddings.

    Args:
        in_dim: width of the vector being quantised.
        n_levels: number of residual levels.
        codebook_size: centroids per level.
        code_dim: width of the embedding each side retrieves for a code.
        dead_threshold: a centroid assigned fewer than this many samples in a
            batch is considered dead and is pulled toward a live sample instead
            of toward its own (nearly empty) mean.
    """

    def __init__(
        self,
        in_dim: int,
        n_levels: int = 3,
        codebook_size: int = 4,
        code_dim: int = 16,
        dead_threshold: int = 10,
        assign_weight: float = 1e-4,
        revive_weight: float = 1e-2,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.dead_threshold = dead_threshold
        self.assign_weight = assign_weight
        self.revive_weight = revive_weight

        # Centroids live in feature space and are trained only by the MSE loss.
        # They start at zero and are replaced by real data points on the first
        # forward pass; see _lazy_init for why zero init alone does not work.
        self.centroids = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(codebook_size, in_dim))
                for _ in range(n_levels)
            ]
        )
        self.register_buffer("initialised", torch.zeros((), dtype=torch.bool))

        # Output embeddings: same index, separate tables per side. This is the
        # whole point -- the towers agree on *which* pattern applies and each
        # decides what that pattern means for it.
        self.user_codes = nn.ModuleList(
            [nn.Embedding(codebook_size, code_dim) for _ in range(n_levels)]
        )
        self.item_codes = nn.ModuleList(
            [nn.Embedding(codebook_size, code_dim) for _ in range(n_levels)]
        )
        for table in list(self.user_codes) + list(self.item_codes):
            nn.init.normal_(table.weight, std=0.01)

        # Diagnostics only; never read by the forward pass.
        self.register_buffer(
            "last_counts", torch.zeros(n_levels, codebook_size)
        )

    @property
    def out_dim(self) -> int:
        return self.n_levels * self.code_dim

    @torch.no_grad()
    def _lazy_init(self, x: torch.Tensor) -> None:
        """Seed the centroids from real data, farthest-point style.

        Zero initialisation makes every distance a tie, so the whole batch is
        assigned to index 0 and the other centroids are dead. The revive path is
        supposed to rescue them, but it draws from the busiest cluster -- which,
        when everything is in one cluster, is the entire dataset. Each dead
        centroid then chases a random sample, and under a momentum optimiser
        those draws average to the data mean, i.e. exactly where centroid 0 is
        already heading. The four centroids move in lockstep and never separate.

        That failure is invisible in a toy test with well-separated clusters,
        where a single random draw is already informative. It shows up on real
        cross features, whose pairwise distance (~2.5) is small next to their
        norm (~4.8).

        Seeding with spread-out real points removes the tie at step 0, which is
        all the MSE updates need to take over.
        """
        for level in range(self.n_levels):
            pool = x if level == 0 else x - self.centroids[level - 1][
                torch.cdist(x, self.centroids[level - 1]).argmin(dim=-1)
            ]
            picks = [pool[torch.randint(len(pool), (1,)).item()]]
            for _ in range(self.codebook_size - 1):
                # Farthest point from what has been picked so far, so the
                # initial centroids span the data instead of clumping.
                d = torch.cdist(pool, torch.stack(picks)).min(dim=-1).values
                picks.append(pool[int(d.argmax())])
            self.centroids[level].copy_(torch.stack(picks))
        self.initialised.fill_(True)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """x (B, in_dim) -> (user_code, item_code, indices).

        Returns the concatenated per-level embeddings for each side, plus the
        raw indices so the caller can compute the codebook loss.
        """
        residual = x
        indices: List[torch.Tensor] = []
        u_parts, v_parts = [], []

        if not bool(self.initialised) and self.training:
            self._lazy_init(x.detach())

        for level in range(self.n_levels):
            # Assignment must not backpropagate into the features being
            # quantised; see the module docstring.
            q = residual.detach()
            dist = torch.cdist(q, self.centroids[level])  # (B, K)
            idx = dist.argmin(dim=-1)
            indices.append(idx)

            u_parts.append(self.user_codes[level](idx))
            v_parts.append(self.item_codes[level](idx))

            # Residual for the next level. Straight-through is deliberately not
            # used: gradients reach the code tables through the embedding
            # lookup, which is all the towers need.
            residual = residual - self.centroids[level][idx].detach()

        with torch.no_grad():
            for level, idx in enumerate(indices):
                self.last_counts[level] = torch.bincount(
                    idx, minlength=self.codebook_size
                ).float()

        return torch.cat(u_parts, dim=-1), torch.cat(v_parts, dim=-1), indices

    def codebook_loss(
        self, x: torch.Tensor, indices: List[torch.Tensor]
    ) -> torch.Tensor:
        """Pull each centroid toward the mean of the samples assigned to it.

        A centroid that wins almost nothing is dead weight, and in a residual
        scheme a dead level-1 centroid starves everything below it. Those are
        instead pulled toward a sample drawn from the busiest cluster, with a
        much larger weight so they actually relocate -- the same escape valve
        the production model uses.
        """
        loss = x.new_zeros(())
        residual = x.detach()

        for level in range(self.n_levels):
            idx = indices[level]
            counts = torch.bincount(idx, minlength=self.codebook_size)
            busiest = int(counts.argmax())
            donor_pool = residual[idx == busiest]

            for k in range(self.codebook_size):
                mask = idx == k
                n = int(mask.sum())
                centroid = self.centroids[level][k]

                if n >= self.dead_threshold:
                    target = residual[mask].mean(dim=0)
                    weight = self.assign_weight
                elif len(donor_pool):
                    # Relocate into the crowded region to share its load.
                    pick = torch.randint(
                        len(donor_pool), (1,), device=x.device
                    )
                    target = donor_pool[pick].squeeze(0)
                    weight = self.revive_weight
                else:
                    continue

                loss = loss + weight * 0.5 * F.mse_loss(
                    centroid, target.detach(), reduction="sum"
                )

            residual = residual - self.centroids[level][idx].detach()

        return loss

    def occupancy(self) -> List[List[int]]:
        """Per-level assignment counts from the last forward pass."""
        return [[int(c) for c in level] for level in self.last_counts]
