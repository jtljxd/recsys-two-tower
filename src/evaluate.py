"""Metrics: AUC, GAUC, logloss and top-K ranking quality.

AUC is computed with the rank-based formula (equivalent to the Mann-Whitney U
statistic) so we never materialise the O(n^2) pair matrix.
"""

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import get_logger

LOGGER = get_logger()
EPS = 1e-7


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC; returns NaN when one class is missing."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((labels > 0.5).sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within tied score groups, otherwise ties inflate AUC.
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rank_sum = ranks[labels > 0.5].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def group_auc(
    labels: np.ndarray, scores: np.ndarray, groups: np.ndarray
) -> float:
    """Per-user AUC averaged with weights proportional to group size.

    Users whose candidate list is all-positive or all-negative carry no ranking
    signal and are skipped, which is the standard GAUC convention.
    """
    order = np.argsort(groups, kind="mergesort")
    labels, scores, groups = labels[order], scores[order], groups[order]
    boundaries = np.flatnonzero(np.diff(groups)) + 1
    total_weight = 0.0
    weighted = 0.0
    for chunk_labels, chunk_scores in zip(
        np.split(labels, boundaries), np.split(scores, boundaries)
    ):
        auc = roc_auc(chunk_labels, chunk_scores)
        if np.isnan(auc):
            continue
        weight = len(chunk_labels)
        weighted += auc * weight
        total_weight += weight
    return float(weighted / total_weight) if total_weight else float("nan")


def ranking_metrics(
    scores: np.ndarray, ks: Sequence[int]
) -> Dict[str, float]:
    """scores is (N, C) with the positive item always at column 0."""
    positive = scores[:, [0]]
    # rank = how many candidates outscore the positive (ties count as half)
    greater = (scores[:, 1:] > positive).sum(axis=1)
    equal = (scores[:, 1:] == positive).sum(axis=1)
    rank = greater + equal / 2.0 + 1.0

    out: Dict[str, float] = {"mrr": float(np.mean(1.0 / rank))}
    n_candidates = scores.shape[1]
    for k in ks:
        if k >= n_candidates:
            continue
        hit = rank <= k
        out["recall@{}".format(k)] = float(hit.mean())
        out["ndcg@{}".format(k)] = float(
            np.mean(np.where(hit, 1.0 / np.log2(rank + 1.0), 0.0))
        )
    return out


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    ks: Sequence[int] = (10, 20, 50),
    compute_gauc: bool = True,
) -> Dict[str, float]:
    model.eval()
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_users: List[np.ndarray] = []
    loss_sum = 0.0
    n_elements = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        labels = batch["labels"]
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
        loss_sum += float(loss.item())
        n_elements += labels.numel()

        all_scores.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        all_users.append(
            batch["user_id"].detach().cpu().numpy().repeat(labels.shape[1])
        )

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    users = np.concatenate(all_users)

    flat_scores = scores.reshape(-1)
    flat_labels = labels.reshape(-1)

    metrics: Dict[str, float] = {
        "loss": loss_sum / max(n_elements, 1),
        "auc": roc_auc(flat_labels, flat_scores),
    }
    if compute_gauc:
        metrics["gauc"] = group_auc(flat_labels, flat_scores, users)

    probs = 1.0 / (1.0 + np.exp(-np.clip(flat_scores, -30, 30)))
    metrics["logloss"] = float(
        -np.mean(
            flat_labels * np.log(probs + EPS)
            + (1 - flat_labels) * np.log(1 - probs + EPS)
        )
    )
    metrics.update(ranking_metrics(scores, ks))
    return metrics


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    keys = ["loss", "auc", "gauc", "logloss", "mrr"]
    keys += [k for k in sorted(metrics) if k not in keys]
    parts = [
        "{}={:.4f}".format(k, metrics[k]) for k in keys if k in metrics
    ]
    return (prefix + " " if prefix else "") + " ".join(parts)
