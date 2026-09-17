"""End-to-end training entrypoint.

    python -m src.train --raw-dir /path/to/amazon --sample-users 10000

Stages are cached: parsing and feature building are skipped when their parquet
outputs already exist, so re-running only repeats the model training.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config, default_config
from .data import encoders as encoders_mod
from .data import features as features_mod
from .data import parse_amazon
from .data.dataset import build_dataloaders
from .evaluate import evaluate, format_metrics
from .model.towers import InteractRankModel, OursModel, TwoTowerModel
from .utils import (
    count_parameters,
    ensure_dir,
    get_logger,
    load_table,
    resolve_device,
    save_json,
    set_seed,
    table_exists,
    timed,
)

LOGGER = get_logger()


def prepare_data(cfg: Config, raw_dir: Optional[str], force: bool = False):
    """Parse -> features -> encoders, reusing cached artifacts when possible."""
    processed = cfg.processed_dir

    if force or not table_exists(processed, "interactions"):
        parse_amazon.run(cfg, raw_dir=Path(raw_dir) if raw_dir else None)
    else:
        LOGGER.info("reusing cached interactions table in %s", processed)

    interactions = load_table(processed, "interactions")
    items = load_table(processed, "items")

    if force or not table_exists(processed, "samples"):
        with timed("building user features", LOGGER):
            store = features_mod.build(interactions, items, cfg)
        features_mod.save(store, processed)
    else:
        LOGGER.info("reusing cached samples table in %s", processed)
        store = features_mod.load(processed)

    with timed("fitting encoders", LOGGER):
        enc = encoders_mod.fit(store.frame, store.dense, items, interactions, cfg)
    encoders_mod.save(enc, processed)
    return store, enc


def build_model(enc, user_dense_dim: int, cfg: Config) -> TwoTowerModel:
    table = enc.item_table
    cls = TwoTowerModel
    if cfg.model.use_codebook:
        cls = OursModel
    elif cfg.model.use_interactrank:
        cls = InteractRankModel
    return cls(
        n_users=enc.n_users,
        n_items=table.n_items - 1,  # table has a padding row at index 0
        n_brands=enc.n_brands,
        n_cat_l1=enc.n_cat_l1,
        n_cat_leaf=enc.n_cat_leaf,
        user_dense_dim=user_dense_dim,
        item_dense=table.dense,
        item_brand_ids=table.brand_ids,
        item_cat_l1_ids=table.cat_l1_ids,
        item_cat_leaf_ids=table.cat_leaf_ids,
        cfg=cfg.model,
    )


def train_one_epoch(
    model: TwoTowerModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Config,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_elements = 0
    start = time.time()

    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        labels = batch["labels"]

        if cfg.train.loss_type == "bce":
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        elif cfg.train.loss_type == "softmax":
            # Column 0 is always the positive candidate.
            target = torch.zeros(
                logits.size(0), dtype=torch.long, device=logits.device
            )
            loss = F.cross_entropy(logits, target)
        else:
            raise ValueError("unknown loss_type: {}".format(cfg.train.loss_type))

        # Reported separately so train_loss stays comparable to the other
        # models regardless of how the codebook term is weighted.
        main_loss = float(loss.item())
        cb_loss = None
        if getattr(model, "codebook", None) is not None:
            cb = model.codebook_loss()
            cb_loss = float(cb.item())
            loss = loss + cfg.model.code_loss_weight * cb

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()

        total_loss += main_loss * labels.numel()
        total_elements += labels.numel()

        if cfg.train.log_every and step % cfg.train.log_every == 0:
            if cb_loss is not None:
                # Occupancy is the diagnostic that matters: if a level collapses
                # onto one or two centroids the codebook is oversized and its
                # capacity is wasted.
                occ = model.codebook.occupancy()
                LOGGER.info(
                    "epoch %d step %d/%d loss=%.4f cb=%.6f occ=%s",
                    epoch,
                    step,
                    len(loader),
                    main_loss,
                    cb_loss,
                    " ".join(str(lv) for lv in occ),
                )
            else:
                LOGGER.info(
                    "epoch %d step %d/%d loss=%.4f temp=%.3f",
                    epoch,
                    step,
                    len(loader),
                    main_loss,
                    float(model.temperature.item()),
                )

    LOGGER.info("epoch %d finished in %.1fs", epoch, time.time() - start)
    return total_loss / max(total_elements, 1)


def run(cfg: Optional[Config] = None, raw_dir: Optional[str] = None, force: bool = False) -> Dict:
    cfg = cfg or default_config()
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    LOGGER.info("device: %s", device)

    store, enc = prepare_data(cfg, raw_dir, force)
    train_loader, valid_loader, test_loader = build_dataloaders(store, enc, cfg)

    model = build_model(enc, store.dense.shape[1], cfg).to(device)
    LOGGER.info(
        "model: %s | parameters: %d",
        "ours" if cfg.model.use_codebook
        else ("interactrank" if cfg.model.use_interactrank else "base"),
        count_parameters(model),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.train.lr_scheduler_factor,
        patience=cfg.train.lr_scheduler_patience,
    )

    artifact_dir = ensure_dir(cfg.artifact_dir)
    best_score = -float("inf")
    best_state = None
    patience = 0
    history = []

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, cfg, epoch
        )
        valid_metrics = evaluate(
            model, valid_loader, device, cfg.eval.ks, cfg.eval.compute_gauc
        )
        LOGGER.info(
            "epoch %d | train_loss=%.4f | %s",
            epoch,
            train_loss,
            format_metrics(valid_metrics, "valid"),
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, **{"valid_" + k: v for k, v in valid_metrics.items()}}
        )

        score = valid_metrics.get(cfg.train.monitor, float("nan"))
        if np.isnan(score):
            score = -valid_metrics["loss"]
        scheduler.step(score)

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            torch.save(best_state, artifact_dir / "best_model.pt")
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                LOGGER.info("early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(
        model, test_loader, device, cfg.eval.ks, cfg.eval.compute_gauc
    )
    LOGGER.info("=" * 60)
    LOGGER.info(format_metrics(test_metrics, "TEST"))
    LOGGER.info("=" * 60)

    report = {
        "config": cfg.to_dict(),
        "history": history,
        "best_valid_{}".format(cfg.train.monitor): best_score,
        "test": test_metrics,
    }
    save_json(report, artifact_dir / "report.json")
    LOGGER.info("report written to %s", artifact_dir / "report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=str, default=None, help="dir with the .json.gz files")
    parser.add_argument("--sample-users", type=int, default=None, help="<=0 means all users")
    parser.add_argument(
        "--min-item-interactions",
        type=int,
        default=None,
        help="item core threshold; lower it when sampling few users",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--loss", type=str, default=None, choices=["bce", "softmax"])
    parser.add_argument("--force", action="store_true", help="ignore cached parquet files")
    parser.add_argument(
        "--model", type=str, default="base",
        choices=["base", "interactrank", "ours"],
    )
    parser.add_argument(
        "--artifact-dir", type=str, default=None,
        help="where to write report.json / best_model.pt",
    )
    args = parser.parse_args()

    cfg = default_config()
    cfg.model.use_codebook = args.model == "ours"
    # Ours builds on InteractRank, so it needs the same cross features.
    cfg.model.use_interactrank = args.model in ("interactrank", "ours")
    if args.artifact_dir is not None:
        cfg.artifact_dir = Path(args.artifact_dir)
    elif cfg.model.use_interactrank:
        # Keep each model's report separate by default so comparison runs do
        # not clobber each other.
        cfg.artifact_dir = Path(cfg.artifact_dir) / args.model
    if args.sample_users is not None:
        cfg.data.sample_users = None if args.sample_users <= 0 else args.sample_users
    if args.min_item_interactions is not None:
        cfg.data.min_item_interactions = args.min_item_interactions
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.device is not None:
        cfg.train.device = args.device
    if args.loss is not None:
        cfg.train.loss_type = args.loss

    run(cfg, raw_dir=args.raw_dir, force=args.force)


if __name__ == "__main__":
    main()
