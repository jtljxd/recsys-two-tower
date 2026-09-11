"""Shared helpers: seeding, device selection, logging, JSON IO."""

import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

LOGGER_NAME = "two_tower"


def set_seed(seed: int) -> None:
    """Seed every RNG we rely on so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(requested: Optional[str] = None) -> torch.device:
    """Pick a device: explicit request wins, else cuda -> mps -> cpu."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(name: str = LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


@contextmanager
def timed(message: str, logger: Optional[logging.Logger] = None):
    """Log how long a block took."""
    log = logger or get_logger()
    start = time.time()
    log.info("%s ...", message)
    yield
    log.info("%s done in %.1fs", message, time.time() - start)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Tabular IO
# ---------------------------------------------------------------------------
def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401

            return True
        except ImportError:
            return False


def table_path(directory: Path, stem: str) -> Path:
    """Resolve the on-disk path for a table, preferring parquet."""
    suffix = ".parquet" if _parquet_available() else ".pkl.gz"
    return directory / (stem + suffix)


def save_table(df, directory: Path, stem: str) -> Path:
    """Write a DataFrame, falling back to gzipped pickle without pyarrow."""
    ensure_dir(directory)
    path = table_path(directory, stem)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_pickle(path, compression="gzip")
    return path


def load_table(directory: Path, stem: str):
    """Read a table written by :func:`save_table`, accepting either format."""
    import pandas as pd

    parquet = directory / (stem + ".parquet")
    pickled = directory / (stem + ".pkl.gz")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if pickled.exists():
        return pd.read_pickle(pickled, compression="gzip")
    raise FileNotFoundError("no table named {!r} in {}".format(stem, directory))


def table_exists(directory: Path, stem: str) -> bool:
    return (directory / (stem + ".parquet")).exists() or (
        directory / (stem + ".pkl.gz")
    ).exists()
