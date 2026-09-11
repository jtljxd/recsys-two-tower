"""Central configuration for the two-tower retrieval pipeline."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


@dataclass
class DataConfig:
    """Controls how interactions are built and split."""

    dataset: str = "movielens-100k"  # or "synthetic"
    # Ratings >= this value are treated as positive implicit feedback.
    positive_threshold: float = 4.0
    # Drop users/items with fewer than this many positive interactions.
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    # Leave-one-out split: the newest interaction per user becomes the test item,
    # the second newest becomes validation.
    holdout_strategy: str = "leave_one_out"
    # Only used when dataset == "synthetic".
    synthetic_users: int = 2000
    synthetic_items: int = 1000
    synthetic_interactions: int = 60000
    seed: int = 42


@dataclass
class ModelConfig:
    embedding_dim: int = 64
    tower_hidden: tuple = (128, 64)
    dropout: float = 0.1
    # L2-normalize tower outputs and scale logits by 1/temperature.
    normalize: bool = True
    temperature: float = 0.07
    # Subtract log(item_prob) from logits to correct in-batch sampling bias.
    use_logq_correction: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 1024
    epochs: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip: float = 5.0
    num_workers: int = 0
    eval_every: int = 1
    early_stop_patience: int = 3
    device: Optional[str] = None  # auto: cuda -> mps -> cpu
    seed: int = 42


@dataclass
class EvalConfig:
    ks: tuple = (10, 20, 50)
    # Exclude items the user already interacted with in train/val.
    filter_seen: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    artifact_dir: Path = ARTIFACT_DIR

    def to_dict(self) -> dict:
        d = asdict(self)
        d["artifact_dir"] = str(self.artifact_dir)
        return d


def default_config() -> Config:
    return Config()
