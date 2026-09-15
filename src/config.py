"""Central configuration for the Amazon-Electronics two-tower pipeline."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


@dataclass
class DataConfig:
    """Controls raw parsing, sampling and train/valid/test split."""

    dataset: str = "amazon-electronics"

    # --- raw files (override on the dev box if they live elsewhere) ---
    reviews_file: str = "reviews_Electronics_5.json.gz"
    meta_file: str = "meta_Electronics.json.gz"

    # --- sampling: keep the pipeline light for the first run ---
    # Number of users to sample. None == use every user.
    sample_users: Optional[int] = 10000
    # Users must have at least this many interactions to be usable.
    # leave-one-out needs >= 1 train + 1 valid + 1 test.
    min_user_interactions: int = 5
    min_item_interactions: int = 5

    # Ratings >= threshold are positive implicit feedback.
    positive_threshold: float = 4.0

    # --- behaviour sequence ---
    max_seq_len: int = 20

    # --- negative sampling ---
    # Negatives per positive during training.
    train_negatives: int = 4
    # Negatives per positive during evaluation (1 pos + N neg ranking).
    eval_negatives: int = 99
    # Sampling distribution over items: pop ** alpha. 0 == uniform.
    neg_sampling_alpha: float = 0.75

    # --- vocabulary pruning ---
    min_brand_freq: int = 5
    min_category_freq: int = 5

    seed: int = 42


@dataclass
class ModelConfig:
    id_embedding_dim: int = 64
    side_embedding_dim: int = 32
    dense_hidden: int = 64
    tower_hidden: Tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.1
    # L2-normalize tower outputs; logits are scaled by 1 / temperature.
    normalize: bool = True
    temperature: float = 0.07
    learnable_temperature: bool = True
    # Global scalar bias so normalized dot products can calibrate.
    use_global_bias: bool = True

    # --- DAT (Dual Augmented Two-tower, RecSys'21) ---
    # Each side gets a learnable vector trained to mimic the *other* tower's
    # output for positive pairs. It is fed into its own tower as a plain input,
    # so the towers stay independent and item vectors remain precomputable.
    use_dat: bool = False
    # Weight of the mimic (augmented) loss added to the main objective.
    dat_weight: float = 0.1
    # Stop-gradient on the mimic target. Turning this off lets the towers chase
    # the augmented vectors back and the pair can collapse; keep it on.
    dat_detach: bool = True


@dataclass
class TrainConfig:
    # "bce" -> pointwise with explicit negatives (gives AUC / logloss).
    # "softmax" -> in-batch sampled softmax (retrieval oriented).
    loss_type: str = "bce"
    batch_size: int = 1024
    eval_batch_size: int = 2048
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip: float = 5.0
    num_workers: int = 0
    early_stop_patience: int = 3
    # Metric watched for early stopping / LR schedule.
    monitor: str = "auc"
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 1
    device: Optional[str] = None  # auto: cuda -> mps -> cpu
    seed: int = 42
    log_every: int = 50


@dataclass
class EvalConfig:
    ks: Tuple[int, ...] = (10, 20, 50)
    compute_gauc: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR
    artifact_dir: Path = ARTIFACT_DIR

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("raw_dir", "processed_dir", "artifact_dir"):
            d[key] = str(getattr(self, key))
        return d


def default_config() -> Config:
    return Config()
