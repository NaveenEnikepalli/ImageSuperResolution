"""
Configuration module defining dataclass schemas, YAML parsing, validation, and auto-saving.
"""

from dataclasses import dataclass, field, is_dataclass, fields, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, get_origin
import yaml

from training.utils.constants import (
    SUPPORTED_DATASETS,
    SUPPORTED_SCALES,
    DEFAULT_EXPERIMENT_NAME
)


@dataclass
class ModelConfig:
    """Model architecture configuration properties."""
    type: str = "teacher"  # options: "teacher", "student"
    name: str = "edsr_baseline"  # options: "edsr_baseline", "edsr_small", "rfdn", "imdn"
    num_blocks: int = 16
    num_rfdb_blocks: int = 3
    num_channels: int = 64
    scale: int = 4

    # Generic model structure parameters
    in_channels: int = 3
    kernel_size: int = 3
    padding: int = 1
    stride: int = 1
    bias: bool = True
    activation: str = "leaky_relu"
    negative_slope: float = 0.05
    distilled_channels: int = 24
    initial_residual_scale: float = 1.0

    def validate(self) -> None:
        """Validate model config properties."""
        if self.type not in ("teacher", "student"):
            raise ValueError(f"Model type must be 'teacher' or 'student', got '{self.type}'")
        if self.scale not in SUPPORTED_SCALES:
            raise ValueError(f"Model scale {self.scale} not supported. Must be in {SUPPORTED_SCALES}")
        if self.num_blocks <= 0:
            raise ValueError(f"Model num_blocks must be positive, got {self.num_blocks}")
        if self.num_rfdb_blocks <= 0:
            raise ValueError(f"Model num_rfdb_blocks must be positive, got {self.num_rfdb_blocks}")
        if self.num_channels <= 0:
            raise ValueError(f"Model num_channels must be positive, got {self.num_channels}")
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")
        if self.kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {self.kernel_size}")
        if self.padding < 0:
            raise ValueError(f"padding must be non-negative, got {self.padding}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if self.negative_slope < 0.0:
            raise ValueError(f"negative_slope must be non-negative, got {self.negative_slope}")
        if self.distilled_channels <= 0:
            raise ValueError(f"distilled_channels must be positive, got {self.distilled_channels}")
        if self.distilled_channels >= self.num_channels:
            raise ValueError(
                f"distilled_channels ({self.distilled_channels}) must be strictly less than "
                f"num_channels ({self.num_channels}) for complementary split configuration."
            )
        if self.initial_residual_scale < 0.0:
            raise ValueError(f"initial_residual_scale must be non-negative, got {self.initial_residual_scale}")


@dataclass
class DatasetConfig:
    """Dataset filepaths, processing, and pipeline configs."""
    name: str = "DIV2K"
    train_hr_path: str = "datasets/DIV2K/DIV2K_train_HR"
    train_lr_path: Optional[str] = None
    val_datasets: Dict[str, str] = field(default_factory=lambda: {
        "Set5": "datasets/Set5",
        "Set14": "datasets/Set14"
    })
    patch_size: int = 192
    batch_size: int = 16
    num_workers: int = 4

    def validate(self) -> None:
        """Validate dataset pipeline properties."""
        if self.name not in SUPPORTED_DATASETS:
            raise ValueError(f"Dataset name '{self.name}' must be one of {SUPPORTED_DATASETS}")
        if self.patch_size <= 0 or self.patch_size % 2 != 0:
            raise ValueError(f"Dataset patch_size must be positive and even, got {self.patch_size}")
        if self.batch_size <= 0:
            raise ValueError(f"Dataset batch_size must be positive, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"Dataset num_workers cannot be negative, got {self.num_workers}")


@dataclass
class DistillationConfig:
    """Knowledge Distillation coefficients and methods config."""
    enable: bool = False
    alpha_pixel: float = 1.0  # Weight for pixel-level KD loss
    beta_feature: float = 0.1  # Weight for intermediate feature-level KD loss
    distill_loss_type: str = "l1"  # choices: "l1", "charbonnier"

    def validate(self) -> None:
        """Validate distillation pipeline properties."""
        if self.alpha_pixel < 0.0 or self.beta_feature < 0.0:
            raise ValueError("Distillation loss coefficients cannot be negative.")
        if self.distill_loss_type not in ("l1", "charbonnier"):
            raise ValueError(
                f"Distillation loss type must be 'l1' or 'charbonnier', got '{self.distill_loss_type}'"
            )


@dataclass
class TrainingConfig:
    """Optimization hyperparameters and reproducibility configs."""
    epochs: int = 300
    lr: float = 1e-4
    lr_milestones: List[int] = field(default_factory=lambda: [150, 250])
    lr_gamma: float = 0.5
    optimizer: str = "Adam"
    weight_decay: float = 0.0
    seed: int = 42

    def validate(self) -> None:
        """Validate optimizer parameters."""
        if self.epochs <= 0:
            raise ValueError(f"Training epochs must be positive, got {self.epochs}")
        if self.lr <= 0.0:
            raise ValueError(f"Training learning rate must be positive, got {self.lr}")
        if self.lr_gamma <= 0.0 or self.lr_gamma > 1.0:
            raise ValueError(f"lr_gamma must be in (0.0, 1.0], got {self.lr_gamma}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay cannot be negative, got {self.weight_decay}")


@dataclass
class EvaluationConfig:
    """Metrics collection and validation loop configurations."""
    metrics: List[str] = field(default_factory=lambda: ["psnr", "ssim"])
    save_sample_interval: int = 10
    evaluate_only: bool = False

    def validate(self) -> None:
        """Validate evaluation configurations."""
        if self.save_sample_interval <= 0:
            raise ValueError(f"save_sample_interval must be positive, got {self.save_sample_interval}")


@dataclass
class DegradationConfig:
    """Degradation pipeline parameters for dynamic low-resolution generation."""
    blur_kernel: int = 21
    blur_sigma: float = 1.2
    noise_std: float = 0.0
    jpeg_quality: int = 100

    def validate(self) -> None:
        """Validate degradation parameters."""
        if self.blur_kernel <= 0 or self.blur_kernel % 2 == 0:
            raise ValueError(f"blur_kernel must be positive and odd, got {self.blur_kernel}")
        if self.blur_sigma < 0.0:
            raise ValueError(f"blur_sigma cannot be negative, got {self.blur_sigma}")
        if self.noise_std < 0.0:
            raise ValueError(f"noise_std cannot be negative, got {self.noise_std}")
        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError(f"jpeg_quality must be in [1, 100], got {self.jpeg_quality}")


@dataclass
class ExperimentConfig:
    """Top-level configuration orchestrating overall framework properties."""
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    output_dir: str = "outputs"
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        """Validate the full experiment config hierarchy."""
        if not self.experiment_name.strip():
            raise ValueError("experiment_name cannot be empty or whitespace.")
        self.model.validate()
        self.dataset.validate()
        self.degradation.validate()
        self.distillation.validate()
        self.training.validate()
        self.evaluation.validate()

    def save(self, filepath: Path) -> None:
        """Write the active configurations state to a YAML file for reproducibility audit.

        Args:
            filepath: Destination file path.
        """
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise IOError(f"Could not save active configuration backup to {filepath}. Error: {e}")


def _dict_to_dataclass(cls: Any, data: Dict[str, Any]) -> Any:
    """Helper function to recursively convert standard dictionaries to Python dataclasses."""
    if not is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in fields(cls)}
    kwargs = {}
    for f in fields(cls):
        name = f.name
        val = data.get(name)
        if val is None:
            # Retain the dataclass field default value
            continue

        field_type = field_types[name]
        # Resolve Union or Optional types if applicable
        origin = get_origin(field_type)
        if origin is not None:
            # Strip Optional and unpack types
            args = field_type.__args__
            non_none_args = [arg for arg in args if arg is not type(None)]
            if non_none_args:
                field_type = non_none_args[0]

        if is_dataclass(field_type):
            kwargs[name] = _dict_to_dataclass(field_type, val)
        else:
            kwargs[name] = val

    return cls(**kwargs)


def load_config(yaml_path: str) -> ExperimentConfig:
    """Load configuration dictionary from a YAML file, parse, and validate it.

    Args:
        yaml_path: Local system file path to target YAML configuration file.

    Returns:
        ExperimentConfig: Fully parsed, strongly-typed, validated config instance.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {yaml_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        raise yaml.YAMLError(f"Failed to read and parse YAML file: {yaml_path}. Error: {e}")

    # Build dataclass instance
    config = _dict_to_dataclass(ExperimentConfig, data)
    config.validate()
    return config
