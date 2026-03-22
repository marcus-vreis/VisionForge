from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    """CNN architecture and output layer settings."""

    name: Literal[
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "efficientnet_b1",
        "efficientnet_b7",
        "vgg16",
        "vgg19",
        "alexnet",
    ]
    num_classes: int = Field(ge=1)
    pretrained: bool = True
    weights_path: Path | None = None

    @field_validator("weights_path")
    @classmethod
    def weights_path_must_be_file(cls, v: Path | None) -> Path | None:
        if v is None:
            return v
        if not v.exists():
            raise ValueError(f"weights_path does not exist: {v}")
        if not v.is_file():
            raise ValueError(f"weights_path must be a file, got: {v}")
        return v


class TrainingConfig(BaseModel):
    """Hyperparameters and training loop settings."""

    learning_rate: float = Field(gt=0.0)
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    early_stopping_patience: int = Field(default=10, ge=1)
    optimizer: Literal["adam", "sgd", "adamw"] = "adam"
    weight_decay: float = Field(default=0.0, ge=0.0)
    seed: int = Field(default=42, ge=0)

    @field_validator("batch_size")
    @classmethod
    def batch_size_must_be_power_of_two(cls, v: int) -> int:
        if (v & (v - 1)) != 0:
            raise ValueError(f"batch_size must be a power of 2, got {v}.")
        return v


class TransformConfig(BaseModel):
    """Image transform and augmentation settings."""

    image_size: int = Field(default=224, ge=32)
    horizontal_flip: bool = True
    rotation_degrees: int = Field(default=10, ge=0)
    color_jitter: bool = False
    normalize_mean: list[float] = [0.485, 0.456, 0.406]
    normalize_std: list[float] = [0.229, 0.224, 0.225]


class DataConfig(BaseModel):
    """Dataset paths and DataLoader settings."""

    base_dir: Path
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = True
    transforms: TransformConfig = TransformConfig()

    @field_validator("base_dir")
    @classmethod
    def base_dir_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"base_dir does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"base_dir must be a directory, got: {v}")
        return v


class OutputConfig(BaseModel):
    """Output directory paths for models, logs, graphics, and reports."""

    models_dir: Path = Path("outputs/models")
    graphics_dir: Path = Path("outputs/graphics")
    logs_dir: Path = Path("outputs/logs")
    reports_dir: Path = Path("outputs/reports")


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    name: str = Field(min_length=1)
    task: Literal["binary", "multiclass"] = "binary"
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    output: OutputConfig = OutputConfig()

    @model_validator(mode="after")
    def validate_task_and_num_classes(self) -> "ExperimentConfig":
        if self.task == "binary" and self.model.num_classes != 1:
            raise ValueError(
                f"Binary task requires num_classes=1, got {self.model.num_classes}."
            )
        if self.task == "multiclass" and self.model.num_classes < 2:
            raise ValueError(
                f"Multiclass task requires num_classes>=2, got {self.model.num_classes}."
            )
        return self


def load_config(path: Path | str) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file.

    Args:
        path: path to the .yaml config file.

    Returns:
        A fully validated ExperimentConfig instance.

    Raises:
        FileNotFoundError: if the config file does not exist.
        ValueError: if the path is not a file, or the YAML content is not a mapping.
        ValidationError: if any field fails Pydantic validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Config path is not a file: {path}")

    with path.open(encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f)

    if raw is None:
        raw = {}

    if not isinstance(raw, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping, got: {type(raw).__name__}"
        )

    return ExperimentConfig.model_validate(raw)


__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "TransformConfig",
    "OutputConfig",
    "load_config",
]
