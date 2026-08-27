"""Pydantic config models for the anomaly-detection task.

Anomaly detection is a standalone config tree — it does not reuse the
classification ``ExperimentConfig`` because it is unsupervised: training sees
**normal images only**, the model is a reconstruction autoencoder or a memory
bank of normal patch features (no classifier head), and the metric is image-level
AUROC over a labelled test split. It *does* reuse ``OutputConfig``,
``DeviceConfig``, ``TransformConfig``, ``PreprocessingConfig`` and
``SchedulerConfig`` so output layout, device selection and the image pipeline are
identical across tasks.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from visionforge.utils.config import (
    CURRENT_SCHEMA_VERSION,
    DETERMINISTIC_DESCRIPTION,
    DeviceConfig,
    OutputConfig,
    PreprocessingConfig,
    SchedulerConfig,
    TransformConfig,
    check_schema_version,
    migrate_config_dict,
)

# PatchCore feature-extractor backbones (frozen ImageNet). The autoencoder is
# hand-rolled and ignores this field.
_ANOMALY_BACKBONES = ("resnet18", "resnet34", "resnet50", "wide_resnet50_2")


class AnomalyModelConfig(BaseModel):
    """Anomaly model selection and per-family settings."""

    name: Literal["autoencoder", "patchcore"] = "autoencoder"
    backbone: Literal["resnet18", "resnet34", "resnet50", "wide_resnet50_2"] = (
        "resnet18"
    )
    latent_dim: int = Field(default=512, ge=1)
    coreset_ratio: float = Field(default=0.1, gt=0.0, le=1.0)
    pretrained: bool = True


class AnomalyDataConfig(BaseModel):
    """MVTec-AD-style anomaly dataset.

    ``train_dir/normal_dir`` holds normal images only (unsupervised training).
    ``test_dir`` holds the ``normal_dir`` subfolder (label 0) plus one or more
    defect subfolders (any non-``normal_dir`` subdir → label 1).
    """

    base_dir: Path
    train_dir: str = "train"
    test_dir: str = "test"
    normal_dir: str = "good"
    image_size: int = Field(default=256, ge=32)
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = True
    transforms: TransformConfig = Field(default_factory=TransformConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)

    @field_validator("base_dir")
    @classmethod
    def base_dir_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"base_dir does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"base_dir must be a directory, got: {v}")
        return v


class AnomalyTrainingConfig(BaseModel):
    """Anomaly training hyperparameters.

    Used for the autoencoder reconstruction loop; PatchCore ignores the epoch
    fields (its "fit" is a single forward pass). ``threshold_percentile`` sets the
    decision threshold from the normal-score distribution (AUROC itself is
    threshold-free).
    """

    learning_rate: float = Field(default=0.001, gt=0.0)
    epochs: int = Field(default=30, ge=1)
    batch_size: int = Field(default=32, ge=1)
    # Zero by default because the previous default never fired: with
    # `epochs=10` it took ten consecutive epochs without improvement inside
    # ten epochs. It read as a protection that was not there.
    early_stopping_patience: int = Field(
        default=0,
        ge=0,
        description="Épocas seguidas sem melhora antes de encerrar o treino. 0 (padrão) desliga: roda todas as épocas configuradas.",
    )
    optimizer: Literal["adam", "sgd", "adamw"] = "adam"
    weight_decay: float = Field(default=0.0, ge=0.0)
    threshold_percentile: float = Field(default=95.0, ge=0.0, le=100.0)
    seed: int = Field(default=42, ge=0)
    deterministic: bool = Field(default=True, description=DETERMINISTIC_DESCRIPTION)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)


class AnomalyConfig(BaseModel):
    """Top-level anomaly-detection experiment configuration."""

    name: str = Field(default="anomaly_001", min_length=1)
    schema_version: int = Field(default=CURRENT_SCHEMA_VERSION, ge=1)
    task: Literal["anomaly"] = "anomaly"
    model: AnomalyModelConfig = Field(default_factory=AnomalyModelConfig)
    data: AnomalyDataConfig
    training: AnomalyTrainingConfig = Field(default_factory=AnomalyTrainingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)

    @model_validator(mode="after")
    def reject_future_schema_version(self) -> "AnomalyConfig":
        check_schema_version(self.schema_version)
        return self


def load_anomaly_config(path: Path | str) -> AnomalyConfig:
    """Load and validate an anomaly experiment config from a YAML file.

    Raises:
        FileNotFoundError: if the config file does not exist.
        ValueError: if the path is not a file, or the YAML is not a mapping.
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

    return AnomalyConfig.model_validate(migrate_config_dict(raw))


__all__ = [
    "AnomalyConfig",
    "AnomalyModelConfig",
    "AnomalyDataConfig",
    "AnomalyTrainingConfig",
    "load_anomaly_config",
    "_ANOMALY_BACKBONES",
]
