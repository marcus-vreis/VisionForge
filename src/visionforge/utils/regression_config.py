"""Pydantic config models for the image-regression task.

Regression is a standalone config tree — it does not reuse the classification
``ExperimentConfig`` because the targets diverge (continuous values keyed by a
CSV manifest instead of an ImageFolder class index, a linear head instead of a
softmax classifier, MSE/MAE/R² instead of accuracy). It *does* reuse
``OutputConfig``, ``DeviceConfig``, ``TransformConfig``, ``PreprocessingConfig``
and ``SchedulerConfig`` so output layout, device selection and the image
pipeline are identical across tasks.
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

# Backbones reused from the classification factory; regression swaps the final
# layer for a linear head rather than a softmax classifier.
_REGRESSION_BACKBONES = (
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "efficientnet_b1",
    "efficientnet_b7",
    "vgg16",
    "vgg19",
    "alexnet",
)


class RegressionModelConfig(BaseModel):
    """CNN backbone and regression-head settings."""

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
    ] = "resnet50"
    num_targets: int = Field(default=1, ge=1)
    pretrained: bool = True
    weights_path: Path | None = None
    custom_model: str | None = Field(
        default=None,
        description=(
            "Name of a user-registered custom model (drop-in user_models/, "
            "ADR-048/049). When set, the builtin `name`/`pretrained` are ignored "
            "and the model is built from the registry (it receives num_targets); "
            "`weights_path` still loads a local checkpoint."
        ),
    )
    timm_model: str | None = Field(
        default=None,
        description=(
            "Name of a timm architecture (e.g. 'convnext_tiny'; needs the `timm` "
            "extra, ADR-051). When set, the builtin `name` is ignored and the model "
            "is built via timm (head sized to num_targets). Mutually exclusive with "
            "custom_model; `weights_path` still loads a local checkpoint."
        ),
    )

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

    @field_validator("custom_model", "timm_model")
    @classmethod
    def blank_model_source_is_none(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            return None
        return v

    @model_validator(mode="after")
    def one_alternate_model_source(self) -> "RegressionModelConfig":
        if self.custom_model is not None and self.timm_model is not None:
            raise ValueError("Set only one of custom_model or timm_model, not both.")
        return self


class RegressionDataConfig(BaseModel):
    """CSV-manifest regression dataset.

    ``base_dir`` holds the per-split CSVs and the ``images_dir`` root. Each CSV
    row maps an image (``image_column``, a path relative to ``images_dir``) to
    one or more numeric ``target_columns``. The number of target columns must
    equal ``model.num_targets`` (validated at the top level).
    """

    base_dir: Path
    images_dir: str = "images"
    train_csv: str = "train.csv"
    val_csv: str = "val.csv"
    test_csv: str = "test.csv"
    image_column: str = "image"
    target_columns: list[str] = Field(default_factory=lambda: ["target"])
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

    @field_validator("target_columns")
    @classmethod
    def target_columns_non_empty_and_unique(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("target_columns must list at least one column.")
        if len(v) != len(set(v)):
            raise ValueError(f"target_columns must be unique, got {v}")
        return v


class RegressionTrainingConfig(BaseModel):
    """Regression training hyperparameters.

    Unlike classification, ``batch_size`` is any positive int (no power-of-two
    requirement) and the loss is a regression criterion. ``scheduler`` reuses the
    shared classification ``SchedulerConfig``.
    """

    learning_rate: float = Field(default=0.001, gt=0.0)
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=32, ge=1)
    early_stopping_patience: int = Field(default=10, ge=1)
    optimizer: Literal["adam", "sgd", "adamw"] = "adam"
    weight_decay: float = Field(default=0.0, ge=0.0)
    loss: Literal["mse", "mae", "huber"] = "mse"
    huber_delta: float = Field(default=1.0, gt=0.0)
    seed: int = Field(default=42, ge=0)
    deterministic: bool = Field(default=False, description=DETERMINISTIC_DESCRIPTION)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)


class RegressionTransferLearningConfig(BaseModel):
    """Transfer-learning over the shared CNN backbone (optional).

    ``feature_extraction`` freezes the backbone and trains only the regression
    head; ``fine_tuning`` trains the whole network but scales the backbone's
    learning rate by ``backbone_lr_multiplier`` (discriminative fine-tuning).
    Absent from the config → standard full-network training at a single LR.
    """

    mode: Literal["feature_extraction", "fine_tuning"] = "feature_extraction"
    backbone_lr_multiplier: float = Field(default=0.1, gt=0.0, le=1.0)


class RegressionConfig(BaseModel):
    """Top-level image-regression experiment configuration."""

    name: str = Field(default="regression_001", min_length=1)
    schema_version: int = Field(default=CURRENT_SCHEMA_VERSION, ge=1)
    task: Literal["regression"] = "regression"
    model: RegressionModelConfig = Field(default_factory=RegressionModelConfig)
    data: RegressionDataConfig
    training: RegressionTrainingConfig = Field(default_factory=RegressionTrainingConfig)
    transfer_learning: RegressionTransferLearningConfig | None = None
    output: OutputConfig = Field(default_factory=OutputConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)

    @model_validator(mode="after")
    def targets_match_columns(self) -> "RegressionConfig":
        check_schema_version(self.schema_version)
        n_cols = len(self.data.target_columns)
        if n_cols != self.model.num_targets:
            raise ValueError(
                f"model.num_targets ({self.model.num_targets}) must equal the "
                f"number of data.target_columns ({n_cols})."
            )
        return self


def load_regression_config(path: Path | str) -> RegressionConfig:
    """Load and validate a regression experiment config from a YAML file.

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

    return RegressionConfig.model_validate(migrate_config_dict(raw))


__all__ = [
    "RegressionConfig",
    "RegressionModelConfig",
    "RegressionDataConfig",
    "RegressionTrainingConfig",
    "RegressionTransferLearningConfig",
    "load_regression_config",
    "_REGRESSION_BACKBONES",
]
