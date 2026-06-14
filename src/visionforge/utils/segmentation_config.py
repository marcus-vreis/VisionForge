"""Pydantic config models for the semantic-segmentation task (Phase 8).

Segmentation is a standalone config tree — it does not reuse the classification
``ExperimentConfig`` because the target diverges (a per-pixel class mask instead
of a single class index, a dense head instead of a pooled classifier, mean IoU /
Dice instead of accuracy). It *does* reuse ``OutputConfig``, ``DeviceConfig``,
``TransformConfig``, ``PreprocessingConfig`` and ``SchedulerConfig`` so output
layout, device selection and the image pipeline are identical across tasks.
See documentation/PHASE8_SEGMENTATION_PLAN.md.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from visionforge.utils.config import (
    CURRENT_SCHEMA_VERSION,
    DeviceConfig,
    OutputConfig,
    PreprocessingConfig,
    SchedulerConfig,
    TransformConfig,
    check_schema_version,
    migrate_config_dict,
)

# torchvision segmentation families + a hand-rolled U-Net (brick 3). Explicit so
# config validation rejects an unsupported name before any weights are loaded.
_SEGMENTATION_MODELS = (
    "unet",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
    "deeplabv3_mobilenet_v3_large",
    "fcn_resnet50",
    "fcn_resnet101",
    "lraspp_mobilenet_v3_large",
)


class SegmentationModelConfig(BaseModel):
    """Segmentation architecture and head settings."""

    name: Literal[
        "unet",
        "deeplabv3_resnet50",
        "deeplabv3_resnet101",
        "deeplabv3_mobilenet_v3_large",
        "fcn_resnet50",
        "fcn_resnet101",
        "lraspp_mobilenet_v3_large",
    ] = "deeplabv3_resnet50"
    num_classes: int = Field(default=2, ge=1)
    pretrained: bool = True
    weights_path: Path | None = None
    custom_model: str | None = Field(
        default=None,
        description=(
            "Name of a user-registered custom model (drop-in user_models/, "
            "ADR-048/049). When set, the builtin `name`/`pretrained` are ignored "
            "and the model is built from the registry (it receives num_classes and "
            "must emit per-pixel logits); `weights_path` still loads a checkpoint."
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

    @field_validator("custom_model")
    @classmethod
    def blank_custom_model_is_none(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            return None
        return v


class SegmentationDataConfig(BaseModel):
    """Paired image/mask segmentation dataset.

    ``base_dir`` holds per-split directories (``train_dir``/``val_dir``/
    ``test_dir``), each with an ``images_subdir`` and a ``masks_subdir``. Image
    and mask are paired by filename stem; masks are single-channel images whose
    pixel value is the class id (0..num_classes-1), with ``ignore_index`` marking
    void/unlabeled pixels excluded from loss and metrics.
    """

    base_dir: Path
    images_subdir: str = "images"
    masks_subdir: str = "masks"
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"
    image_size: int = Field(default=512, ge=32)
    ignore_index: int = 255
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


class SegmentationTrainingConfig(BaseModel):
    """Segmentation training hyperparameters.

    Like detection/regression, ``batch_size`` is any positive int (no power-of-
    two requirement). ``loss`` is a pixel-wise criterion; ``scheduler`` reuses the
    shared classification ``SchedulerConfig``.
    """

    learning_rate: float = Field(default=0.001, gt=0.0)
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=8, ge=1)
    early_stopping_patience: int = Field(default=10, ge=1)
    optimizer: Literal["adam", "sgd", "adamw"] = "adam"
    weight_decay: float = Field(default=0.0, ge=0.0)
    loss: Literal["cross_entropy", "dice", "combined"] = "cross_entropy"
    seed: int = Field(default=42, ge=0)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)


class SegmentationTransferLearningConfig(BaseModel):
    """Transfer-learning over a pretrained segmentation backbone (optional).

    ``feature_extraction`` freezes the backbone and trains only the dense head
    (the model's last named child — ``classifier`` for the torchvision families,
    ``outc`` for U-Net); ``fine_tuning`` trains the whole network but scales the
    backbone's learning rate by ``backbone_lr_multiplier``. Absent → standard
    full-network training at a single LR. Most useful for the torchvision families
    (ImageNet-pretrained backbones); U-Net has no pretrained weights.
    """

    mode: Literal["feature_extraction", "fine_tuning"] = "feature_extraction"
    backbone_lr_multiplier: float = Field(default=0.1, gt=0.0, le=1.0)


class SegmentationConfig(BaseModel):
    """Top-level semantic-segmentation experiment configuration."""

    name: str = Field(default="segmentation_001", min_length=1)
    schema_version: int = Field(default=CURRENT_SCHEMA_VERSION, ge=1)
    task: Literal["segmentation"] = "segmentation"
    model: SegmentationModelConfig = Field(default_factory=SegmentationModelConfig)
    data: SegmentationDataConfig
    training: SegmentationTrainingConfig = Field(
        default_factory=SegmentationTrainingConfig
    )
    transfer_learning: SegmentationTransferLearningConfig | None = None
    output: OutputConfig = Field(default_factory=OutputConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)

    @model_validator(mode="after")
    def ignore_index_must_not_collide(self) -> "SegmentationConfig":
        check_schema_version(self.schema_version)
        idx = self.data.ignore_index
        if 0 <= idx < self.model.num_classes:
            raise ValueError(
                f"data.ignore_index ({idx}) collides with a real class id "
                f"(0..{self.model.num_classes - 1}); use a value outside that "
                f"range (e.g. 255 or -1)."
            )
        return self


def load_segmentation_config(path: Path | str) -> SegmentationConfig:
    """Load and validate a segmentation experiment config from a YAML file.

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

    return SegmentationConfig.model_validate(migrate_config_dict(raw))


__all__ = [
    "SegmentationConfig",
    "SegmentationModelConfig",
    "SegmentationDataConfig",
    "SegmentationTrainingConfig",
    "SegmentationTransferLearningConfig",
    "load_segmentation_config",
    "_SEGMENTATION_MODELS",
]
