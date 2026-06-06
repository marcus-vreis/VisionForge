"""Pydantic config models for the object-detection task (Phase 7).

Detection is a standalone config tree — it does not reuse the classification
``ExperimentConfig`` because the field sets diverge (no power-of-two batch,
mAP instead of accuracy, dataset is boxes not ImageFolder). It does reuse
``OutputConfig`` and ``DeviceConfig`` so output layout and device selection are
identical across tasks. See documentation/PHASE7_DETECTION_PLAN.md.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from visionforge.utils.config import (
    CURRENT_SCHEMA_VERSION,
    DeviceConfig,
    OutputConfig,
    check_schema_version,
    migrate_config_dict,
)

# Model names per backend. Explicit so config validation rejects a name that
# does not belong to the chosen backend before any weights are downloaded.
# Covers every detection family Ultralytics ships: YOLOv8, YOLOv9, YOLOv10,
# YOLO11, YOLO12, YOLO26, and RT-DETR. Variant letters follow each family's
# own convention (YOLOv9 uses t/s/m/c/e; YOLOv10 adds a 'b'); a wrong name
# would only fail when Ultralytics tries to fetch the weights, so we gate here.
_ULTRALYTICS_MODELS = (
    # YOLOv8
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    # YOLOv9
    "yolov9t",
    "yolov9s",
    "yolov9m",
    "yolov9c",
    "yolov9e",
    # YOLOv10
    "yolov10n",
    "yolov10s",
    "yolov10m",
    "yolov10b",
    "yolov10l",
    "yolov10x",
    # YOLO11
    "yolo11n",
    "yolo11s",
    "yolo11m",
    "yolo11l",
    "yolo11x",
    # YOLO12
    "yolo12n",
    "yolo12s",
    "yolo12m",
    "yolo12l",
    "yolo12x",
    # YOLO26 (NMS-free, DFL-free; n/s/m/l/x)
    "yolo26n",
    "yolo26s",
    "yolo26m",
    "yolo26l",
    "yolo26x",
    # RT-DETR
    "rtdetr-l",
    "rtdetr-x",
)
_TORCHVISION_MODELS = (
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "retinanet_resnet50_fpn",
    "ssd300_vgg16",
    "ssdlite320_mobilenet_v3_large",
)


class DetectionModelConfig(BaseModel):
    """Detector architecture and backend selection."""

    backend: Literal["ultralytics", "torchvision"] = "ultralytics"
    name: str = "yolo11n"
    num_classes: int = Field(default=1, ge=1)
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

    @model_validator(mode="after")
    def name_must_match_backend(self) -> "DetectionModelConfig":
        allowed = (
            _ULTRALYTICS_MODELS
            if self.backend == "ultralytics"
            else _TORCHVISION_MODELS
        )
        if self.name not in allowed:
            raise ValueError(
                f"Model '{self.name}' is not a valid {self.backend} detector. "
                f"Choose one of: {', '.join(allowed)}."
            )
        return self


class DetectionDataConfig(BaseModel):
    """Detection dataset source.

    Either ``data_yaml`` (an existing Ultralytics ``data.yaml``) or ``base_dir``
    (a YOLO-layout root that the DataModule synthesizes a ``data.yaml`` from)
    must be provided. ``image_size`` is the square training resolution (Ultralytics
    ``imgsz``).
    """

    data_yaml: Path | None = None
    base_dir: Path | None = None
    image_size: int = Field(default=640, ge=32)
    class_names: list[str] | None = None

    @field_validator("data_yaml")
    @classmethod
    def data_yaml_must_be_file(cls, v: Path | None) -> Path | None:
        if v is None:
            return v
        if not v.exists():
            raise ValueError(f"data_yaml does not exist: {v}")
        if not v.is_file():
            raise ValueError(f"data_yaml must be a file, got: {v}")
        return v

    @field_validator("base_dir")
    @classmethod
    def base_dir_must_be_directory(cls, v: Path | None) -> Path | None:
        if v is None:
            return v
        if not v.exists():
            raise ValueError(f"base_dir does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"base_dir must be a directory, got: {v}")
        return v

    @model_validator(mode="after")
    def at_least_one_source(self) -> "DetectionDataConfig":
        if self.data_yaml is None and self.base_dir is None:
            raise ValueError(
                "DetectionDataConfig requires either data_yaml or base_dir."
            )
        return self


class DetectionAugmentationConfig(BaseModel):
    """Data-augmentation hyperparameters (Ultralytics naming and defaults).

    Every field maps 1:1 to an Ultralytics ``train`` augmentation argument. The
    defaults reproduce Ultralytics' own defaults, so an unmodified config trains
    identically to the bare ``YOLO.train`` call it replaced.
    """

    hsv_h: float = Field(default=0.015, ge=0.0, le=1.0)  # hue jitter fraction
    hsv_s: float = Field(default=0.7, ge=0.0, le=1.0)  # saturation jitter
    hsv_v: float = Field(default=0.4, ge=0.0, le=1.0)  # value/brightness jitter
    degrees: float = Field(default=0.0, ge=-180.0, le=180.0)  # rotation
    translate: float = Field(default=0.1, ge=0.0, le=1.0)
    scale: float = Field(default=0.5, ge=0.0)  # gain (+/-) range
    shear: float = Field(default=0.0, ge=-180.0, le=180.0)
    perspective: float = Field(default=0.0, ge=0.0, le=0.001)
    flipud: float = Field(default=0.0, ge=0.0, le=1.0)  # vertical-flip prob
    fliplr: float = Field(default=0.5, ge=0.0, le=1.0)  # horizontal-flip prob
    bgr: float = Field(default=0.0, ge=0.0, le=1.0)  # channel-swap prob
    mosaic: float = Field(default=1.0, ge=0.0, le=1.0)  # 4-image mosaic prob
    mixup: float = Field(default=0.0, ge=0.0, le=1.0)
    copy_paste: float = Field(default=0.0, ge=0.0, le=1.0)
    auto_augment: Literal["randaugment", "autoaugment", "augmix"] | None = "randaugment"
    erasing: float = Field(default=0.4, ge=0.0, le=1.0)  # random-erasing prob


class DetectionTrainingConfig(BaseModel):
    """Detection training hyperparameters (Ultralytics naming).

    Unlike classification, ``batch_size`` is any positive int (Ultralytics does
    not require a power of two). ``learning_rate`` maps to Ultralytics ``lr0``;
    ``patience`` to its early-stopping window. The optimizer, schedule, loss-gain,
    and regularization fields all map 1:1 to Ultralytics ``train`` arguments and
    default to its own defaults, so an unmodified config is behaviour-preserving.
    The ``optimizer``/``momentum``/``weight_decay`` trio is also honoured by the
    torchvision backend (which previously hard-coded SGD).
    """

    epochs: int = Field(default=100, gt=0)
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=0.01, gt=0.0)  # Ultralytics lr0
    patience: int = Field(default=50, ge=0)
    seed: int = Field(default=0, ge=0)
    workers: int = Field(default=8, ge=0)

    # Optimizer
    optimizer: Literal[
        "auto", "SGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp"
    ] = "auto"
    momentum: float = Field(default=0.937, ge=0.0, le=1.0)  # SGD momentum / Adam beta1
    weight_decay: float = Field(default=0.0005, ge=0.0)

    # Learning-rate schedule
    lrf: float = Field(default=0.01, gt=0.0)  # final LR = lr0 * lrf
    cos_lr: bool = False  # cosine schedule instead of linear
    warmup_epochs: float = Field(default=3.0, ge=0.0)
    warmup_momentum: float = Field(default=0.8, ge=0.0, le=1.0)
    warmup_bias_lr: float = Field(default=0.1, ge=0.0)

    # Loss gains
    box: float = Field(default=7.5, ge=0.0)  # box-regression loss weight
    cls: float = Field(default=0.5, ge=0.0)  # classification loss weight
    dfl: float = Field(default=1.5, ge=0.0)  # distribution-focal loss weight

    # Regularization / training mechanics
    label_smoothing: float = Field(default=0.0, ge=0.0, le=1.0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    nbs: int = Field(default=64, ge=1)  # nominal batch size for loss normalization
    freeze: int | None = Field(default=None, ge=0)  # freeze first N layers
    amp: bool = True  # automatic mixed precision
    close_mosaic: int = Field(default=10, ge=0)  # disable mosaic in last N epochs
    single_cls: bool = False  # treat all classes as one
    rect: bool = False  # rectangular batches (min padding)
    multi_scale: bool = False  # vary imgsz +/- 50% during training

    augmentation: DetectionAugmentationConfig = Field(
        default_factory=DetectionAugmentationConfig
    )


class DetectionConfig(BaseModel):
    """Top-level object-detection experiment configuration."""

    name: str = Field(default="detection_001", min_length=1)
    schema_version: int = Field(default=CURRENT_SCHEMA_VERSION, ge=1)
    task: Literal["detection"] = "detection"
    model: DetectionModelConfig = Field(default_factory=DetectionModelConfig)
    data: DetectionDataConfig
    training: DetectionTrainingConfig = Field(default_factory=DetectionTrainingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)

    @model_validator(mode="after")
    def reject_future_schema_version(self) -> "DetectionConfig":
        check_schema_version(self.schema_version)
        return self


def load_detection_config(path: Path | str) -> DetectionConfig:
    """Load and validate a detection experiment config from a YAML file.

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

    return DetectionConfig.model_validate(migrate_config_dict(raw))


__all__ = [
    "DetectionConfig",
    "DetectionModelConfig",
    "DetectionDataConfig",
    "DetectionTrainingConfig",
    "DetectionAugmentationConfig",
    "load_detection_config",
]
