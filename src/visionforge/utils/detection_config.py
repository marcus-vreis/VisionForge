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

from visionforge.utils.config import DeviceConfig, OutputConfig

# Model names per backend. Explicit so config validation rejects a name that
# does not belong to the chosen backend before any weights are downloaded.
_ULTRALYTICS_MODELS = (
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    "yolo11n",
    "yolo11s",
    "yolo11m",
    "yolo11l",
    "yolo11x",
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


class DetectionTrainingConfig(BaseModel):
    """Detection training hyperparameters (Ultralytics naming).

    Unlike classification, ``batch_size`` is any positive int (Ultralytics does
    not require a power of two). ``learning_rate`` maps to Ultralytics ``lr0``;
    ``patience`` to its early-stopping window.
    """

    epochs: int = Field(default=100, gt=0)
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=0.01, gt=0.0)
    patience: int = Field(default=50, ge=0)
    seed: int = Field(default=0, ge=0)
    workers: int = Field(default=8, ge=0)


class DetectionConfig(BaseModel):
    """Top-level object-detection experiment configuration."""

    name: str = Field(default="detection_001", min_length=1)
    task: Literal["detection"] = "detection"
    model: DetectionModelConfig = Field(default_factory=DetectionModelConfig)
    data: DetectionDataConfig
    training: DetectionTrainingConfig = Field(default_factory=DetectionTrainingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)


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

    return DetectionConfig.model_validate(raw)


__all__ = [
    "DetectionConfig",
    "DetectionModelConfig",
    "DetectionDataConfig",
    "DetectionTrainingConfig",
    "load_detection_config",
]
