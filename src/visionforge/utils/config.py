from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    """
    Configuration for the CNN model.

    Args:
        name:        Architecture to use. Must be one of the supported families.
        num_classes: Number of output neurons. Use 1 for binary, N for multiclass.
        pretrained:  Whether to initialize with ImageNet weights.
    """

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


class TrainingConfig(BaseModel):
    """
    Hyperparameters and training loop settings.

    Args:
        learning_rate:           Step size for the optimizer. Must be > 0.
        epochs:                  Maximum number of training epochs.
        batch_size:              Samples per gradient update. Must be a power of 2.
        early_stopping_patience: Stop after N epochs without validation improvement.
        optimizer:               Optimization algorithm.
        weight_decay:            L2 regularization coefficient.
    """

    learning_rate: float = Field(gt=0.0)
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    early_stopping_patience: int = Field(default=10, ge=1)
    optimizer: Literal["adam", "sgd", "adamw"] = "adam"
    weight_decay: float = Field(default=0.0, ge=0.0)

    @field_validator("batch_size")
    @classmethod
    def batch_size_must_be_power_of_two(cls, v: int) -> int:
        if (v & (v - 1)) != 0:
            raise ValueError(f"batch_size must be a power of 2, got {v}.")
        return v


class DataConfig(BaseModel):
    """
    Dataset paths and DataLoader settings.

    Args:
        base_dir:    Root directory containing train/, val/, and test/ subfolders.
        train_dir:   Name of the training subdirectory inside base_dir.
        val_dir:     Name of the validation subdirectory inside base_dir.
        test_dir:    Name of the test subdirectory inside base_dir.
        image_size:  Images are resized to (image_size x image_size) pixels.
        num_workers: Parallel worker processes for data loading. Use 0 to disable.
                     Increase for faster loading when CPU has multiple cores.
        pin_memory:  Pin DataLoader memory for faster CPU-to-GPU transfers.
                     Recommended when training on GPU.
    """

    base_dir: Path
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"
    image_size: int = Field(default=224, ge=32)
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = True

    @field_validator("base_dir")
    @classmethod
    def base_dir_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"base_dir does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"base_dir must be a directory, got: {v}")
        return v


class OutputConfig(BaseModel):
    """
    Output directory paths for models, graphics, logs, and reports.

    Args:
        models_dir:   Directory for saved model checkpoints (.pth files).
        graphics_dir: Directory for exported plots (accuracy, loss, confusion matrix).
        logs_dir:     Directory for log files.
        reports_dir:  Directory for HTML/PDF experiment reports.
    """

    models_dir: Path = Path("outputs/models")
    graphics_dir: Path = Path("outputs/graphics")
    logs_dir: Path = Path("outputs/logs")
    reports_dir: Path = Path("outputs/reports")


class ExperimentConfig(BaseModel):
    """
    Top-level experiment configuration.

    Args:
        name:     Unique identifier for the experiment. Used in output filenames.
        task:     Classification task type. Determines loss function and output layer.
        model:    Model architecture and initialization settings.
        training: Hyperparameters and training loop settings.
        data:     Dataset paths and DataLoader settings.
        output:   Output directory paths. Defaults to outputs/<type>/.
    """

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
    """
    Load and validate an experiment config from a YAML file.

    Args:
        path: Path to the .yaml config file.

    Returns:
        A fully validated ExperimentConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError:        If the given path exists but is not a regular file,
                           or if the YAML content is not a mapping.
        ValidationError:   If any field fails Pydantic validation.
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
    "OutputConfig",
    "load_config",
]
