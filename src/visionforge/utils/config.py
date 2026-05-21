from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GridSearchConfig(BaseModel):
    """Search space for exhaustive hyperparameter grid search."""

    hyperparameters: dict[str, list[Any]] = Field(default_factory=dict)


class RandomSearchConfig(BaseModel):
    """Search space and budget for random hyperparameter search."""

    n_trials: int = Field(ge=1)
    seed: int = Field(default=42, ge=0)
    # Each value is a raw param dict: {type: uniform|log_uniform|choice, ...}
    search_space: dict[str, Any] = Field(default_factory=dict)


class CrossValidationConfig(BaseModel):
    """Settings for K-Fold and Stratified K-Fold cross-validation."""

    n_folds: int = Field(default=5, ge=2)
    stratified: bool = True
    shuffle: bool = True
    fold_seed: int = 42


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
    ] = "resnet50"
    num_classes: int = Field(default=2, ge=1)
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

    learning_rate: float = Field(default=0.001, gt=0.0)
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=32, ge=1)
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

    models_dir: Path = Field(
        default=Path("outputs/models"), json_schema_extra={"default": "outputs/models"}
    )
    graphics_dir: Path = Field(
        default=Path("outputs/graphics"),
        json_schema_extra={"default": "outputs/graphics"},
    )
    logs_dir: Path = Field(
        default=Path("outputs/logs"), json_schema_extra={"default": "outputs/logs"}
    )
    reports_dir: Path = Field(
        default=Path("outputs/reports"),
        json_schema_extra={"default": "outputs/reports"},
    )


class ClassificationConfig(BaseModel):
    """Classification block operating mode and optional checkpoint."""

    mode: Literal["train", "evaluate", "infer"] = "train"
    checkpoint_path: Path | None = None


class TransferLearningConfig(BaseModel):
    """Settings for feature extraction and fine-tuning transfer learning."""

    mode: Literal["feature_extraction", "fine_tuning"] = "feature_extraction"
    unfreeze_from_layer: str | None = None
    backbone_lr_multiplier: float = Field(default=0.1, gt=0.0, le=1.0)


# Architectures mirrored from ModelConfig.name — kept in sync manually.
_ArchitectureLiteral = Literal[
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


class ModelComparisonConfig(BaseModel):
    """Settings for ranking N architectures on the same dataset."""

    model_names: list[_ArchitectureLiteral] = Field(min_length=2)  # type: ignore[valid-type]
    metric: Literal["accuracy", "f1", "auc_roc"] = "f1"


class BatchPredictionConfig(BaseModel):
    """Settings for batch inference over a folder of images."""

    checkpoint_path: Path
    input_dir: Path
    output_csv: Path
    recursive: bool = True
    image_extensions: list[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    )
    class_names: list[str] | None = None

    @field_validator("checkpoint_path")
    @classmethod
    def checkpoint_must_be_file(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"checkpoint_path does not exist: {v}")
        if not v.is_file():
            raise ValueError(f"checkpoint_path must be a file, got: {v}")
        return v

    @field_validator("input_dir")
    @classmethod
    def input_dir_must_be_directory(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"input_dir does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"input_dir must be a directory, got: {v}")
        return v


class ExportONNXConfig(BaseModel):
    """Settings for ONNX export, validation, and latency benchmarking."""

    # populate_by_name lets callers use either the alias ("validate") or the
    # field name ("run_validate") when constructing via dict/YAML.  The alias
    # is necessary because "validate" shadows a deprecated Pydantic BaseModel
    # classmethod and triggers a UserWarning at class-definition time.
    model_config = ConfigDict(populate_by_name=True)

    checkpoint_path: Path
    output_onnx: Path
    opset_version: int = Field(default=17, ge=11, le=20)
    dynamic_axes: bool = True
    run_validate: bool = Field(default=True, alias="validate")
    validation_tolerance: float = Field(default=1e-4, gt=0.0)
    benchmark: bool = True
    benchmark_runs: int = Field(default=50, ge=5)

    @field_validator("checkpoint_path")
    @classmethod
    def checkpoint_path_must_be_file(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"checkpoint_path does not exist: {v}")
        if not v.is_file():
            raise ValueError(f"checkpoint_path must be a file, got: {v}")
        return v


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    name: str = Field(default="experiment_001", min_length=1)
    task: Literal["binary", "multiclass"] = "multiclass"
    block: Literal[
        "classification",
        "grid_search",
        "random_search",
        "cross_validation",
        "transfer_learning",
        "model_comparison",
        "batch_prediction",
        "export_onnx",
    ] = "classification"
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig
    output: OutputConfig = OutputConfig()
    classification: ClassificationConfig = ClassificationConfig()
    grid_search: GridSearchConfig | None = None
    random_search: RandomSearchConfig | None = None
    cross_validation: CrossValidationConfig | None = None
    transfer_learning: TransferLearningConfig | None = None
    model_comparison: ModelComparisonConfig | None = None
    batch_prediction: BatchPredictionConfig | None = None
    export_onnx: ExportONNXConfig | None = None

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
        if (
            self.model_comparison is not None
            and self.model_comparison.metric == "auc_roc"
            and self.task == "multiclass"
        ):
            raise ValueError(
                "auc_roc metric is not defined for multiclass tasks. "
                "Use 'accuracy' or 'f1' instead."
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
    "ClassificationConfig",
    "GridSearchConfig",
    "RandomSearchConfig",
    "CrossValidationConfig",
    "TransferLearningConfig",
    "ModelComparisonConfig",
    "BatchPredictionConfig",
    "ExportONNXConfig",
    "load_config",
]
