from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Bumped whenever a breaking change is made to the experiment config schema. A
# config carries the version it was written with so old YAML/run.json files can
# be migrated forward (see ``migrate_config_dict``) without losing traceability —
# the reproducibility guarantee depends on this (CLAUDE.md §6.2).
CURRENT_SCHEMA_VERSION = 1


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
    custom_model: str | None = Field(
        default=None,
        description=(
            "Name of a user-registered custom model (drop-in user_models/, ADR-048). "
            "When set, the builtin `name`/`pretrained` are ignored and the model is "
            "built from the registry; `weights_path` still loads a local checkpoint."
        ),
    )
    timm_model: str | None = Field(
        default=None,
        description=(
            "Name of a timm architecture (e.g. 'convnext_tiny'; needs the `timm` "
            "extra, ADR-051). When set, the builtin `name` is ignored and the model "
            "is built via timm with `pretrained` honoured. Mutually exclusive with "
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
        # An empty string (e.g. an untouched GUI text field) means "not set" —
        # coerce to None so the builtin backbone path runs.
        if v is not None and not v.strip():
            return None
        return v

    @model_validator(mode="after")
    def one_alternate_model_source(self) -> "ModelConfig":
        if self.custom_model is not None and self.timm_model is not None:
            raise ValueError("Set only one of custom_model or timm_model, not both.")
        return self


class SchedulerConfig(BaseModel):
    """Learning rate scheduler choice and its parameters.

    ``kind="none"`` disables scheduling (fixed LR). The other choices map
    directly to ``torch.optim.lr_scheduler``:

    - ``cosine``: ``CosineAnnealingLR(T_max=epochs)`` — smooth decay from
      ``learning_rate`` to ~0 over the full training horizon.
    - ``step``: ``StepLR(step_size, gamma)`` — multiplicative drops at
      fixed epoch intervals.
    - ``plateau``: ``ReduceLROnPlateau(patience, factor)`` — reactive
      decay based on validation loss; stepped after the eval epoch.
    """

    kind: Literal["none", "cosine", "step", "plateau"] = "none"
    step_size: int = Field(default=10, ge=1)
    gamma: float = Field(default=0.1, gt=0.0, le=1.0)
    patience: int = Field(default=5, ge=1)
    factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    min_lr: float = Field(default=1e-6, ge=0.0)


# Shared by every task's training config (ADR-062): the knob means the same
# thing everywhere, so it is described once instead of drifting five times.
DETERMINISTIC_DESCRIPTION = (
    "When True, forces cuDNN deterministic algorithms and disables "
    "auto-tuning. Guarantees bit-exact reproducibility across runs "
    "but significantly reduces GPU utilization and throughput. "
    "Leave False (default) for normal training."
)


class TrainingConfig(BaseModel):
    """Hyperparameters and training loop settings."""

    learning_rate: float = Field(default=0.001, gt=0.0)
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=32, ge=1)
    early_stopping_patience: int = Field(default=10, ge=1)
    optimizer: Literal["adam", "sgd", "adamw"] = "adam"
    weight_decay: float = Field(default=0.0, ge=0.0)
    seed: int = Field(default=42, ge=0)
    deterministic: bool = Field(default=False, description=DETERMINISTIC_DESCRIPTION)
    mixed_precision: bool = Field(
        default=False,
        description=(
            "Enable torch.amp autocast + GradScaler. 2-3x speedup on "
            "Ampere+ GPUs at minor accuracy cost. Ignored on CPU."
        ),
    )
    scheduler: SchedulerConfig = Field(default_factory=lambda: SchedulerConfig())

    @field_validator("batch_size")
    @classmethod
    def batch_size_must_be_power_of_two(cls, v: int) -> int:
        if (v & (v - 1)) != 0:
            raise ValueError(f"batch_size must be a power of 2, got {v}.")
        return v


class PreprocessingStep(BaseModel):
    """One step in the preprocessing pipeline applied before augmentation."""

    # populate_by_name lets configs use {"kind": "gaussian_blur", "radius": 1.5}
    # — extra params are accepted because each filter consumes a different set
    # (median needs size, wavelet needs band, etc.). They're forwarded as a dict
    # to ``visionforge.core.preprocessing.apply_step``.
    model_config = ConfigDict(extra="allow")

    kind: Literal[
        "gaussian_blur",
        "median_blur",
        "unsharp",
        "edges",
        "emboss",
        "grayscale",
        "equalize",
        "autocontrast",
        "wavelet",
    ]


class PreprocessingConfig(BaseModel):
    """Ordered list of preprocessing steps applied to every loaded image.

    Pipeline runs before the standard augmentation/normalize transforms in
    ``DataModule``. Empty list = identity (no preprocessing).
    """

    steps: list[PreprocessingStep] = Field(default_factory=list)


class TransformConfig(BaseModel):
    """Image transform and augmentation settings."""

    image_size: int = Field(default=224, ge=32)
    # Off skips the augmenting steps below without discarding their values, so a
    # baseline run and the tuned run differ by one field in the run.json — and
    # turning it back on restores the tuning instead of losing it.
    augment: bool = True
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
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)

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


class DeviceConfig(BaseModel):
    """Compute device selection for training and evaluation.

    ``kind`` chooses the broad category:
    - ``cpu``: force CPU even when CUDA is available
    - ``cuda``: single GPU (``gpu_ids[0]`` if given, otherwise GPU 0)
    - ``multi_cuda``: DataParallel across ``gpu_ids`` (or all visible GPUs)
    """

    kind: Literal["cpu", "cuda", "multi_cuda"] = "cuda"
    gpu_ids: list[int] | None = None

    @field_validator("gpu_ids")
    @classmethod
    def gpu_ids_must_be_non_negative(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return v
        if any(i < 0 for i in v):
            raise ValueError(f"gpu_ids must all be >= 0, got {v}")
        if len(v) != len(set(v)):
            raise ValueError(f"gpu_ids must be unique, got {v}")
        return v

    @model_validator(mode="after")
    def validate_kind_and_ids(self) -> "DeviceConfig":
        if (
            self.kind == "multi_cuda"
            and self.gpu_ids is not None
            and len(self.gpu_ids) < 2
        ):
            raise ValueError(
                f"multi_cuda requires at least 2 gpu_ids, got {self.gpu_ids}"
            )
        return self


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
    schema_version: int = Field(default=CURRENT_SCHEMA_VERSION, ge=1)
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
    device: DeviceConfig = Field(default_factory=DeviceConfig)
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
        if self.schema_version > CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"Config schema_version {self.schema_version} was written by a "
                f"newer version of VisionForge (this build supports up to "
                f"{CURRENT_SCHEMA_VERSION}). Please upgrade VisionForge."
            )
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


def check_schema_version(version: int) -> None:
    """Reject a config whose ``schema_version`` is newer than this build supports.

    Shared by every task config so a config from a future VisionForge release
    fails loudly instead of being silently mis-parsed.
    """
    if version > CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"Config schema_version {version} was written by a newer version of "
            f"VisionForge (this build supports up to {CURRENT_SCHEMA_VERSION}). "
            f"Please upgrade VisionForge."
        )


def migrate_config_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a raw config mapping to the current schema version.

    Returns a new dict (the input is not mutated). A config without an explicit
    ``schema_version`` is treated as the initial schema (v1) — every legacy YAML
    and run.json predates the field. Future breaking schema changes add a step
    here that rewrites older shapes forward before validation.
    """
    if not isinstance(raw, dict):
        return raw
    out = dict(raw)
    version = out.get("schema_version", 1)
    # (no migration steps yet — v1 is the initial schema)
    out["schema_version"] = version
    return out


def load_config(path: Path | str) -> ExperimentConfig:
    """Load, migrate, and validate an experiment config from a YAML file.

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

    return ExperimentConfig.model_validate(migrate_config_dict(raw))


__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "TransformConfig",
    "OutputConfig",
    "ClassificationConfig",
    "DeviceConfig",
    "SchedulerConfig",
    "PreprocessingConfig",
    "PreprocessingStep",
    "GridSearchConfig",
    "RandomSearchConfig",
    "CrossValidationConfig",
    "TransferLearningConfig",
    "ModelComparisonConfig",
    "BatchPredictionConfig",
    "ExportONNXConfig",
    "load_config",
    "migrate_config_dict",
    "check_schema_version",
    "CURRENT_SCHEMA_VERSION",
]
