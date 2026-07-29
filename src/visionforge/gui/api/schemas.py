"""Response models for the VisionForge API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RunStatus(BaseModel):
    """Current state of the experiment runner."""

    status: Literal["idle", "running", "completed", "failed"]
    run_id: str | None = None
    error: str | None = None


class RunResponse(BaseModel):
    """Response after submitting an experiment."""

    run_id: str
    status: Literal["running"] = "running"


class ComparisonRequest(BaseModel):
    """Compare N architectures for a standalone task (ADR-044).

    ``config`` is the task's base config dict (its ``model.name`` is overridden
    per trial); ``metric`` defaults to the task runner's primary metric.
    """

    config: dict[str, Any]
    model_names: list[str] = Field(min_length=2)
    metric: str | None = None


class SweepRequest(BaseModel):
    """Grid/random/Optuna hyperparameter sweep for a standalone task (ADR-045/052).

    ``config`` is the task's base config dict. For ``mode="grid"`` the
    ``search_space`` is ``{dot.path: [values]}``; for ``mode="random"`` and
    ``mode="optuna"`` each entry is
    ``{dot.path: {"type": "uniform"|"log_uniform"|"choice", ...}}`` (optuna needs
    the optional ``optuna`` extra). ``metric`` defaults to the runner's primary.
    """

    config: dict[str, Any]
    mode: Literal["grid", "random", "optuna"] = "grid"
    search_space: dict[str, Any] = Field(min_length=1)
    metric: str | None = None
    n_trials: int = Field(default=10, ge=1)
    seed: int = 0


class TaskDescriptor(BaseModel):
    """One task the GUI can render as a tab — built-in or custom (ADR-058)."""

    key: str
    label: str
    accent: str
    description: str = ""
    custom: bool = False
    metrics: dict[str, str] = Field(default_factory=dict)
    primary_metric: str = ""


class TaskListResponse(BaseModel):
    """Every renderable task: the five built-ins + registered custom tasks."""

    tasks: list[TaskDescriptor]


class TaskCvRequest(BaseModel):
    """K-fold cross-validation for a standalone task (ADR-050 + parity).

    ``config`` is the task's config dict; the pooled train samples are split
    into ``n_folds`` (each fold trains fresh and scores on its held-out part).
    Serves regression (CSV rows) and segmentation (image/mask pairs).
    """

    config: dict[str, Any]
    n_folds: int = Field(default=5, ge=2, le=20)
    shuffle: bool = True
    fold_seed: int = 42


class ReplicatesRequest(BaseModel):
    """Multi-seed replicates of one config for any task (ADR-056).

    ``config`` is the task's base config dict, trained once per seed
    (``training.seed`` overridden per replicate). Explicit ``seeds`` win;
    otherwise ``n_replicates`` consecutive seeds are derived from the config's
    own ``training.seed``. ``metric`` defaults to the runner's primary and
    names the headline aggregate of the report.
    """

    config: dict[str, Any]
    seeds: list[int] | None = Field(default=None, min_length=2)
    n_replicates: int = Field(default=5, ge=2, le=50)
    metric: str | None = None


class ReplicatedComparisonRequest(BaseModel):
    """Compare N configurations over the same seeds, with a paired test (ADR-061).

    ``variants`` maps a label to the dot-path overrides applied on top of
    ``config`` (the baseline is a variant with no overrides). Every variant is
    trained on the *same* seed list, which is what makes the comparison paired
    — same seed, same split and initialization, so the difference isolates the
    change instead of seed noise. Cost is ``len(variants) * len(seeds)``
    trainings, so the seed count is capped conservatively.
    """

    config: dict[str, Any]
    variants: dict[str, dict[str, Any]] = Field(min_length=2)
    seeds: list[int] | None = Field(default=None, min_length=2)
    n_replicates: int = Field(default=5, ge=2, le=20)
    metric: str | None = None
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)


class RunResult(BaseModel):
    """Full result of a completed experiment run."""

    run_id: str
    metrics: dict[str, Any]
    # Same shape and field name as RunDetail.metric_cis (ADR-074) so the results
    # view and the run-detail panel read one contract instead of two.
    metric_cis: dict[str, Any] = {}
    report: dict[str, Any]
    artifacts: dict[str, Any]


class RunSummary(BaseModel):
    """Summary of one historical experiment run for the history browser."""

    run_id: str
    experiment_name: str
    model_arch: str
    task: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    epochs_completed: int
    final_metrics: dict[str, float]
    preprocessing_count: int = 0
    # Defaults to "classification" so legacy run.json files (which never
    # serialized a block field) still parse without migration.
    block: str = "classification"


class DatasetDetectRequest(BaseModel):
    """Path to a candidate dataset root for split auto-detection."""

    base_dir: str


class DatasetDetectResponse(BaseModel):
    """Result of attempting to map train/val/test subdirectories."""

    base_dir: str
    detected: bool
    train_dir: str | None = None
    val_dir: str | None = None
    test_dir: str | None = None
    candidates: list[str] = []
    message: str


class DatasetPickResponse(BaseModel):
    """Result of server-side native folder picker.

    Empty path means the user cancelled or the OS dialog is unavailable.
    """

    path: str
    cancelled: bool = False
    message: str | None = None


class CheckpointPickResponse(BaseModel):
    """Result of server-side native .pth file picker.

    Empty path means the user cancelled or the OS dialog is unavailable.
    """

    path: str
    cancelled: bool = False
    message: str | None = None


class GPUInfo(BaseModel):
    """Per-GPU details for the device selector UI."""

    index: int
    name: str
    total_memory_mb: int
    compute_capability: str | None = None


class DeviceInfoResponse(BaseModel):
    """Runtime device probe — what the user can actually pick."""

    cuda_available: bool
    cuda_version: str | None = None
    cpu_name: str
    gpus: list[GPUInfo] = []


class RunDetail(BaseModel):
    """Full run record served from /api/runs/{run_id}.

    Mirrors run.json on disk plus a few computed convenience fields.
    """

    run_id: str
    experiment_name: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    device_used: str | None = None
    environment: dict[str, str] = {}
    run_dir: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    # Bootstrap interval per test metric (ADR-074). Empty for runs written
    # before it existed, and for tasks that do not keep per-sample predictions.
    metric_cis: dict[str, Any] = {}
    history: list[dict[str, Any]]
    artifacts: dict[str, Any]
    tests: list[dict[str, Any]] = []


class RunTestRequest(BaseModel):
    """Test a saved checkpoint against a new dataset path."""

    base_dir: str
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"
    label: str | None = None


class ExportOnnxRequest(BaseModel):
    """Settings for exporting a run's checkpoint to ONNX.

    ``output_onnx`` is optional — when omitted, the file is written next to
    the checkpoint as ``<run_dir>/best_model.onnx``. ``validate`` is exposed
    by alias because the literal name shadows a deprecated BaseModel
    classmethod (same trick the underlying ``ExportONNXConfig`` uses).
    """

    model_config = ConfigDict(populate_by_name=True)

    output_onnx: str | None = None
    opset_version: int = 17
    dynamic_axes: bool = True
    run_validate: bool = Field(default=True, alias="validate")
    benchmark: bool = True
    benchmark_runs: int = 50


class ExportOnnxResponse(BaseModel):
    """Result of a successful ONNX export, mirroring ExportONNXBlock.report()."""

    output_onnx: str
    file_size_bytes: int
    validation: dict[str, Any] | None = None
    benchmark: dict[str, Any] | None = None


class BatchPredictRequest(BaseModel):
    """Settings for running batch inference with a run's checkpoint.

    ``output_csv`` is optional — when omitted, predictions are written to
    ``<run_dir>/predictions/<timestamp>.csv`` so each batch run gets its own
    file without clobbering previous ones.
    """

    input_dir: str
    output_csv: str | None = None
    recursive: bool = True
    class_names: list[str] | None = None


class BatchPredictResponse(BaseModel):
    """Result of a successful batch prediction run."""

    output_csv: str
    total_processed: int
    failed_count: int
    failed_files: list[str] = []


class GradCamRequest(BaseModel):
    """Settings for generating Grad-CAM overlays from a run's checkpoint."""

    input_dir: str
    num_samples: int = Field(default=8, ge=1, le=64)
    target_class: int | None = None
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    recursive: bool = True


class GradCamItem(BaseModel):
    """One Grad-CAM overlay: source image + overlay artifact + a prediction label.

    ``predicted_class`` is the argmax class for classification/segmentation runs
    (None for regression); ``prediction`` is a human-readable label that also
    carries the regression values / segmentation target so the GUI can show it for
    every task.
    """

    source: str
    overlay: str
    predicted_class: int | None = None
    prediction: str | None = None


class GradCamResponse(BaseModel):
    """Result of a Grad-CAM run over a folder of sample images."""

    run_id: str
    count: int
    target_layer: str
    items: list[GradCamItem] = []


class RunTestResponse(BaseModel):
    """Result of a single test run on a saved checkpoint."""

    test_id: str
    run_id: str
    label: str
    base_dir: str
    timestamp: datetime
    metrics: dict[str, Any]
    artifacts: dict[str, Any] = {}


class DatasetSamplesRequest(BaseModel):
    """Request thumbnail paths for the first N images per class in a split."""

    base_dir: str
    split: Literal["train", "val", "test"] = "train"
    per_class: int = 4


class DatasetSamplesResponse(BaseModel):
    """Per-class sample image paths so the UI can show thumbnails for label sanity-check."""

    base_dir: str
    split: str
    samples: dict[str, list[str]]
    message: str | None = None


class PreprocessPreviewRequest(BaseModel):
    """Request a strip of preview images for a preprocessing pipeline."""

    base_dir: str
    split: Literal["train", "val", "test"] = "train"
    class_name: str | None = None  # None → first class found
    steps: list[dict[str, Any]] = []  # each: {kind, ...params}


class PreprocessPreviewStep(BaseModel):
    """One step's rendered preview image + the step config that produced it."""

    kind: str
    artifact: str
    params: dict[str, Any] = {}


class PreprocessPreviewResponse(BaseModel):
    """Original + per-step + final preview artifacts for the requested pipeline."""

    original: str  # artifact path
    steps: list[PreprocessPreviewStep]
    final: str  # artifact path (equals steps[-1].artifact when steps non-empty)
    source_image: str  # absolute path of the source image
    available_kinds: list[str]
    message: str | None = None


class AugmentPreviewRequest(BaseModel):
    """Request a strip of randomly-augmented variants of one dataset image."""

    base_dir: str
    split: Literal["train", "val", "test"] = "train"
    class_name: str | None = None  # None → first class found
    transforms: dict[str, Any] = {}  # validated into TransformConfig
    num_variants: int = Field(default=4, ge=1, le=12)


class AugmentPreviewResponse(BaseModel):
    """Original + N randomly-augmented variant artifacts for the source image."""

    original: str  # artifact path
    variants: list[str] = []  # artifact paths
    source_image: str  # absolute path of the source image
    active: list[str] = []  # which augmentations were applied (flip/rotation/jitter)
    message: str | None = None


class DetectionDatasetStatsRequest(BaseModel):
    """Path to a YOLO-layout dataset root for annotation distribution analysis."""

    base_dir: str


class DetectionSplitStats(BaseModel):
    """Per-split detection dataset counts.

    ``class_counts`` tallies *instances* (annotated boxes) per class, not images
    — the relevant balance signal for detection. ``unlabeled_images`` flags
    images without a matching/non-empty label file (background-only or missing
    annotations).
    """

    total_images: int
    total_annotations: int
    class_counts: dict[str, int]
    unlabeled_images: int = 0
    missing: bool = False


class DetectionDatasetStatsResponse(BaseModel):
    """Per-split annotation distribution + imbalance verdict for a YOLO dataset.

    ``imbalanced`` is True when, across all splits, the ratio
    max(count) / min(count) over per-class instance totals exceeds 2.0.
    """

    base_dir: str
    splits: dict[str, DetectionSplitStats]
    class_names: list[str]
    imbalanced: bool
    message: str | None = None


class SegmentationDatasetStatsRequest(BaseModel):
    """Paired image/mask dataset root + dir names for pre-training inspection."""

    base_dir: str
    images_subdir: str = "images"
    masks_subdir: str = "masks"
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"


class SegmentationSplitStats(BaseModel):
    """Per-split paired-dataset counts; unpaired files signal stem mismatches."""

    images: int
    masks: int
    paired: int
    unpaired_images: int = 0
    unpaired_masks: int = 0
    missing: bool = False


class SegmentationDatasetStatsResponse(BaseModel):
    """Per-split pairing stats + class ids sampled from up to 20 train masks.

    ``mask_class_ids`` helps set ``num_classes`` and spot an ``ignore_index``
    collision before GPU time.
    """

    base_dir: str
    splits: dict[str, SegmentationSplitStats]
    mask_class_ids: list[int]
    message: str | None = None


class AnomalyDatasetStatsRequest(BaseModel):
    """MVTec-style dataset root + dir names for pre-training inspection."""

    base_dir: str
    train_dir: str = "train"
    test_dir: str = "test"
    normal_dir: str = "good"


class AnomalyDatasetStatsResponse(BaseModel):
    """Normal-only train count + per-subdir test counts (normal vs defects)."""

    base_dir: str
    train_normal: int
    test_normal: int
    test_anomalous: dict[str, int]
    missing_train: bool = False
    missing_test: bool = False
    message: str | None = None


class RegressionTargetStats(BaseModel):
    """Distribution summary of one numeric target column within a split."""

    count: int
    mean: float | None = None
    min: float | None = None
    max: float | None = None


class RegressionSplitStats(BaseModel):
    """Per-CSV manifest counts; ``missing_images`` is checked on a sample."""

    rows: int
    missing_columns: list[str] = Field(default_factory=list)
    missing_images: int = 0
    checked_images: int = 0
    targets: dict[str, RegressionTargetStats] = Field(default_factory=dict)
    missing: bool = False


class RegressionDatasetStatsRequest(BaseModel):
    """CSV-manifest dataset root + column names for pre-training inspection."""

    base_dir: str
    images_dir: str = "images"
    train_csv: str = "train.csv"
    val_csv: str = "val.csv"
    test_csv: str = "test.csv"
    image_column: str = "image"
    target_columns: list[str] = Field(default_factory=lambda: ["target"])


class RegressionDatasetStatsResponse(BaseModel):
    """Per-split manifest stats + target distributions for a regression dataset."""

    base_dir: str
    splits: dict[str, RegressionSplitStats]
    message: str | None = None


class SystemInfo(BaseModel):
    """System probe for sensible UI defaults (workers, threads)."""

    cpu_count: int
    suggested_workers: int
    platform: str
    # Shown in the header so a screenshot of a bug carries its own version.
    version: str


class DatasetStatsRequest(BaseModel):
    """Path to a dataset root for class distribution analysis."""

    base_dir: str
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"


class SplitStats(BaseModel):
    """Image counts per class for a single split."""

    total_images: int
    classes: dict[str, int]
    missing: bool = False


class DatasetStatsResponse(BaseModel):
    """Per-split class distribution + imbalance verdict.

    ``imbalanced`` is True when, in any present split, the ratio
    max(count) / min(count) over classes exceeds 2.0 — a conventional rule of
    thumb past which loss weighting or resampling becomes worth considering.
    """

    base_dir: str
    splits: dict[str, SplitStats]
    class_names: list[str]
    imbalanced: bool
    message: str | None = None


class DatasetDownloadRequest(BaseModel):
    """One-shot dataset download to a local folder (ADR-055).

    ``provider`` selects the source; ``dataset`` is its identifier (e.g. "cifar10"
    for torchvision, "owner/project" for Roboflow). ``out_dir`` is the local target.
    ``limit`` caps images per class per split (None = all). Provider credentials
    (api_key/token) are passed through for the providers that need them.
    """

    provider: Literal["torchvision", "roboflow", "kaggle", "huggingface"] = (
        "torchvision"
    )
    dataset: str
    out_dir: str
    splits: list[str] = ["train", "test"]
    limit: int | None = Field(default=None, ge=1)
    api_key: str | None = None
    token: str | None = None
    version: int | None = Field(default=None, ge=1)  # Roboflow dataset version
    dataset_format: str | None = None  # Roboflow export format (e.g. folder, yolov8)


class DatasetDownloadResponse(BaseModel):
    """Result of a one-shot dataset download."""

    provider: str
    dataset: str
    out_dir: str
    total_images: int
    splits: dict[str, int] = {}
    classes: list[str] = []


__all__ = [
    "RunStatus",
    "RunResponse",
    "RunResult",
    "RunSummary",
    "RunDetail",
    "RunTestRequest",
    "RunTestResponse",
    "ExportOnnxRequest",
    "ExportOnnxResponse",
    "BatchPredictRequest",
    "BatchPredictResponse",
    "DatasetDetectRequest",
    "DatasetDetectResponse",
    "DatasetPickResponse",
    "CheckpointPickResponse",
    "DatasetStatsRequest",
    "DatasetStatsResponse",
    "DetectionDatasetStatsRequest",
    "DetectionDatasetStatsResponse",
    "DetectionSplitStats",
    "DatasetSamplesRequest",
    "DatasetSamplesResponse",
    "PreprocessPreviewRequest",
    "PreprocessPreviewResponse",
    "AugmentPreviewRequest",
    "AugmentPreviewResponse",
    "PreprocessPreviewStep",
    "SplitStats",
    "GPUInfo",
    "DeviceInfoResponse",
    "SystemInfo",
    "DatasetDownloadRequest",
    "DatasetDownloadResponse",
]


class CredentialSaveRequest(BaseModel):
    """One provider's API key, on its way to the local store."""

    provider: Literal["roboflow", "kaggle", "huggingface"]
    value: str = Field(min_length=1)


class CredentialEntry(BaseModel):
    """Whether a provider has a stored key, and its masked form.

    Never carries the real value: the GUI only needs to show that a key exists
    and which one, and the download runs server-side where the value already
    is.
    """

    saved: bool
    masked: str


class CredentialsResponse(BaseModel):
    """Stored-credential status per provider, plus where the file lives."""

    providers: dict[str, CredentialEntry]
    config_dir: str


class CustomTaskActionResponse(BaseModel):
    """Outcome of hiding, unhiding or deleting a researcher-defined task."""

    key: str
    action: Literal["hidden", "unhidden", "deleted"]
    detail: str
