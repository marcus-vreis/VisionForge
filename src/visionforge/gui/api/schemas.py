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


class RunResult(BaseModel):
    """Full result of a completed experiment run."""

    run_id: str
    metrics: dict[str, Any]
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
    """One Grad-CAM overlay: source image + predicted class + overlay artifact."""

    source: str
    overlay: str
    predicted_class: int


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


class SystemInfo(BaseModel):
    """System probe for sensible UI defaults (workers, threads)."""

    cpu_count: int
    suggested_workers: int
    platform: str


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
]
