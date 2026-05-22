"""Response models for the VisionForge API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


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


class RunTestResponse(BaseModel):
    """Result of a single test run on a saved checkpoint."""

    test_id: str
    run_id: str
    label: str
    base_dir: str
    timestamp: datetime
    metrics: dict[str, Any]
    artifacts: dict[str, Any] = {}


__all__ = [
    "RunStatus",
    "RunResponse",
    "RunResult",
    "RunSummary",
    "RunDetail",
    "RunTestRequest",
    "RunTestResponse",
    "DatasetDetectRequest",
    "DatasetDetectResponse",
    "DatasetPickResponse",
    "GPUInfo",
    "DeviceInfoResponse",
]
