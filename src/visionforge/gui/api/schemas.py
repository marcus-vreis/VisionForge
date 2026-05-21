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


__all__ = [
    "RunStatus",
    "RunResponse",
    "RunResult",
    "RunSummary",
    "DatasetDetectRequest",
    "DatasetDetectResponse",
]
