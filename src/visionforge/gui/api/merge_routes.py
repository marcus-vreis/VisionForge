"""Endpoints that merge two datasets into a new one (ADR-167).

The flow is preview (background: index and hash both datasets), plan (instant:
apply a class mapping and count the result), apply (background: write a sibling
folder of dataset A and rename it into place) and cancel. Neither dataset is ever
written to; the result is always a new folder.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from visionforge.core.dataset_merge import (
    MergeError,
    check_sources,
    ensure_sources_unchanged,
    plan_merge,
    suggest_merge_name,
)
from visionforge.gui.api.datasets_routes import _is_inside_allowed_root
from visionforge.gui.api.derived_jobs import DerivedJob, JobBusyError
from visionforge.gui.api.explorer_routes import ExplorerRequest, _split_dirs
from visionforge.gui.api.http_responses import (
    BAD_REQUEST_TEXT,
    CONFLICT_TEXT,
    FORBIDDEN_TEXT,
    NOT_FOUND_TEXT,
    TAG_EXPLORER,
)
from visionforge.gui.api.merge_jobs import MergeJobs
from visionforge.utils.messages import tr
from visionforge.utils.paths import validate_path_component

router = APIRouter(prefix="/explorer/merge", tags=[TAG_EXPLORER])

_JOBS = MergeJobs()
_JOB_MISSING = "merge.job_missing"
_BUSY = "merge.busy"

# Refusals that mean "the state of the disk is not what the request assumed".
_CONFLICT_CODES = {
    "merge.dest_exists",
    "merge.a_changed",
    "merge.b_changed",
    "merge.image_changed",
}


class MergePreviewRequest(ExplorerRequest):
    """The opened dataset (A) plus the folder of the other one (B)."""

    other_dir: str
    other_train_dir: str = "train"
    other_val_dir: str = "val"
    other_test_dir: str = "test"


class MergePlanRequest(BaseModel):
    # ``{"a": {class: target or null}, "b": {...}}``; null discards the class and a
    # class left out keeps its own name.
    mapping: dict[str, dict[str, str | None]] = Field(default_factory=dict)


class MergeApplyRequest(MergePlanRequest):
    dest_name: str = Field(min_length=1, max_length=200)


def _refuse(exc: MergeError) -> HTTPException:
    status = 409 if exc.code in _CONFLICT_CODES else 400
    return HTTPException(status, tr(exc.code, **exc.params))


def _ready_job(job_id: str) -> DerivedJob:
    job = _JOBS.get(job_id)
    if job is None or job.kind != "preview":
        raise HTTPException(404, tr(_JOB_MISSING))
    if job.state != "done" or job.analysis is None:
        raise HTTPException(409, tr("merge.preview_not_ready"))
    return job


def _other_split_dirs(req: MergePreviewRequest) -> dict[str, str]:
    return {
        "train": req.other_train_dir,
        "val": req.other_val_dir,
        "test": req.other_test_dir,
    }


@router.post(
    "/preview",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        403: {"description": FORBIDDEN_TEXT},
        409: {"description": CONFLICT_TEXT},
    },
)
async def merge_preview(req: MergePreviewRequest) -> dict[str, Any]:
    """Start reading both datasets; poll ``/{job_id}`` for the preview."""
    a_dir = Path(req.base_dir)
    if not a_dir.is_dir():
        raise HTTPException(400, tr("merge.a_missing", path=req.base_dir))
    # Only a dataset the explorer already opened (the same allow-list as thumbnails).
    if not _is_inside_allowed_root(a_dir.resolve()):
        raise HTTPException(403, tr("merge.base_not_opened"))
    # The other dataset is the user's own pick in the local GUI; it is only read.
    b_dir = Path(req.other_dir)
    if not b_dir.is_dir():  # NOSONAR pythonsecurity:S6549
        raise HTTPException(400, tr("merge.b_missing", path=req.other_dir))
    a_resolved = a_dir.resolve()
    b_resolved = b_dir.resolve()  # NOSONAR pythonsecurity:S6549
    try:
        check_sources(a_resolved, b_resolved)
        job = _JOBS.start_preview(
            a_resolved,
            req.task,
            b_resolved,
            _split_dirs(req),
            _other_split_dirs(req),
            await asyncio.to_thread(suggest_merge_name, a_resolved, b_resolved),
        )
    except MergeError as exc:
        raise _refuse(exc) from exc
    except JobBusyError as exc:
        raise HTTPException(409, tr(_BUSY)) from exc
    return job.snapshot()


@router.get("/{job_id}", responses={404: {"description": NOT_FOUND_TEXT}})
async def merge_status(job_id: str) -> dict[str, Any]:
    """Progress, and the preview or the result once the job has finished."""
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, tr(_JOB_MISSING))
    return job.snapshot()


@router.post(
    "/{job_id}/plan",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        409: {"description": CONFLICT_TEXT},
    },
)
async def merge_plan(job_id: str, req: MergePlanRequest) -> dict[str, Any]:
    """What the result holds for this class mapping; nothing is hashed again."""
    job = _ready_job(job_id)
    try:
        plan = plan_merge(job.analysis, req.mapping)
    except MergeError as exc:
        raise _refuse(exc) from exc
    return plan.summary()


@router.post(
    "/{job_id}/apply",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        409: {"description": CONFLICT_TEXT},
    },
)
async def merge_apply(job_id: str, req: MergeApplyRequest) -> dict[str, Any]:
    """Start writing the merged dataset; poll ``/{job_id}`` of the returned job."""
    job = _ready_job(job_id)
    analysis = job.analysis
    try:
        validate_path_component(req.dest_name)
    except ValueError as exc:
        raise HTTPException(400, tr("merge.bad_dest_name")) from exc
    try:
        if (analysis.roots["a"].parent / req.dest_name).exists():
            raise HTTPException(409, tr("merge.dest_exists", name=req.dest_name))
        plan = plan_merge(analysis, req.mapping)
        # A source edited after the preview would make the preview a lie.
        await asyncio.to_thread(ensure_sources_unchanged, analysis)
        started = _JOBS.start_apply(analysis, plan, req.dest_name)
    except MergeError as exc:
        raise _refuse(exc) from exc
    except JobBusyError as exc:
        raise HTTPException(409, tr(_BUSY)) from exc
    return started.snapshot()


@router.post("/{job_id}/cancel", responses={404: {"description": NOT_FOUND_TEXT}})
async def merge_cancel(job_id: str) -> dict[str, bool]:
    """Ask a running job to stop. An apply removes its temporary folder."""
    if not _JOBS.cancel(job_id):
        raise HTTPException(404, tr(_JOB_MISSING))
    return {"cancelling": True}


__all__ = ["router"]
