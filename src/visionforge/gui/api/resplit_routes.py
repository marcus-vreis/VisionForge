"""Endpoints that re-split a dataset into a new one (ADR-168).

The flow is preview (background: index and hash the dataset, read the optional
group file), plan (instant: pool, group, stratify and count for the chosen ratios,
seed and options), apply (background: write a sibling folder and rename it into
place) and cancel. The opened dataset is never written to; the result is always a
new folder next to it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from visionforge.core.dataset_resplit import (
    DEFAULT_SEED,
    Analysis,
    Plan,
    ResplitError,
    check_unchanged,
    plan_resplit,
    suggest_name,
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
from visionforge.gui.api.resplit_jobs import ResplitJobs
from visionforge.utils.messages import tr
from visionforge.utils.paths import validate_path_component

router = APIRouter(prefix="/explorer/resplit", tags=[TAG_EXPLORER])

_JOBS = ResplitJobs()
_JOB_MISSING = "resplit.job_missing"
_BUSY = "resplit.busy"

# Refusals that mean "the state of the disk is not what the request assumed".
_CONFLICT_CODES = {
    "resplit.dest_exists",
    "resplit.base_changed",
    "resplit.group_csv_changed",
}


class ResplitPreviewRequest(ExplorerRequest):
    """The opened dataset plus, optionally, a ``path,group`` CSV (ADR-150)."""

    group_csv: str | None = None


class ResplitPlanRequest(BaseModel):
    # Per split, relative weights; absent means the pooled splits' own proportion.
    ratios: dict[str, float] | None = None
    seed: int = Field(default=DEFAULT_SEED, ge=0, le=2**31 - 1)
    lock_test: bool = True
    # ``None``: by group when a group file was given.
    by_group: bool | None = None


class ResplitApplyRequest(ResplitPlanRequest):
    dest_name: str = Field(min_length=1, max_length=200)


def _refuse(exc: ResplitError) -> HTTPException:
    status = 409 if exc.code in _CONFLICT_CODES else 400
    return HTTPException(status, tr(exc.code, **exc.params))


def _ready_job(job_id: str) -> DerivedJob:
    job = _JOBS.get(job_id)
    if job is None or job.kind != "preview":
        raise HTTPException(404, tr(_JOB_MISSING))
    if job.state != "done" or job.analysis is None:
        raise HTTPException(409, tr("resplit.preview_not_ready"))
    return job


def _plan(analysis: Analysis, req: ResplitPlanRequest) -> Plan:
    return plan_resplit(
        analysis,
        ratios=req.ratios,
        seed=req.seed,
        lock_test=req.lock_test,
        by_group=req.by_group,
    )


@router.post(
    "/preview",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        403: {"description": FORBIDDEN_TEXT},
        409: {"description": CONFLICT_TEXT},
    },
)
async def resplit_preview(req: ResplitPreviewRequest) -> dict[str, Any]:
    """Start reading the dataset; poll ``/{job_id}`` for the preview."""
    base = Path(req.base_dir)
    if not base.is_dir():
        raise HTTPException(400, tr("resplit.base_missing", path=req.base_dir))
    # Only a dataset the explorer already opened (the same allow-list as thumbnails).
    if not _is_inside_allowed_root(base.resolve()):
        raise HTTPException(403, tr("resplit.base_not_opened"))
    group_csv = (req.group_csv or "").strip() or None
    # The group file is the user's own pick in the local GUI; it is only read.
    if (
        group_csv is not None and not (base / group_csv).is_file()
    ):  # NOSONAR pythonsecurity:S6549
        raise HTTPException(400, tr("resplit.group_csv_missing", path=group_csv))
    try:
        job = _JOBS.start_preview(
            base.resolve(),
            req.task,
            _split_dirs(req),
            group_csv,
            await asyncio.to_thread(suggest_name, base),
        )
    except JobBusyError as exc:
        raise HTTPException(409, tr(_BUSY)) from exc
    return job.snapshot()


@router.get("/{job_id}", responses={404: {"description": NOT_FOUND_TEXT}})
async def resplit_status(job_id: str) -> dict[str, Any]:
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
async def resplit_plan(job_id: str, req: ResplitPlanRequest) -> dict[str, Any]:
    """Counts per class and split for the chosen ratios, seed and options."""
    job = _ready_job(job_id)
    try:
        plan = _plan(job.analysis, req)
    except ResplitError as exc:
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
async def resplit_apply(job_id: str, req: ResplitApplyRequest) -> dict[str, Any]:
    """Start writing the new dataset; poll ``/{job_id}`` of the returned job."""
    job = _ready_job(job_id)
    analysis = job.analysis
    try:
        validate_path_component(req.dest_name)
    except ValueError as exc:
        raise HTTPException(400, tr("resplit.bad_dest_name")) from exc
    try:
        if (analysis.root.parent / req.dest_name).exists():
            raise HTTPException(409, tr("resplit.dest_exists", name=req.dest_name))
        plan = _plan(analysis, req)
        # A dataset or group file edited after the preview would make it a lie.
        await asyncio.to_thread(check_unchanged, analysis)
        started = _JOBS.start_apply(analysis, plan, req.dest_name)
    except ResplitError as exc:
        raise _refuse(exc) from exc
    except JobBusyError as exc:
        raise HTTPException(409, tr(_BUSY)) from exc
    return started.snapshot()


@router.post("/{job_id}/cancel", responses={404: {"description": NOT_FOUND_TEXT}})
async def resplit_cancel(job_id: str) -> dict[str, bool]:
    """Ask a running job to stop. An apply removes its temporary folder."""
    if not _JOBS.cancel(job_id):
        raise HTTPException(404, tr(_JOB_MISSING))
    return {"cancelling": True}


__all__ = ["router"]
