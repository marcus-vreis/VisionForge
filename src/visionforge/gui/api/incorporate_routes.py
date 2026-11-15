"""Endpoints that add new images to a dataset as a new dataset (ADR-166).

The flow is preview (background: read, check and hash the new folder), plan
(instant: split the survivors given the accepted classes, ratios and seed), apply
(background: write a sibling folder and rename it into place) and cancel. The
opened dataset is never written to; the result is always a new folder next to it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from visionforge.core.dataset_incorporate import (
    DEFAULT_SEED,
    IncorporateError,
    plan_incorporation,
    suggest_name,
)
from visionforge.gui.api.datasets_routes import _is_inside_allowed_root
from visionforge.gui.api.explorer_routes import ExplorerRequest, _split_dirs
from visionforge.gui.api.http_responses import (
    BAD_REQUEST_TEXT,
    CONFLICT_TEXT,
    FORBIDDEN_TEXT,
    NOT_FOUND_TEXT,
    TAG_EXPLORER,
)
from visionforge.gui.api.incorporate_jobs import (
    IncorporateJob,
    IncorporateJobs,
    JobBusyError,
)
from visionforge.utils.messages import tr
from visionforge.utils.paths import validate_path_component

router = APIRouter(prefix="/explorer/incorporate", tags=[TAG_EXPLORER])

_JOBS = IncorporateJobs()
_JOB_MISSING = "incorporate.job_missing"

# Refusals that mean "the state of the disk is not what the request assumed".
_CONFLICT_CODES = {
    "incorporate.dest_exists",
    "incorporate.base_changed",
    "incorporate.source_changed",
}


class PreviewRequest(ExplorerRequest):
    """The opened dataset plus the folder with the new images."""

    source_dir: str


class PlanRequest(BaseModel):
    accept_classes: list[str] = Field(default_factory=list)
    # Per split, relative weights; absent means the dataset's own proportion.
    ratios: dict[str, float] | None = None
    seed: int = Field(default=DEFAULT_SEED, ge=0, le=2**31 - 1)


class ApplyRequest(PlanRequest):
    dest_name: str = Field(min_length=1, max_length=200)


def _refuse(exc: IncorporateError) -> HTTPException:
    status = 409 if exc.code in _CONFLICT_CODES else 400
    return HTTPException(status, tr(exc.code, **exc.params))


def _ready_job(job_id: str) -> IncorporateJob:
    job = _JOBS.get(job_id)
    if job is None or job.kind != "preview":
        raise HTTPException(404, tr(_JOB_MISSING))
    if job.state != "done" or job.analysis is None:
        raise HTTPException(409, tr("incorporate.preview_not_ready"))
    return job


@router.post(
    "/preview",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        403: {"description": FORBIDDEN_TEXT},
        409: {"description": CONFLICT_TEXT},
    },
)
async def incorporate_preview(req: PreviewRequest) -> dict[str, Any]:
    """Start reading the new folder; poll ``/{job_id}`` for the preview."""
    base = Path(req.base_dir)
    if not base.is_dir():
        raise HTTPException(400, tr("incorporate.base_missing", path=req.base_dir))
    # Only a dataset the explorer already opened (the same allow-list as thumbnails).
    if not _is_inside_allowed_root(base.resolve()):
        raise HTTPException(403, tr("incorporate.base_not_opened"))
    # The new-images folder is the user's own pick in the local GUI; it is only read.
    source = Path(req.source_dir)
    if not source.is_dir():  # NOSONAR pythonsecurity:S6549
        raise HTTPException(400, tr("incorporate.source_missing", path=req.source_dir))
    try:
        job = _JOBS.start_preview(
            base.resolve(),
            req.task,
            source.resolve(),  # NOSONAR pythonsecurity:S6549
            _split_dirs(req),
            await asyncio.to_thread(suggest_name, base),
        )
    except JobBusyError as exc:
        raise HTTPException(409, tr("incorporate.busy")) from exc
    return job.snapshot()


@router.get("/{job_id}", responses={404: {"description": NOT_FOUND_TEXT}})
async def incorporate_status(job_id: str) -> dict[str, Any]:
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
async def incorporate_plan(job_id: str, req: PlanRequest) -> dict[str, Any]:
    """Counts per class and split for the chosen classes, ratios and seed."""
    job = _ready_job(job_id)
    assert job.analysis is not None
    try:
        plan = plan_incorporation(
            job.analysis,
            accept_classes=req.accept_classes,
            ratios=req.ratios,
            seed=req.seed,
        )
    except IncorporateError as exc:
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
async def incorporate_apply(job_id: str, req: ApplyRequest) -> dict[str, Any]:
    """Start writing the new dataset; poll ``/{job_id}`` of the returned job."""
    job = _ready_job(job_id)
    analysis = job.analysis
    assert analysis is not None
    try:
        validate_path_component(req.dest_name)
    except ValueError as exc:
        raise HTTPException(400, tr("incorporate.bad_dest_name")) from exc
    if (analysis.base_root.parent / req.dest_name).exists():
        raise HTTPException(409, tr("incorporate.dest_exists", name=req.dest_name))
    try:
        plan = plan_incorporation(
            analysis,
            accept_classes=req.accept_classes,
            ratios=req.ratios,
            seed=req.seed,
        )
        started = _JOBS.start_apply(analysis, plan, req.dest_name)
    except IncorporateError as exc:
        raise _refuse(exc) from exc
    except JobBusyError as exc:
        raise HTTPException(409, tr("incorporate.busy")) from exc
    return started.snapshot()


@router.post("/{job_id}/cancel", responses={404: {"description": NOT_FOUND_TEXT}})
async def incorporate_cancel(job_id: str) -> dict[str, bool]:
    """Ask a running job to stop. An apply removes its temporary folder."""
    if not _JOBS.cancel(job_id):
        raise HTTPException(404, tr(_JOB_MISSING))
    return {"cancelling": True}


__all__ = ["router"]
