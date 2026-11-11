"""Endpoints that convert a dataset between YOLO and COCO into a new one (ADR-169).

Both directions follow the flow of the other derived-dataset tools: preview
(background: read the dataset, open the images, convert), plan (instant: the split
of each COCO file, the destination name), apply (background: write a sibling folder
and rename it into place) and cancel. The source folder is never written to; the
result is always a new folder next to it. Export and import share one job registry,
so only one conversion runs at a time.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from visionforge.core.dataset_coco import (
    CocoError,
    ExportAnalysis,
    check_export_unchanged,
    suggest_export_name,
)
from visionforge.core.dataset_coco_import import (
    ImportAnalysis,
    ImportPlan,
    check_import_unchanged,
    plan_import,
    suggest_import_name,
)
from visionforge.gui.api.coco_jobs import CocoJobs
from visionforge.gui.api.datasets_routes import _is_inside_allowed_root
from visionforge.gui.api.derived_jobs import DerivedJob, JobBusyError
from visionforge.gui.api.http_responses import (
    BAD_REQUEST_TEXT,
    CONFLICT_TEXT,
    FORBIDDEN_TEXT,
    NOT_FOUND_TEXT,
    TAG_EXPLORER,
)
from visionforge.utils.messages import tr
from visionforge.utils.paths import validate_path_component

router = APIRouter(prefix="/explorer/coco", tags=[TAG_EXPLORER])
export_router = APIRouter(prefix="/export")
import_router = APIRouter(prefix="/import")

_JOBS = CocoJobs()
_JOB_MISSING = "coco.job_missing"
_BUSY = "coco.busy"
_BAD_DEST = "coco.bad_dest_name"
_DEST_EXISTS = "coco.dest_exists"

# Refusals that mean "the state of the disk is not what the request assumed".
_CONFLICT_CODES = {_DEST_EXISTS, "coco.base_changed"}


class CocoExportPreviewRequest(BaseModel):
    """The YOLO detection dataset the explorer has opened."""

    base_dir: str


class CocoImportPreviewRequest(BaseModel):
    """The folder with the COCO annotation files and images."""

    source_dir: str


class CocoPlanRequest(BaseModel):
    """Optional: check a destination name now instead of at the apply."""

    dest_name: str | None = Field(default=None, max_length=200)


class CocoImportPlanRequest(CocoPlanRequest):
    # JSON name -> train/val/test/skip; absent means what the file name says.
    splits: dict[str, str] | None = None


class CocoApplyRequest(BaseModel):
    dest_name: str = Field(min_length=1, max_length=200)


class CocoImportApplyRequest(CocoApplyRequest):
    splits: dict[str, str] | None = None


def _refuse(exc: CocoError) -> HTTPException:
    status = 409 if exc.code in _CONFLICT_CODES else 400
    return HTTPException(status, tr(exc.code, **exc.params))


def _preview_job(job_id: str, analysis_type: type) -> DerivedJob:
    """The finished preview ``job_id``, of the export or of the import."""
    job = _JOBS.get(job_id)
    if job is None or job.kind != "preview":
        raise HTTPException(404, tr(_JOB_MISSING))
    if job.state != "done" or not isinstance(job.analysis, analysis_type):
        raise HTTPException(409, tr("coco.preview_not_ready"))
    return job


def _check_dest(parent: Path, dest_name: str) -> None:
    try:
        validate_path_component(dest_name)
    except ValueError as exc:
        raise HTTPException(400, tr(_BAD_DEST)) from exc
    if (parent / dest_name).exists():
        raise HTTPException(409, tr(_DEST_EXISTS, name=dest_name))


# ── status and cancel (the same job registry for both directions) ────────────


@export_router.get("/{job_id}", responses={404: {"description": NOT_FOUND_TEXT}})
@import_router.get("/{job_id}", responses={404: {"description": NOT_FOUND_TEXT}})
async def coco_status(job_id: str) -> dict[str, Any]:
    """Progress, and the preview or the result once the job has finished."""
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, tr(_JOB_MISSING))
    return job.snapshot()


@export_router.post(
    "/{job_id}/cancel", responses={404: {"description": NOT_FOUND_TEXT}}
)
@import_router.post(
    "/{job_id}/cancel", responses={404: {"description": NOT_FOUND_TEXT}}
)
async def coco_cancel(job_id: str) -> dict[str, bool]:
    """Ask a running job to stop. An apply removes its temporary folder."""
    if not _JOBS.cancel(job_id):
        raise HTTPException(404, tr(_JOB_MISSING))
    return {"cancelling": True}


# ── export: YOLO -> COCO ─────────────────────────────────────────────────────


@export_router.post(
    "/preview",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        403: {"description": FORBIDDEN_TEXT},
        409: {"description": CONFLICT_TEXT},
    },
)
async def coco_export_preview(req: CocoExportPreviewRequest) -> dict[str, Any]:
    """Start reading the dataset; poll ``/{job_id}`` for the preview."""
    # The dataset folder is the user's own pick in the local GUI and is only read;
    # it must also be one the explorer already opened (the thumbnails' allow-list).
    base = Path(req.base_dir)
    if not base.is_dir():  # NOSONAR pythonsecurity:S6549
        raise HTTPException(400, tr("coco.base_missing", path=req.base_dir))
    if not _is_inside_allowed_root(base.resolve()):  # NOSONAR pythonsecurity:S6549
        raise HTTPException(403, tr("coco.base_not_opened"))
    try:
        job = _JOBS.start_export_preview(
            base.resolve(),  # NOSONAR pythonsecurity:S6549
            await asyncio.to_thread(suggest_export_name, base),
        )
    except JobBusyError as exc:
        raise HTTPException(409, tr(_BUSY)) from exc
    return job.snapshot()


@export_router.post(
    "/{job_id}/plan",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        409: {"description": CONFLICT_TEXT},
    },
)
async def coco_export_plan(job_id: str, req: CocoPlanRequest) -> dict[str, Any]:
    """The preview again, and the destination name checked when one is given."""
    analysis: ExportAnalysis = _preview_job(job_id, ExportAnalysis).analysis
    if req.dest_name is not None:
        _check_dest(analysis.root.parent, req.dest_name)
    return {**analysis.summary(), "dest_name": req.dest_name}


@export_router.post(
    "/{job_id}/apply",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        409: {"description": CONFLICT_TEXT},
    },
)
async def coco_export_apply(job_id: str, req: CocoApplyRequest) -> dict[str, Any]:
    """Start writing the COCO dataset; poll ``/{job_id}`` of the returned job."""
    analysis: ExportAnalysis = _preview_job(job_id, ExportAnalysis).analysis
    _check_dest(analysis.root.parent, req.dest_name)
    try:
        # A dataset edited after the preview would make it a lie.
        await asyncio.to_thread(check_export_unchanged, analysis)
        started = _JOBS.start_export_apply(analysis, req.dest_name)
    except CocoError as exc:
        raise _refuse(exc) from exc
    except JobBusyError as exc:
        raise HTTPException(409, tr(_BUSY)) from exc
    return started.snapshot()


# ── import: COCO -> YOLO ─────────────────────────────────────────────────────


@import_router.post(
    "/preview",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        409: {"description": CONFLICT_TEXT},
    },
)
async def coco_import_preview(req: CocoImportPreviewRequest) -> dict[str, Any]:
    """Start reading the COCO folder; poll ``/{job_id}`` for the preview."""
    source = Path(req.source_dir)
    # The folder is the user's own pick in the local GUI; it is only read.
    if not source.is_dir():  # NOSONAR pythonsecurity:S6549
        raise HTTPException(400, tr("coco.source_missing", path=req.source_dir))
    try:
        job = _JOBS.start_import_preview(
            source.resolve(),  # NOSONAR pythonsecurity:S6549
            await asyncio.to_thread(suggest_import_name, source),
        )
    except JobBusyError as exc:
        raise HTTPException(409, tr(_BUSY)) from exc
    return job.snapshot()


def _import_plan(analysis: ImportAnalysis, splits: dict[str, str] | None) -> ImportPlan:
    try:
        return plan_import(analysis, splits)
    except CocoError as exc:
        raise _refuse(exc) from exc


@import_router.post(
    "/{job_id}/plan",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        409: {"description": CONFLICT_TEXT},
    },
)
async def coco_import_plan(job_id: str, req: CocoImportPlanRequest) -> dict[str, Any]:
    """Split of each COCO file and the numbering of the classes."""
    analysis: ImportAnalysis = _preview_job(job_id, ImportAnalysis).analysis
    plan = _import_plan(analysis, req.splits)
    if req.dest_name is not None:
        _check_dest(analysis.root.parent, req.dest_name)
    return plan.summary()


@import_router.post(
    "/{job_id}/apply",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        409: {"description": CONFLICT_TEXT},
    },
)
async def coco_import_apply(job_id: str, req: CocoImportApplyRequest) -> dict[str, Any]:
    """Start writing the YOLO dataset; poll ``/{job_id}`` of the returned job."""
    analysis: ImportAnalysis = _preview_job(job_id, ImportAnalysis).analysis
    _check_dest(analysis.root.parent, req.dest_name)
    plan = _import_plan(analysis, req.splits)
    try:
        plan.require_ready()
        # A folder edited after the preview would make it a lie.
        await asyncio.to_thread(check_import_unchanged, analysis)
        started = _JOBS.start_import_apply(analysis, plan, req.dest_name)
    except CocoError as exc:
        raise _refuse(exc) from exc
    except JobBusyError as exc:
        raise HTTPException(409, tr(_BUSY)) from exc
    return started.snapshot()


router.include_router(export_router)
router.include_router(import_router)

__all__ = ["router"]
