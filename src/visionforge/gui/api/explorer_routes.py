"""Read-only dataset explorer endpoints (ADR-165).

Summary and gallery come from an index that only lists directories; thumbnails are
resized server-side so the page never downloads a full-resolution image; the
integrity scan (corrupted files, duplicates, leakage, sizes) is a background job
with progress and cancellation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from visionforge.core.dataset_explorer import (
    DEFAULT_SCAN_LIMIT,
    MAX_SCAN_LIMIT,
    SPLIT_NAMES,
    DatasetIndex,
    ExplorerTask,
    IndexCache,
    build_index,
    contained_path,
    render_thumbnail,
    select_images,
    summarize,
)
from visionforge.core.scan_store import load_scan, save_scan, scan_key, scan_signature
from visionforge.core.split_balance import split_balance
from visionforge.gui.api.dataset_scope import DatasetDir, DatasetPart, scan_store_dir
from visionforge.gui.api.datasets_routes import (
    _IMAGE_EXTS,
    SCAN_BASE_MISSING_KEY,
    _is_inside_allowed_root,
    _remember_dataset_root,
)
from visionforge.gui.api.explorer_jobs import ScanBusyError, ScanJobs
from visionforge.gui.api.http_responses import (
    BAD_REQUEST_TEXT,
    CONFLICT_TEXT,
    FORBIDDEN_TEXT,
    NOT_FOUND_TEXT,
    TAG_EXPLORER,
    UNPROCESSABLE_TEXT,
    VALIDATION_ERROR_CONTENT,
)
from visionforge.utils.messages import tr

router = APIRouter(prefix="/explorer", tags=[TAG_EXPLORER])

_INDEXES = IndexCache()
_SCANS = ScanJobs()


class ExplorerRequest(BaseModel):
    """Which dataset to open. ``*_dir`` only matter for ImageFolder layouts."""

    base_dir: DatasetDir
    task: ExplorerTask = "classification"
    train_dir: DatasetPart = "train"
    val_dir: DatasetPart = "val"
    test_dir: DatasetPart = "test"


class ExplorerImagesRequest(ExplorerRequest):
    split: str | None = None
    class_name: str | None = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=60, ge=1, le=200)


class ExplorerScanRequest(ExplorerRequest):
    max_images: int = Field(default=DEFAULT_SCAN_LIMIT, ge=1, le=MAX_SCAN_LIMIT)
    # A second pass for visual near-copies across splits (ADR-231).
    near_duplicates: bool = False


# Finished scans, one per dataset; on a server inside each person's profile.
_SCAN_STORE = Path("outputs/explorer_scans")


def _key(req: ExplorerRequest) -> tuple[Any, ...]:
    return (
        str(Path(req.base_dir).resolve()),
        req.task,
        req.train_dir,
        req.val_dir,
        req.test_dir,
    )


def _split_dirs(req: ExplorerRequest) -> dict[str, str]:
    return {"train": req.train_dir, "val": req.val_dir, "test": req.test_dir}


def _load_index(req: ExplorerRequest, *, refresh: bool) -> DatasetIndex:
    base = Path(req.base_dir)
    if not base.is_dir():
        raise HTTPException(400, tr(SCAN_BASE_MISSING_KEY, path=req.base_dir))
    key = _key(req)
    if not refresh:
        cached = _INDEXES.get(key)
        if cached is not None:
            return cached
    _remember_dataset_root(base)
    index = build_index(base, req.task, _split_dirs(req))
    _INDEXES.put(key, index)
    return index


@router.post("/summary")
async def explorer_summary(req: ExplorerRequest) -> dict[str, Any]:
    """Counts per class and split (and boxes per class for detection).

    Always re-lists the disk, so it doubles as the refresh button.
    """
    index = await asyncio.to_thread(_load_index, req, refresh=True)
    summary = summarize(index)
    per_class = {s: v["per_class"] for s, v in summary["splits"].items()}
    return {
        "base_dir": str(index.root.resolve()),
        **summary,
        # Whether the splits share the class mix (ADR-232): counts only, cheap.
        "balance": split_balance(per_class, SPLIT_NAMES),
    }


@router.post("/images", responses={400: {"description": BAD_REQUEST_TEXT}})
async def explorer_images(req: ExplorerImagesRequest) -> dict[str, Any]:
    """One page of the gallery, filtered by split and class."""

    def work() -> dict[str, Any]:
        index = _load_index(req, refresh=False)
        total, page = select_images(
            index,
            split=req.split,
            class_name=req.class_name,
            offset=req.offset,
            limit=req.limit,
        )
        return {
            "total": total,
            "offset": req.offset,
            "limit": req.limit,
            "items": [
                {
                    "split": im.split,
                    "path": im.rel,
                    "label_path": im.label_rel,
                    "classes": sorted(set(im.classes)),
                    "boxes": len(im.classes) if index.task == "detection" else 0,
                }
                for im in page
            ],
        }

    return await asyncio.to_thread(work)


@router.get(
    "/thumb",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        403: {"description": FORBIDDEN_TEXT},
        404: {"description": NOT_FOUND_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
    },
)
def explorer_thumb(
    root: str, path: str, size: int = 160, label_path: str | None = None
) -> Response:
    """A small JPEG of one dataset image, with the boxes drawn when labeled.

    ``path`` is relative to ``root`` and ``root`` must be a dataset the explorer
    has already opened, so the endpoint cannot be used to read arbitrary files.
    """
    base = Path(root)
    if not base.is_absolute():
        raise HTTPException(400, tr("file.not_absolute"))
    try:
        # Only roots the explorer already opened are served (checked below).
        resolved_root = base.resolve(strict=True)  # NOSONAR pythonsecurity:S6549
    except OSError as exc:
        raise HTTPException(404, tr("file.not_found")) from exc
    if not _is_inside_allowed_root(resolved_root):
        raise HTTPException(403, tr("file.not_in_dataset"))
    try:
        image = contained_path(resolved_root, path)
        label = contained_path(resolved_root, label_path) if label_path else None
    except ValueError as exc:
        raise HTTPException(400, tr("file.traversal")) from exc
    except OSError as exc:
        raise HTTPException(404, tr("file.not_found")) from exc
    if image.suffix.lower() not in _IMAGE_EXTS:
        raise HTTPException(400, tr("file.not_image"))
    try:
        data = render_thumbnail(image, size, label)
    # A broken file is a 422, not a crash.
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, tr("explorer.unreadable_image")) from exc
    return Response(
        data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300"}
    )


@router.post("/scan", responses={409: {"description": CONFLICT_TEXT}})
async def explorer_scan_start(req: ExplorerScanRequest) -> dict[str, Any]:
    """Start the integrity scan in the background; poll ``/scan/{job_id}``."""
    index = await asyncio.to_thread(_load_index, req, refresh=False)
    # Resolved here: the worker thread does not see the request's scope.
    store, key = scan_store_dir(_SCAN_STORE), scan_key(index, _split_dirs(req))

    def keep(report: dict[str, Any]) -> None:
        save_scan(store, key, scan_signature(index), report)

    try:
        job = _SCANS.start(
            index, req.max_images, near_duplicates=req.near_duplicates, on_done=keep
        )
    except ScanBusyError as exc:
        raise HTTPException(409, tr("explorer.scan_busy")) from exc
    return job.snapshot()


@router.post("/scan/saved")
async def explorer_scan_saved(req: ExplorerRequest) -> dict[str, Any]:
    """The last finished scan of this dataset, and whether it changed since.

    ``report`` is None when the dataset was never scanned here.
    """
    index = await asyncio.to_thread(_load_index, req, refresh=True)
    store, key = scan_store_dir(_SCAN_STORE), scan_key(index, _split_dirs(req))
    signature = await asyncio.to_thread(scan_signature, index)
    saved = load_scan(store, key, signature)
    return saved or {"report": None, "saved_at": None, "stale": False}


@router.get("/scan/{job_id}", responses={404: {"description": NOT_FOUND_TEXT}})
async def explorer_scan_status(job_id: str) -> dict[str, Any]:
    """Progress, and the report once the scan has finished or been cancelled."""
    job = _SCANS.get(job_id)
    if job is None:
        raise HTTPException(404, tr("explorer.scan_missing"))
    return job.snapshot()


@router.post("/scan/{job_id}/cancel", responses={404: {"description": NOT_FOUND_TEXT}})
async def explorer_scan_cancel(job_id: str) -> dict[str, bool]:
    """Ask a running scan to stop; it keeps the partial report."""
    if not _SCANS.cancel(job_id):
        raise HTTPException(404, tr("explorer.scan_missing"))
    return {"cancelling": True}


__all__ = ["SPLIT_NAMES", "router"]
