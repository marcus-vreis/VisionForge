"""Read-only dataset explorer endpoints (ADR-165).

Summary and gallery come from an index that only lists directories; thumbnails are
resized server-side so the page never downloads a full-resolution image; the
integrity scan (corrupted files, duplicates, leakage, sizes) is a background job
with progress and cancellation.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from visionforge.core.anomaly_stats import (
    DEFAULT_LIMIT as ANOMALY_DEFAULT_LIMIT,
)
from visionforge.core.anomaly_stats import (
    MAX_LIMIT as ANOMALY_MAX_LIMIT,
)
from visionforge.core.anomaly_stats import (
    analyze_anomaly,
    images_signature,
    list_images,
)
from visionforge.core.cancellation import CancellationToken
from visionforge.core.class_similarity import (
    DEFAULT_PER_CLASS,
    MAX_PER_CLASS,
    analyze_class_similarity,
)
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
from visionforge.core.mask_stats import (
    DEFAULT_LIMIT as MASK_DEFAULT_LIMIT,
)
from visionforge.core.mask_stats import (
    MAX_LIMIT as MASK_MAX_LIMIT,
)
from visionforge.core.mask_stats import (
    analyze_masks,
    list_pairs,
    pairs_signature,
)
from visionforge.core.scan_store import load_scan, save_scan, scan_key, scan_signature
from visionforge.core.split_balance import split_balance
from visionforge.core.target_stats import analyze_targets, csv_signature
from visionforge.gui.api.dataset_scope import DatasetDir, DatasetPart, scan_store_dir
from visionforge.gui.api.datasets_routes import (
    _IMAGE_EXTS,
    SCAN_BASE_MISSING_KEY,
    _is_inside_allowed_root,
    _remember_dataset_root,
)
from visionforge.gui.api.explorer_jobs import Progress, ScanBusyError, ScanJobs
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

# One explorer job per person at a time: the scan or any per-task analysis.
SCAN_BUSY_KEY = "explorer.scan_busy"

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


class MaskAnalysisRequest(BaseModel):
    """A segmentation dataset as its config names it (ADR-234)."""

    base_dir: DatasetDir
    train_dir: DatasetPart = "train"
    val_dir: DatasetPart = "val"
    test_dir: DatasetPart = "test"
    images_subdir: DatasetPart = "images"
    masks_subdir: DatasetPart = "masks"
    num_classes: int = Field(default=2, ge=1, le=65_536)
    ignore_index: int = 255
    max_images: int = Field(default=MASK_DEFAULT_LIMIT, ge=1, le=MASK_MAX_LIMIT)


class TargetAnalysisRequest(BaseModel):
    """A regression dataset as its config names it (ADR-235)."""

    base_dir: DatasetDir
    images_dir: DatasetPart = "images"
    train_csv: DatasetPart = "train.csv"
    val_csv: DatasetPart = "val.csv"
    test_csv: DatasetPart = "test.csv"
    image_column: str = Field(default="image", min_length=1, max_length=200)
    target_columns: list[str] = Field(
        default_factory=lambda: ["target"], min_length=1, max_length=64
    )


class AnomalyAnalysisRequest(BaseModel):
    """An anomaly dataset as its config names it (ADR-236)."""

    base_dir: DatasetDir
    train_dir: DatasetPart = "train"
    test_dir: DatasetPart = "test"
    normal_dir: DatasetPart = "good"
    max_images: int = Field(default=ANOMALY_DEFAULT_LIMIT, ge=1, le=ANOMALY_MAX_LIMIT)


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
        raise HTTPException(409, tr(SCAN_BUSY_KEY)) from exc
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


def _mask_split_dirs(req: MaskAnalysisRequest) -> dict[str, str]:
    return {"train": req.train_dir, "val": req.val_dir, "test": req.test_dir}


def _analysis_root(base_dir: str) -> Path:
    """The dataset folder of a per-task analysis, resolved; 400 if it is missing."""
    base = Path(base_dir)
    if not base.is_dir():
        raise HTTPException(400, tr(SCAN_BASE_MISSING_KEY, path=base_dir))
    _remember_dataset_root(base)
    return base.resolve()


def _mask_key(root: Path, req: MaskAnalysisRequest) -> str:
    """Which analysis a report belongs to: the folders and the class settings."""
    parts = [str(root), "segmentation-masks", req.images_subdir, req.masks_subdir]
    parts += [f"{s}={d}" for s, d in sorted(_mask_split_dirs(req).items())]
    parts += [f"classes={req.num_classes}", f"ignore={req.ignore_index}"]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _mask_signature(root: Path, req: MaskAnalysisRequest) -> str:
    pairs, _ = list_pairs(
        root, _mask_split_dirs(req), req.images_subdir, req.masks_subdir
    )
    return pairs_signature(pairs)


@router.post("/masks", responses={409: {"description": CONFLICT_TEXT}})
async def explorer_masks_start(req: MaskAnalysisRequest) -> dict[str, Any]:
    """Start the mask analysis of a segmentation dataset (ADR-234).

    Progress, result and cancelling go through ``/scan/{job_id}``.
    """
    root = await asyncio.to_thread(_analysis_root, req.base_dir)
    store, key = scan_store_dir(_SCAN_STORE), _mask_key(root, req)
    signature = await asyncio.to_thread(_mask_signature, root, req)

    def work(token: CancellationToken, progress: Progress) -> dict[str, Any]:
        return analyze_masks(
            root,
            _mask_split_dirs(req),
            images_subdir=req.images_subdir,
            masks_subdir=req.masks_subdir,
            num_classes=req.num_classes,
            ignore_index=req.ignore_index,
            max_images=req.max_images,
            token=token,
            on_progress=progress,
        )

    def keep(report: dict[str, Any]) -> None:
        save_scan(store, key, signature, report)

    try:
        job = _SCANS.start_work(work, on_done=keep)
    except ScanBusyError as exc:
        raise HTTPException(409, tr(SCAN_BUSY_KEY)) from exc
    return job.snapshot()


@router.post("/masks/saved")
async def explorer_masks_saved(req: MaskAnalysisRequest) -> dict[str, Any]:
    """The last finished mask analysis of this dataset, and whether it changed."""
    root = await asyncio.to_thread(_analysis_root, req.base_dir)
    store, key = scan_store_dir(_SCAN_STORE), _mask_key(root, req)
    signature = await asyncio.to_thread(_mask_signature, root, req)
    saved = load_scan(store, key, signature)
    return saved or {"report": None, "saved_at": None, "stale": False}


def _target_csvs(req: TargetAnalysisRequest) -> dict[str, str]:
    return {"train": req.train_csv, "val": req.val_csv, "test": req.test_csv}


def _target_root(req: TargetAnalysisRequest) -> Path:
    root = _analysis_root(req.base_dir)
    # Refused here, not in the job: a CSV outside the dataset is a bad request.
    for name in (*_target_csvs(req).values(), req.images_dir):
        if not (root / name).resolve().is_relative_to(root):
            raise HTTPException(400, tr("explorer.outside_dataset", path=name))
    return root


def _target_key(root: Path, req: TargetAnalysisRequest) -> str:
    """Which analysis a report belongs to: the CSVs, columns and image folder."""
    parts = [str(root), "regression-targets", req.images_dir, req.image_column]
    parts += [f"{s}={f}" for s, f in sorted(_target_csvs(req).items())]
    parts += [f"target={t}" for t in req.target_columns]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


@router.post("/targets", responses={409: {"description": CONFLICT_TEXT}})
async def explorer_targets_start(req: TargetAnalysisRequest) -> dict[str, Any]:
    """Start the target analysis of a regression dataset (ADR-235).

    Progress, result and cancelling go through ``/scan/{job_id}``.
    """
    root = await asyncio.to_thread(_target_root, req)
    store, key = scan_store_dir(_SCAN_STORE), _target_key(root, req)
    signature = csv_signature(root, _target_csvs(req))

    def work(token: CancellationToken, progress: Progress) -> dict[str, Any]:
        return analyze_targets(
            root,
            _target_csvs(req),
            images_dir=req.images_dir,
            image_column=req.image_column,
            targets=req.target_columns,
            token=token,
            on_progress=progress,
        )

    def keep(report: dict[str, Any]) -> None:
        save_scan(store, key, signature, report)

    try:
        job = _SCANS.start_work(work, on_done=keep)
    except ScanBusyError as exc:
        raise HTTPException(409, tr(SCAN_BUSY_KEY)) from exc
    return job.snapshot()


@router.post("/targets/saved")
async def explorer_targets_saved(req: TargetAnalysisRequest) -> dict[str, Any]:
    """The last finished target analysis of this dataset, and whether it changed."""
    root = await asyncio.to_thread(_target_root, req)
    store, key = scan_store_dir(_SCAN_STORE), _target_key(root, req)
    saved = load_scan(store, key, csv_signature(root, _target_csvs(req)))
    return saved or {"report": None, "saved_at": None, "stale": False}


def _anomaly_key(root: Path, req: AnomalyAnalysisRequest) -> str:
    """Which analysis a report belongs to: the dataset and its three folders."""
    parts = [str(root), "anomaly", req.train_dir, req.test_dir, req.normal_dir]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _anomaly_signature(root: Path, req: AnomalyAnalysisRequest) -> str:
    items, _ = list_images(root, req.train_dir, req.test_dir, req.normal_dir)
    return images_signature(items)


@router.post("/anomaly", responses={409: {"description": CONFLICT_TEXT}})
async def explorer_anomaly_start(req: AnomalyAnalysisRequest) -> dict[str, Any]:
    """Start the analysis of an anomaly dataset (ADR-236).

    Progress, result and cancelling go through ``/scan/{job_id}``.
    """
    root = await asyncio.to_thread(_analysis_root, req.base_dir)
    store, key = scan_store_dir(_SCAN_STORE), _anomaly_key(root, req)
    signature = await asyncio.to_thread(_anomaly_signature, root, req)

    def work(token: CancellationToken, progress: Progress) -> dict[str, Any]:
        return analyze_anomaly(
            root,
            train_dir=req.train_dir,
            test_dir=req.test_dir,
            normal_dir=req.normal_dir,
            max_images=req.max_images,
            token=token,
            on_progress=progress,
        )

    def keep(report: dict[str, Any]) -> None:
        save_scan(store, key, signature, report)

    try:
        job = _SCANS.start_work(work, on_done=keep)
    except ScanBusyError as exc:
        raise HTTPException(409, tr(SCAN_BUSY_KEY)) from exc
    return job.snapshot()


@router.post("/anomaly/saved")
async def explorer_anomaly_saved(req: AnomalyAnalysisRequest) -> dict[str, Any]:
    """The last finished anomaly analysis of this dataset, and whether it changed."""
    root = await asyncio.to_thread(_analysis_root, req.base_dir)
    store, key = scan_store_dir(_SCAN_STORE), _anomaly_key(root, req)
    signature = await asyncio.to_thread(_anomaly_signature, root, req)
    saved = load_scan(store, key, signature)
    return saved or {"report": None, "saved_at": None, "stale": False}


__all__ = ["SPLIT_NAMES", "router"]


class ClassSimilarityRequest(ExplorerRequest):
    """Which classification dataset, and how many images per class to describe."""

    per_class: int = Field(DEFAULT_PER_CLASS, ge=1, le=MAX_PER_CLASS)


def _classes_index(req: ClassSimilarityRequest) -> DatasetIndex:
    if req.task != "classification":
        raise HTTPException(400, tr("explorer.classes_classification_only"))
    return _load_index(req, refresh=True)


def _classes_key(index: DatasetIndex, req: ClassSimilarityRequest) -> str:
    """Which comparison a report belongs to: the dataset's folders and the sample."""
    parts = [scan_key(index, _split_dirs(req)), "class-similarity", str(req.per_class)]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


@router.post(
    "/classes",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        409: {"description": CONFLICT_TEXT},
    },
)
async def explorer_classes_start(req: ClassSimilarityRequest) -> dict[str, Any]:
    """Start comparing the classes of a classification dataset (ADR-239).

    Progress, result and cancelling go through ``/scan/{job_id}``.
    """
    index = await asyncio.to_thread(_classes_index, req)
    store, key = scan_store_dir(_SCAN_STORE), _classes_key(index, req)
    signature = await asyncio.to_thread(scan_signature, index)

    def work(token: CancellationToken, progress: Progress) -> dict[str, Any]:
        return analyze_class_similarity(
            index, per_class=req.per_class, token=token, on_progress=progress
        )

    def keep(report: dict[str, Any]) -> None:
        save_scan(store, key, signature, report)

    try:
        job = _SCANS.start_work(work, on_done=keep)
    except ScanBusyError as exc:
        raise HTTPException(409, tr(SCAN_BUSY_KEY)) from exc
    return job.snapshot()


@router.post("/classes/saved", responses={400: {"description": BAD_REQUEST_TEXT}})
async def explorer_classes_saved(req: ClassSimilarityRequest) -> dict[str, Any]:
    """The last finished class comparison of this dataset, and whether it changed."""
    index = await asyncio.to_thread(_classes_index, req)
    store, key = scan_store_dir(_SCAN_STORE), _classes_key(index, req)
    signature = await asyncio.to_thread(scan_signature, index)
    saved = load_scan(store, key, signature)
    return saved or {"report": None, "saved_at": None, "stale": False}
