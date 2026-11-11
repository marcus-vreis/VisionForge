"""The preprocessing preview and the guide's sample dataset (#204).

``POST /api/dataset/preview_preprocess`` runs the form's preprocessing steps on
one image of a split and saves each step as a PNG; ``POST /api/sample-dataset``
writes a task's synthetic dataset for the guide (ADR-115, ADR-117).

``routes.py`` includes ``router`` where these routes used to be defined, so the
OpenAPI document and the route order do not change. A test that moves where
they write patches ``_PREVIEW_CACHE_DIR`` or ``_SAMPLE_DATASETS_DIR`` here; both
are read per request. Pillow and the filters are imported per request, so
importing this module stays light (GUI-11, #202).
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from visionforge.core.detection_data import resolve_yolo_split
from visionforge.gui.api.datasets_routes import _IMAGE_EXTS, _remember_dataset_root
from visionforge.gui.api.http_responses import (
    BAD_REQUEST_TEXT,
    CONFLICT_TEXT,
    SERVER_ERROR_TEXT,
    TAG_DATASETS,
    UNPROCESSABLE_TEXT,
    VALIDATION_ERROR_CONTENT,
)
from visionforge.gui.api.sample_dataset import (
    SampleDatasetExistsError,
    SampleTaskError,
    create_sample_dataset,
)
from visionforge.gui.api.schemas import (
    PreprocessPreviewRequest,
    PreprocessPreviewResponse,
    PreprocessPreviewStep,
    SampleDatasetRequest,
    SampleDatasetResponse,
)
from visionforge.utils.messages import tr

# Where the preview's PNGs are written, one folder per source class; served by
# GET /api/artifacts/{path}, which only serves files under outputs/.
_PREVIEW_CACHE_DIR = Path("outputs/preview_cache")
# Where the guide's synthetic sample dataset is written (ADR-115). Datasets are
# shared across profiles, so this is not under a profile folder.
_SAMPLE_DATASETS_DIR = Path("datasets")

# The key of a preprocessing step that names its filter.
_KIND = "kind"

router = APIRouter()


@router.post(
    "/dataset/preview_preprocess",
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
    },
    tags=[TAG_DATASETS],
)
async def preview_preprocess(
    req: PreprocessPreviewRequest,
) -> PreprocessPreviewResponse:
    """Generate a strip of PNG previews showing the preprocessing pipeline.

    Picks one image from the requested split (first one alphabetically inside
    ``class_name`` or the first class found) and runs every step. Each step's
    output is saved as a PNG under ``outputs/preview_cache/<run-id>/`` and
    served via the existing /api/artifacts/{path} endpoint.

    Raises:
        HTTPException: 422 when a step names a filter that does not exist —
            that is bad input, not a server fault, and the message lists the
            filters that do (ADR-078).
    """
    try:
        return await asyncio.to_thread(_render_preprocess_preview, req)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post(
    "/sample-dataset",
    status_code=201,
    response_model=SampleDatasetResponse,
    responses={
        400: {"description": BAD_REQUEST_TEXT},
        409: {"description": CONFLICT_TEXT},
        500: {"description": SERVER_ERROR_TEXT},
    },
    tags=[TAG_DATASETS],
)
async def sample_dataset(
    req: SampleDatasetRequest,
) -> SampleDatasetResponse | JSONResponse:
    """Write the synthetic sample dataset of a task for the guide (ADR-115, ADR-117).

    Local and offline: the self-test builder draws tiny images with a
    deterministic seed. Every built-in task has one, in its own layout.

    Raises:
        HTTPException: 400 for an unknown task, 409 when
            ``datasets/exemplo-<tarefa>/`` already holds files (the body carries
            ``path`` so the guide can use that folder), 500 when the OS refuses
            to write.
    """
    try:
        built = await asyncio.to_thread(
            create_sample_dataset, _SAMPLE_DATASETS_DIR, req.task
        )
    except SampleTaskError as exc:
        raise HTTPException(400, str(exc)) from exc
    except SampleDatasetExistsError as exc:
        # `path` rides next to `detail` instead of inside it, so a client that
        # only knows string details still shows a readable message.
        return JSONResponse(
            status_code=409, content={"detail": str(exc), "path": str(exc.path)}
        )
    except OSError as exc:
        logger.warning("GUI: could not write the sample dataset: {}", exc)
        raise HTTPException(500, tr("sample.write_failed", error=exc)) from exc
    _remember_dataset_root(built.path)
    logger.info("GUI: wrote the {} sample dataset at {}", req.task, built.path)
    return SampleDatasetResponse(
        task=built.task,
        path=str(built.path),
        classes=built.classes,
        counts=built.counts,
        targets=built.targets,
    )


def _empty_preview(message: str) -> PreprocessPreviewResponse:
    """A preview with no image, saying why."""
    from visionforge.core.preprocessing import available_kinds

    return PreprocessPreviewResponse(
        original="",
        steps=[],
        final="",
        source_image="",
        available_kinds=available_kinds(),
        message=message,
    )


def _preview_source(req: PreprocessPreviewRequest) -> tuple[Path, Path] | str:
    """The class folder and image to preview, or the message saying why there is none."""
    base = Path(req.base_dir)
    split_dir = base / req.split
    yolo_split = None if split_dir.is_dir() else resolve_yolo_split(base, req.split)
    if not split_dir.is_dir() and yolo_split is None:
        return tr("scan.split_missing", split=req.split)

    # Pick the first available class + first image inside it. The images/<split>
    # layout has no class folders: its images sit loose in that one directory.
    class_dirs = (
        [yolo_split[0]]
        if yolo_split is not None
        else [d for d in sorted(split_dir.iterdir()) if d.is_dir()]
    )
    if not class_dirs:
        return tr("scan.split_no_classes")

    target_class = None
    if req.class_name is not None:
        target_class = next((d for d in class_dirs if d.name == req.class_name), None)
    if target_class is None:
        target_class = class_dirs[0]

    img_path = next(
        (
            p
            for p in sorted(target_class.iterdir())
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        ),
        None,
    )
    if img_path is None:
        return tr("scan.class_no_images", path=target_class)
    return target_class, img_path


def _fresh_cache_dir(name: str) -> Path:
    """The preview folder of class ``name``, emptied of earlier renders.

    Deterministic per source class, so repeated previews don't accumulate
    hundreds of files.
    """
    cache_dir = _PREVIEW_CACHE_DIR / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob("*.png"):
        with contextlib.suppress(OSError):
            old.unlink()
    return cache_dir


def _render_preprocess_preview(
    req: PreprocessPreviewRequest,
) -> PreprocessPreviewResponse:
    """Save original + per-step previews and return their artifact paths."""
    from PIL import Image

    from visionforge.core.preprocessing import (
        apply_pipeline,
        available_kinds,
    )

    _remember_dataset_root(req.base_dir)
    found = _preview_source(req)
    if isinstance(found, str):
        return _empty_preview(found)
    target_class, img_path = found
    cache_dir = _fresh_cache_dir(target_class.name)

    # Unique token per invocation: the browser caches /dataset/file by URL, so
    # reusing a filename (99_final.png) shows the PREVIOUS render after the
    # pipeline changes. Fresh names every call — old ones were unlinked above.
    token = uuid.uuid4().hex[:8]

    original_img = Image.open(img_path).convert("RGB")
    original_path = cache_dir / f"00_original_{token}.png"
    original_img.save(original_path)

    final_img, intermediates = apply_pipeline(original_img, req.steps)

    step_records: list[PreprocessPreviewStep] = []
    for i, (step_cfg, img) in enumerate(zip(req.steps, intermediates, strict=False)):
        step_path = cache_dir / f"{i + 1:02d}_{step_cfg.get(_KIND, 'step')}_{token}.png"
        img.save(step_path)
        step_records.append(
            PreprocessPreviewStep(
                kind=str(step_cfg.get(_KIND, "")),
                artifact=str(step_path),
                params={k: v for k, v in step_cfg.items() if k != _KIND},
            )
        )

    if step_records:
        # Save a separate "final" copy so callers don't depend on intermediates.
        final_path = cache_dir / f"99_final_{token}.png"
        final_img.save(final_path)
    else:
        final_path = original_path

    return PreprocessPreviewResponse(
        original=str(original_path),
        steps=step_records,
        final=str(final_path),
        source_image=str(img_path.resolve()),
        available_kinds=available_kinds(),
    )
