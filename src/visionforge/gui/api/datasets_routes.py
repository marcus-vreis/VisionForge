"""Dataset, picker and credential endpoints split out of ``routes.py`` (ADR-132)."""

from __future__ import annotations

import asyncio
import base64
import csv
import io
import os
import traceback
import uuid
from collections.abc import Collection
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from loguru import logger

from visionforge.core.detection_data import resolve_yolo_split
from visionforge.gui.api.dataset_download import credentials_in_play, download_dataset
from visionforge.gui.api.schemas import (
    AnomalyDatasetStatsRequest,
    AnomalyDatasetStatsResponse,
    AugmentPreviewRequest,
    AugmentPreviewResponse,
    CheckpointPickResponse,
    CredentialEntry,
    CredentialSaveRequest,
    CredentialsResponse,
    DatasetDetectRequest,
    DatasetDetectResponse,
    DatasetDownloadRequest,
    DatasetDownloadResponse,
    DatasetPickResponse,
    DatasetSamplesRequest,
    DatasetSamplesResponse,
    DatasetStatsRequest,
    DatasetStatsResponse,
    DetectionDatasetSamplesRequest,
    DetectionDatasetSamplesResponse,
    DetectionDatasetStatsRequest,
    DetectionDatasetStatsResponse,
    DetectionSplitStats,
    RegressionDatasetStatsRequest,
    RegressionDatasetStatsResponse,
    RegressionSplitStats,
    RegressionTargetStats,
    SegmentationDatasetStatsRequest,
    SegmentationDatasetStatsResponse,
    SegmentationSplitStats,
    SplitStats,
)
from visionforge.utils.messages import tr
from visionforge.utils.redact import redact_secrets

router = APIRouter()

_BAD_REQUEST: dict[int | str, dict[str, Any]] = {
    400: {"description": "Invalid request"}
}
_FORBIDDEN: dict[int | str, dict[str, Any]] = {403: {"description": "Access denied"}}
_NOT_FOUND: dict[int | str, dict[str, Any]] = {
    404: {"description": "Resource not found"}
}
_SERVER_ERROR: dict[int | str, dict[str, Any]] = {
    500: {"description": "Operation failed"}
}

SCAN_BASE_MISSING_KEY = "scan.base_missing"
PICK_NO_TKINTER_KEY = "pick.no_tkinter"
PICK_FAILED_KEY = "pick.failed"
CANCELLED_MESSAGE = "Cancelado."
_AUGMENT_CACHE_DIR = Path("outputs/preview_cache/augment")

# Common split folder names per role. Lowercased, accent-stripped at match time.
_TRAIN_ALIASES = {"train", "training", "treino", "trains", "tr"}
_VAL_ALIASES = {
    "val",
    "valid",
    "validation",
    "validacao",
    "validação",
    "eval",
    "dev",
    "vl",
}
_TEST_ALIASES = {"test", "testing", "teste", "ts", "holdout", "hold_out"}

# Image extensions accepted both for dataset stats / sample probes and for the
# /dataset/file thumbnail endpoint.
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Dataset roots the user has explicitly probed this session — populated by
# /dataset/pick, /dataset/stats, /dataset/samples, /dataset/preview_preprocess
# and /runs/{id}/test. The /dataset/file endpoint refuses paths that aren't
# inside one of these roots, so a crafted request cannot read arbitrary files
# even though the server runs locally with full FS access.
_ALLOWED_DATASET_ROOTS: set[Path] = set()


def _remember_dataset_root(base_dir: str | Path) -> None:
    """Add a resolved dataset root to the allow-list for /dataset/file."""
    try:
        resolved = Path(base_dir).resolve()
        if resolved.is_dir():
            _ALLOWED_DATASET_ROOTS.add(resolved)
    except OSError:
        pass


def _is_inside_allowed_root(path: Path) -> bool:
    """True iff ``path`` resolves inside one of the remembered dataset roots."""
    for root in _ALLOWED_DATASET_ROOTS:
        try:
            if path.is_relative_to(root):
                return True
        except ValueError:
            continue
    return False


@router.post("/dataset/detect")
async def detect_dataset(req: DatasetDetectRequest) -> DatasetDetectResponse:
    """Scan a base_dir for standard train/val/test split subdirectories."""
    return _detect_dataset_layout(req.base_dir)


@router.post("/dataset/preview_augment")
async def preview_augment(req: AugmentPreviewRequest) -> AugmentPreviewResponse:
    """Generate a strip of randomly-augmented variants of one dataset image.

    Applies the configured train-time augmentations (horizontal flip, rotation,
    color jitter) to a sample image ``num_variants`` times so the user can see
    the augmentation effect before training. PNGs are written under
    ``outputs/preview_cache/augment/`` and served via /api/artifacts/{path}.
    """
    return await asyncio.to_thread(_render_augment_preview, req)


@router.post("/dataset/samples")
async def dataset_samples(req: DatasetSamplesRequest) -> DatasetSamplesResponse:
    """Return the first N image paths per class in a chosen split.

    The frontend renders thumbnails so the researcher can verify labels are
    correct (i.e. ``class_a`` folder really does contain class A images).
    """
    return await asyncio.to_thread(_collect_dataset_samples, req)


@router.post("/dataset/stats")
async def dataset_stats(req: DatasetStatsRequest) -> DatasetStatsResponse:
    """Count images per class in train/val/test and flag imbalance.

    Walks the standard ImageFolder layout and counts files with image
    extensions. Splits whose directory is missing are reported with
    ``missing=true`` instead of raising — partial datasets are common during
    a first-time setup.
    """
    return await asyncio.to_thread(_collect_dataset_stats, req)


@router.post("/checkpoint/pick")
async def pick_checkpoint_file() -> CheckpointPickResponse:
    """Open a native file picker (filtered to .pth/.pt) and return the path.

    Used by the ModelConfig.weights_path field — researchers loading a custom
    pretrained checkpoint into the architecture instead of ImageNet weights.
    """
    return await asyncio.to_thread(_open_native_checkpoint_dialog)


@router.post("/dataset/pick")
async def pick_dataset_folder() -> DatasetPickResponse:
    """Open a native folder picker on the server and return the absolute path.

    Because VisionForge runs locally (browser + Python on the same machine),
    we can open a tkinter dialog server-side so the user gets a *real* absolute
    path — something the browser's File System Access API never provides.
    """
    return await asyncio.to_thread(_open_native_folder_dialog)


@router.post("/dataset/download", responses={**_BAD_REQUEST, **_SERVER_ERROR})
async def dataset_download(req: DatasetDownloadRequest) -> DatasetDownloadResponse:
    """One-shot download of a dataset to a local folder (ADR-055).

    Runs in a worker thread (downloads can be slow). A missing credential — or a
    damaged install — surfaces as a 400 with a clear message.

    Nothing that leaves here — the HTTP detail or the log — may carry the
    provider's key. The Roboflow client puts it in the URL it posts to, so an
    offline machine raises "Max retries exceeded with url: /?api_key=<KEY>", and
    that text used to be shown in the GUI and written to the server log. Both
    messages go through ``redact_secrets`` with the credentials the request could
    have used, and the exception itself is never handed to the logger: its
    traceback repeats the message, and ``diagnose`` would add the variables.
    """
    try:
        return await asyncio.to_thread(_execute_dataset_download, req)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        secrets = credentials_in_play(req.api_key, req.token)
        # `from None`: whoever formats this HTTPException must not walk back to
        # the original exception and its unredacted message.
        raise HTTPException(400, redact_secrets(str(exc), *secrets)) from None
    except Exception as exc:  # noqa: BLE001
        secrets = credentials_in_play(req.api_key, req.token)
        detail = redact_secrets(f"{type(exc).__name__}: {exc}", *secrets)
        trace = redact_secrets("".join(traceback.format_exception(exc)), *secrets)
        logger.error(
            "Dataset download ({}) failed: {}\n{}", req.provider, detail, trace
        )
        raise HTTPException(500, detail) from None


def _execute_dataset_download(req: DatasetDownloadRequest) -> DatasetDownloadResponse:
    """Dispatch a dataset download to the chosen provider and shape the response."""
    result = download_dataset(
        req.provider,
        dataset=req.dataset,
        out_dir=req.out_dir,
        splits=tuple(req.splits),
        limit=req.limit,
        api_key=req.api_key,
        token=req.token,
        version=req.version,
        dataset_format=req.dataset_format,
    )
    _remember_dataset_root(Path(result.out_dir))
    return DatasetDownloadResponse(
        provider=result.provider,
        dataset=result.dataset,
        out_dir=result.out_dir,
        total_images=result.total_images,
        splits=result.splits,
        classes=result.classes,
    )


@router.post("/detection/dataset/pick_yaml")
async def pick_detection_yaml() -> DatasetPickResponse:
    """Open a native file picker (filtered to .yaml/.yml) for a data.yaml.

    Detection datasets in the standard Ultralytics/Roboflow format ship a
    ``data.yaml`` describing splits and class names; this lets the user point
    at it directly instead of synthesizing one from a folder layout.
    """
    return await asyncio.to_thread(_open_native_yaml_dialog)


@router.post("/detection/dataset/stats")
async def detection_dataset_stats(
    req: DetectionDatasetStatsRequest,
) -> DetectionDatasetStatsResponse:
    """Per-split instance distribution for a YOLO-layout detection dataset.

    The detection analogue of /dataset/stats: instead of one folder per class,
    it counts annotated boxes per class across the ``.txt`` label files so the
    researcher can spot class imbalance and unlabeled images before training.
    """
    return await asyncio.to_thread(_collect_detection_dataset_stats, req)


@router.post("/detection/dataset/samples")
async def detection_dataset_samples(
    req: DetectionDatasetSamplesRequest,
) -> DetectionDatasetSamplesResponse:
    """One or more box crops per class, so labels can be eyeballed before training.

    The detection analogue of /dataset/samples. That one returns file paths
    because a classification class *is* a folder; here a class is a box inside an
    image that may contain several, so a path would show a picture and leave the
    researcher guessing which part of it the label meant.
    """
    return await asyncio.to_thread(_collect_detection_dataset_samples, req)


@router.post("/segmentation/dataset/stats")
async def segmentation_dataset_stats(
    req: SegmentationDatasetStatsRequest,
) -> SegmentationDatasetStatsResponse:
    """Per-split image/mask pairing stats for a paired segmentation dataset.

    Unpaired files signal filename-stem mismatches before GPU time; the class
    ids sampled from train masks help set ``num_classes`` and spot an
    ``ignore_index`` collision.
    """
    return await asyncio.to_thread(_collect_segmentation_dataset_stats, req)


@router.post("/anomaly/dataset/stats")
async def anomaly_dataset_stats(
    req: AnomalyDatasetStatsRequest,
) -> AnomalyDatasetStatsResponse:
    """Normal-vs-defect counts for an MVTec-style anomaly dataset."""
    return await asyncio.to_thread(_collect_anomaly_dataset_stats, req)


@router.post("/regression/dataset/stats")
async def regression_dataset_stats(
    req: RegressionDatasetStatsRequest,
) -> RegressionDatasetStatsResponse:
    """Manifest row counts, column checks and target distributions per split."""
    return await asyncio.to_thread(_collect_regression_dataset_stats, req)


@router.get("/credentials")
async def get_credentials() -> CredentialsResponse:
    """Which dataset providers have a key stored, masked."""
    from visionforge.utils.credentials import config_dir, credential_status

    return CredentialsResponse(
        providers={k: CredentialEntry(**v) for k, v in credential_status().items()},
        config_dir=str(config_dir()),
    )


@router.post("/credentials", responses=_BAD_REQUEST)
async def save_credential_route(req: CredentialSaveRequest) -> CredentialsResponse:
    """Store (or replace) one provider's key, then return the masked status."""
    from visionforge.utils.credentials import config_dir, credential_status
    from visionforge.utils.credentials import save_credential as _save

    try:
        _save(req.provider, req.value)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return CredentialsResponse(
        providers={k: CredentialEntry(**v) for k, v in credential_status().items()},
        config_dir=str(config_dir()),
    )


@router.delete("/credentials/{provider}")
async def forget_credential_route(provider: str) -> CredentialsResponse:
    """Remove one provider's stored key."""
    from visionforge.utils.credentials import config_dir, credential_status
    from visionforge.utils.credentials import forget_credential as _forget

    _forget(provider)
    return CredentialsResponse(
        providers={k: CredentialEntry(**v) for k, v in credential_status().items()},
        config_dir=str(config_dir()),
    )


@router.get("/dataset/file", responses={**_BAD_REQUEST, **_FORBIDDEN, **_NOT_FOUND})
async def serve_dataset_file(path: str) -> FileResponse:
    """Serve a local image file for dataset preview thumbnails.

    Defense in depth — even though VisionForge runs locally, the path must:
    1. resolve to a real file
    2. carry a known image extension
    3. sit inside one of the dataset roots the user has already probed via
       /dataset/pick, /dataset/stats, /dataset/samples, etc.

    Without #3 a crafted query could read any file the server process can
    open. Restricting to the session's allow-list keeps the endpoint useful
    for the thumbnail UI without exposing arbitrary FS reads.
    """
    # Fail fast on obvious traversal before we even resolve. The allow-list
    # check below is the authoritative guard, but these two early checks make
    # the intent obvious to readers (and to taint-analysis tools).
    if ".." in path.replace("\\", "/").split("/"):
        raise HTTPException(400, tr("file.traversal"))
    candidate = Path(path)
    if not candidate.is_absolute():
        raise HTTPException(400, tr("file.not_absolute"))
    try:
        resolved = candidate.resolve(strict=True)  # NOSONAR pythonsecurity:S6549
    except OSError as exc:
        raise HTTPException(404, tr("file.not_found")) from exc
    if resolved.suffix.lower() not in _IMAGE_EXTS:
        raise HTTPException(400, tr("file.not_image"))
    if not _is_inside_allowed_root(resolved):
        raise HTTPException(403, tr("file.not_in_dataset"))
    return FileResponse(resolved)  # NOSONAR python:S2083


def _first_image_in(directory: Path) -> Path | None:
    """Return the first image file directly inside ``directory`` (sorted), if any."""
    return next(
        (
            p
            for p in sorted(directory.iterdir())
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        ),
        None,
    )


def _first_image_recursive(base: Path, max_entries: int = 5000) -> Path | None:
    """Walk ``base`` breadth-first (sorted) and return the first image found.

    Bounded by ``max_entries`` visited paths so a huge dataset root cannot make
    the preview endpoints crawl the whole disk.
    """
    queue: list[Path] = [base]
    visited = 0
    while queue:
        current = queue.pop(0)
        subdirs: list[Path] = []
        try:
            for p in sorted(current.iterdir()):
                visited += 1
                if visited > max_entries:
                    return None
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                    return p
                if p.is_dir():
                    subdirs.append(p)
        except OSError:
            continue
        queue.extend(subdirs)
    return None


def _pick_preview_image(
    base_dir: str, split: str, class_name: str | None
) -> Path | None:
    """Return a representative sample image for the preview endpoints.

    Tries the classification ImageFolder layout first (``base_dir/split/class/``),
    then degrades gracefully for the other tasks' layouts: a split dir with loose
    files (regression ``images/``), and finally a bounded recursive scan of
    ``base_dir`` (CSV-manifest and other custom layouts).
    """
    _remember_dataset_root(base_dir)
    base = Path(base_dir)
    split_dir = base / split
    if split_dir.is_dir():
        class_dirs = [d for d in sorted(split_dir.iterdir()) if d.is_dir()]
        target = None
        if class_name is not None:
            target = next((d for d in class_dirs if d.name == class_name), None)
        if target is None and class_dirs:
            target = class_dirs[0]
        if target is not None:
            found = _first_image_in(target)
            if found is not None:
                return found
        # Split exists but has no class subdirs (or they held no images):
        # accept loose files directly under the split.
        found = _first_image_in(split_dir)
        if found is not None:
            return found
    if not base.is_dir():
        return None
    return _first_image_recursive(base)


def _render_augment_preview(req: AugmentPreviewRequest) -> AugmentPreviewResponse:
    """Save N randomly-augmented variants of one dataset image and return paths."""
    import random

    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    from PIL import Image

    from visionforge.utils.config import TransformConfig

    img_path = _pick_preview_image(req.base_dir, req.split, req.class_name)
    if img_path is None:
        return AugmentPreviewResponse(
            original="",
            variants=[],
            source_image="",
            active=[],
            message=tr("scan.split_no_images", split=req.split),
        )

    tc = TransformConfig.model_validate(req.transforms)
    size = tc.image_size
    active: list[str] = []
    if tc.horizontal_flip:
        active.append("flip")
    if tc.rotation_degrees > 0:
        active.append("rotation")
    if tc.color_jitter:
        active.append("jitter")
    # Drawn from a generator of its own, never torch's or Python's global one:
    # this runs in a thread while a job may be training in another, and the job's
    # augmentation and shuffling draw from the global generator, so a preview
    # that drew from it too would change a seeded run's result (ADR-121).
    rng = random.Random()
    resize = T.Compose([T.Resize(size), T.CenterCrop(size)])

    def transform(img: Any) -> Any:
        img = resize(img)
        if tc.horizontal_flip and rng.random() < 0.5:
            img = TF.hflip(img)
        if tc.rotation_degrees > 0:
            img = TF.rotate(img, rng.uniform(-tc.rotation_degrees, tc.rotation_degrees))
        if tc.color_jitter:  # the same ranges as the training transform's ColorJitter
            img = TF.adjust_brightness(img, rng.uniform(0.8, 1.2))
            img = TF.adjust_contrast(img, rng.uniform(0.8, 1.2))
            img = TF.adjust_saturation(img, rng.uniform(0.8, 1.2))
        return img

    cache_dir = _AUGMENT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob("*.png"):
        try:
            old.unlink()
        except OSError:
            pass

    # Unique names per invocation — same browser-cache pitfall as the
    # preprocess preview above.
    token = uuid.uuid4().hex[:8]

    original_img = Image.open(img_path).convert("RGB")
    original_path = cache_dir / f"00_original_{token}.png"
    T.Compose([T.Resize(size), T.CenterCrop(size)])(original_img).save(original_path)

    variants: list[str] = []
    for i in range(req.num_variants):
        variant_path = cache_dir / f"{i + 1:02d}_variant_{token}.png"
        transform(original_img).save(variant_path)
        variants.append(str(variant_path))

    return AugmentPreviewResponse(
        original=str(original_path),
        variants=variants,
        source_image=str(img_path.resolve()),
        active=active,
    )


def _collect_dataset_samples(req: DatasetSamplesRequest) -> DatasetSamplesResponse:
    """Walk the chosen split and return up to ``per_class`` image paths per class."""
    _remember_dataset_root(req.base_dir)
    base = Path(req.base_dir)
    split_dir = base / req.split
    if not split_dir.is_dir():
        return DatasetSamplesResponse(
            base_dir=req.base_dir,
            split=req.split,
            samples={},
            message=tr("scan.split_missing_in", split=req.split, path=base),
        )

    samples: dict[str, list[str]] = {}
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        picks: list[str] = []
        for p in sorted(class_dir.iterdir()):
            if len(picks) >= req.per_class:
                break
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                picks.append(str(p.resolve()))
        samples[class_dir.name] = picks

    return DatasetSamplesResponse(
        base_dir=str(base.resolve()),
        split=req.split,
        samples=samples,
    )


def _collect_dataset_stats(req: DatasetStatsRequest) -> DatasetStatsResponse:
    """Walk an ImageFolder layout and count per-class image files per split."""
    _remember_dataset_root(req.base_dir)
    base = Path(req.base_dir)
    if not base.exists() or not base.is_dir():
        return DatasetStatsResponse(
            base_dir=req.base_dir,
            splits={},
            class_names=[],
            imbalanced=False,
            message=tr(SCAN_BASE_MISSING_KEY, path=req.base_dir),
        )

    split_map = {"train": req.train_dir, "val": req.val_dir, "test": req.test_dir}
    splits: dict[str, SplitStats] = {}
    all_class_names: set[str] = set()

    for split_label, subdir in split_map.items():
        split_dir = base / subdir
        if not split_dir.is_dir():
            splits[split_label] = SplitStats(total_images=0, classes={}, missing=True)
            continue

        counts: dict[str, int] = {}
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_count = sum(
                1
                for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
            )
            counts[class_dir.name] = class_count
            all_class_names.add(class_dir.name)

        splits[split_label] = SplitStats(
            total_images=sum(counts.values()), classes=counts, missing=False
        )

    imbalanced = False
    for s in splits.values():
        non_zero = [c for c in s.classes.values() if c > 0]
        if len(non_zero) >= 2 and max(non_zero) / min(non_zero) > 2.0:
            imbalanced = True
            break

    return DatasetStatsResponse(
        base_dir=str(base.resolve()),
        splits=splits,
        class_names=sorted(all_class_names),
        imbalanced=imbalanced,
    )


def _read_yolo_class_names(base: Path) -> list[str]:
    """Class names from a ``classes.txt``/``names.txt`` at the root, else []."""
    for candidate in ("classes.txt", "names.txt"):
        path = base / candidate
        if path.is_file():
            names = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if names:
                return names
    return []


def _crop_to_data_uri(
    image_path: Path, box: tuple[float, float, float, float]
) -> str | None:
    """Crop a YOLO-normalized box and return it as a PNG data URI.

    Returns None rather than raising: a sample thumbnail is a convenience, and a
    single unreadable image must not fail the whole panel.
    """
    from PIL import Image

    try:
        with Image.open(image_path) as opened:
            img = opened.convert("RGB")
            w, h = img.size
            cx, cy, bw, bh = box
            left = max(int((cx - bw / 2) * w), 0)
            top = max(int((cy - bh / 2) * h), 0)
            right = min(int((cx + bw / 2) * w), w)
            bottom = min(int((cy + bh / 2) * h), h)
            if right - left < 2 or bottom - top < 2:
                return None
            crop = img.crop((left, top, right, bottom))
            crop.thumbnail((160, 160))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
    # a bad image costs one thumbnail, not the panel
    except Exception:  # noqa: BLE001
        return None
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _collect_detection_dataset_samples(
    req: DetectionDatasetSamplesRequest,
) -> DetectionDatasetSamplesResponse:
    """Crop up to ``per_class`` boxes per class from the requested split."""
    base = Path(req.base_dir).expanduser()
    if not base.is_dir():
        return DetectionDatasetSamplesResponse(
            base_dir=req.base_dir,
            split=req.split,
            crops={},
            message=tr("scan.dir_missing", path=req.base_dir),
        )

    resolved = resolve_yolo_split(base, req.split)
    if resolved is None:
        return DetectionDatasetSamplesResponse(
            base_dir=req.base_dir,
            split=req.split,
            crops={},
            message=tr("scan.split_missing_dataset", split=req.split),
        )

    images_dir, labels_dir = resolved
    names = _read_yolo_class_names(base)
    crops: dict[str, list[str]] = {}

    for image_path in sorted(images_dir.iterdir()):
        _sample_detection_image(image_path, labels_dir, names, req.per_class, crops)

    return DetectionDatasetSamplesResponse(
        base_dir=req.base_dir,
        split=req.split,
        crops=crops,
        message=None if crops else tr("scan.no_boxes"),
    )


def _sample_detection_image(
    image_path: Path,
    labels_dir: Path,
    names: list[str],
    per_class: int,
    crops: dict[str, list[str]],
) -> None:
    """Add bounded crops for one image with a matching YOLO label file."""
    if not image_path.is_file() or image_path.suffix.lower() not in _IMAGE_EXTS:
        return
    label_path = labels_dir / f"{image_path.stem}.txt"
    if not label_path.is_file():
        return
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parsed = _parse_sample_box(line)
        if parsed is None:
            continue
        cid, box = parsed
        name = names[cid] if 0 <= cid < len(names) else f"class_{cid}"
        if len(crops.get(name, [])) >= per_class:
            continue
        uri = _crop_to_data_uri(image_path, box)
        if uri is not None:
            crops.setdefault(name, []).append(uri)


def _parse_sample_box(
    line: str,
) -> tuple[int, tuple[float, float, float, float]] | None:
    """Parse the first five YOLO fields used for a crop, ignoring later fields."""
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        cid = int(float(parts[0]))
        x, y, width, height = (float(v) for v in parts[1:5])
    except ValueError:
        return None
    return cid, (x, y, width, height)


def _count_split_instances(
    images_dir: Path, labels_dir: Path
) -> tuple[int, dict[int, int], int]:
    """Return (image_count, per-class-id instance counts, unlabeled_image_count).

    An image is "unlabeled" when it has no matching label file or none of its
    lines is a valid ``class cx cy w h`` row.
    """
    images = [
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    ]
    counts: dict[int, int] = {}
    unlabeled = 0
    for img in images:
        label_path = labels_dir / f"{img.stem}.txt"
        class_ids = _label_class_ids(label_path)
        for cls in class_ids:
            counts[cls] = counts.get(cls, 0) + 1
        if not class_ids:
            unlabeled += 1
    return len(images), counts, unlabeled


def _label_class_ids(label_path: Path) -> list[int]:
    """Read class IDs from exactly five-field YOLO annotation rows."""
    if not label_path.is_file():
        return []
    ids: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            ids.append(int(float(parts[0])))
        except ValueError:
            continue
    return ids


def _collect_detection_dataset_stats(
    req: DetectionDatasetStatsRequest,
) -> DetectionDatasetStatsResponse:
    """Tally per-class annotation instances across a YOLO dataset's splits."""
    _remember_dataset_root(req.base_dir)
    base = Path(req.base_dir)
    if not base.exists() or not base.is_dir():
        return DetectionDatasetStatsResponse(
            base_dir=req.base_dir,
            splits={},
            class_names=[],
            imbalanced=False,
            message=tr(SCAN_BASE_MISSING_KEY, path=req.base_dir),
        )

    declared_names = _read_yolo_class_names(base)

    # Scan each split once, keeping raw int-id counts + a missing flag.
    raw: dict[str, tuple[int, dict[int, int], int, bool]] = {}
    resolved_dirs: dict[str, tuple[str, str]] = {}
    max_class_id = -1
    for split in ("train", "val", "test"):
        resolved = resolve_yolo_split(base, split)
        if resolved is None:
            raw[split] = (0, {}, 0, True)
            continue
        image_count, counts, unlabeled = _count_split_instances(*resolved)
        raw[split] = (image_count, counts, unlabeled, False)
        resolved_dirs[split] = (
            resolved[0].relative_to(base).as_posix(),
            resolved[1].relative_to(base).as_posix(),
        )
        if counts:
            max_class_id = max(max_class_id, *counts.keys())

    # Declared names win; otherwise generate up to the highest class id seen.
    num_classes = max(len(declared_names), max_class_id + 1)
    class_names = [
        declared_names[i] if i < len(declared_names) else f"class_{i}"
        for i in range(num_classes)
    ]

    def _name(cid: int) -> str:
        return class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"

    splits: dict[str, DetectionSplitStats] = {}
    aggregate: dict[str, int] = {}
    for split, (image_count, counts, unlabeled, is_missing) in raw.items():
        named_counts: dict[str, int] = {}
        for cid, n in counts.items():
            named = _name(cid)
            named_counts[named] = named_counts.get(named, 0) + n
            aggregate[named] = aggregate.get(named, 0) + n
        splits[split] = DetectionSplitStats(
            total_images=image_count,
            total_annotations=sum(counts.values()),
            class_counts=named_counts,
            unlabeled_images=unlabeled,
            missing=is_missing,
            images_dir=resolved_dirs.get(split, (None, None))[0],
            labels_dir=resolved_dirs.get(split, (None, None))[1],
        )

    non_zero = [c for c in aggregate.values() if c > 0]
    imbalanced = len(non_zero) >= 2 and max(non_zero) / min(non_zero) > 2.0

    message = None
    if all(m for _, _, _, m in raw.values()):
        message = tr("scan.no_yolo_split")

    return DetectionDatasetStatsResponse(
        base_dir=str(base.resolve()),
        splits=splits,
        class_names=class_names,
        imbalanced=imbalanced,
        message=message,
    )


def _image_stems(
    directory: Path, extensions: Collection[str] = _IMAGE_EXTS
) -> set[str]:
    """Filename stems of the files with ``extensions`` directly inside ``directory``."""
    if not directory.is_dir():
        return set()
    return {
        p.stem
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    }


def _count_images(directory: Path) -> int:
    """Number of image files directly inside ``directory`` (0 when missing)."""
    if not directory.is_dir():
        return 0
    return sum(
        1
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


def _collect_segmentation_dataset_stats(
    req: SegmentationDatasetStatsRequest,
) -> SegmentationDatasetStatsResponse:
    """Pair images↔masks by filename stem per split + sample train class ids."""
    base = Path(req.base_dir).expanduser()
    if not base.is_dir():
        return SegmentationDatasetStatsResponse(
            base_dir=req.base_dir,
            splits={},
            mask_class_ids=[],
            message=tr(SCAN_BASE_MISSING_KEY, path=req.base_dir),
        )

    splits: dict[str, SegmentationSplitStats] = {}
    for role, dname in (
        ("train", req.train_dir),
        ("val", req.val_dir),
        ("test", req.test_dir),
    ):
        splits[role] = _segmentation_split_stats(base / dname, req)

    class_ids = _sample_mask_class_ids(base / req.train_dir / req.masks_subdir)

    message = None
    if all(s.missing for s in splits.values()):
        message = tr("scan.no_seg_split")

    return SegmentationDatasetStatsResponse(
        base_dir=str(base.resolve()),
        splits=splits,
        mask_class_ids=class_ids,
        message=message,
    )


def _segmentation_split_stats(
    split_dir: Path, req: SegmentationDatasetStatsRequest
) -> SegmentationSplitStats:
    """Count image and mask stem matches in one segmentation split."""
    if not split_dir.is_dir():
        return SegmentationSplitStats(images=0, masks=0, paired=0, missing=True)
    imgs = _image_stems(split_dir / req.images_subdir)
    masks = _image_stems(split_dir / req.masks_subdir)
    return SegmentationSplitStats(
        images=len(imgs),
        masks=len(masks),
        paired=len(imgs & masks),
        unpaired_images=len(imgs - masks),
        unpaired_masks=len(masks - imgs),
    )


def _sample_mask_class_ids(mask_dir: Path) -> list[int]:
    """Read class IDs from at most 20 decodable train masks."""
    if not mask_dir.is_dir():
        return []
    import numpy as np
    from PIL import Image

    class_ids: set[int] = set()
    sampled = 0
    for path in sorted(mask_dir.iterdir()):
        if sampled >= 20:
            break
        if not (path.is_file() and path.suffix.lower() in _IMAGE_EXTS):
            continue
        try:
            with Image.open(path) as image:
                class_ids.update(
                    int(value) for value in np.unique(np.asarray(image.convert("L")))
                )
            sampled += 1
        except OSError:
            continue
    return sorted(class_ids)


def _collect_anomaly_dataset_stats(
    req: AnomalyDatasetStatsRequest,
) -> AnomalyDatasetStatsResponse:
    """Count normal train images + normal/defect test images per subdir."""
    base = Path(req.base_dir).expanduser()
    if not base.is_dir():
        return AnomalyDatasetStatsResponse(
            base_dir=req.base_dir,
            train_normal=0,
            test_normal=0,
            test_anomalous={},
            missing_train=True,
            missing_test=True,
            message=tr(SCAN_BASE_MISSING_KEY, path=req.base_dir),
        )

    train_normal_dir = base / req.train_dir / req.normal_dir
    test_dir = base / req.test_dir
    train_normal = _count_images(train_normal_dir)
    test_normal = _count_images(test_dir / req.normal_dir)
    test_anomalous: dict[str, int] = {}
    if test_dir.is_dir():
        for sub in sorted(d for d in test_dir.iterdir() if d.is_dir()):
            if sub.name == req.normal_dir:
                continue
            test_anomalous[sub.name] = _count_images(sub)

    message = None
    if not train_normal_dir.is_dir():
        message = tr(
            "scan.anomaly_train_missing",
            train_dir=req.train_dir,
            normal_dir=req.normal_dir,
        )
    elif train_normal == 0:
        message = tr("scan.anomaly_train_empty")

    return AnomalyDatasetStatsResponse(
        base_dir=str(base.resolve()),
        train_normal=train_normal,
        test_normal=test_normal,
        test_anomalous=test_anomalous,
        missing_train=not train_normal_dir.is_dir(),
        missing_test=not test_dir.is_dir(),
        message=message,
    )


def _collect_regression_dataset_stats(
    req: RegressionDatasetStatsRequest,
) -> RegressionDatasetStatsResponse:
    """Row counts, column checks, target distributions and a sampled image check.

    Reads at most 50k rows per CSV and verifies image paths for the first 500
    rows only, so a huge manifest cannot stall the endpoint.
    """
    base = Path(req.base_dir).expanduser()
    if not base.is_dir():
        return RegressionDatasetStatsResponse(
            base_dir=req.base_dir,
            splits={},
            message=tr(SCAN_BASE_MISSING_KEY, path=req.base_dir),
        )

    splits: dict[str, RegressionSplitStats] = {}
    for role, fname in (
        ("train", req.train_csv),
        ("val", req.val_csv),
        ("test", req.test_csv),
    ):
        splits[role] = _regression_split_stats(base / fname, base / req.images_dir, req)

    message = None
    if all(s.missing for s in splits.values()):
        message = tr("scan.no_manifest")

    return RegressionDatasetStatsResponse(
        base_dir=str(base.resolve()),
        splits=splits,
        message=message,
    )


def _regression_split_stats(
    csv_path: Path, images_root: Path, req: RegressionDatasetStatsRequest
) -> RegressionSplitStats:
    """Scan one CSV manifest with the existing row and image sample limits."""
    if not csv_path.is_file():
        return RegressionSplitStats(rows=0, missing=True)
    try:
        with csv_path.open(newline="", encoding="utf-8") as stream:
            return _scan_regression_rows(csv.DictReader(stream), images_root, req)
    except OSError as exc:
        return RegressionSplitStats(rows=0, missing=True, missing_columns=[str(exc)])


def _scan_regression_rows(
    reader: csv.DictReader[str], images_root: Path, req: RegressionDatasetStatsRequest
) -> RegressionSplitStats:
    """Aggregate bounded rows and image checks from an open manifest."""
    rows = 0
    missing_images = 0
    checked_images = 0
    sums: dict[str, tuple[int, float, float, float]] = {}
    header = reader.fieldnames or []
    expected = [req.image_column, *req.target_columns]
    missing_columns = [column for column in expected if column not in header]
    for row in reader:
        rows += 1
        if rows > 50_000:
            rows = 50_000
            break
        if checked_images < 500 and req.image_column in row:
            checked_images += 1
            if not (images_root / str(row[req.image_column])).is_file():
                missing_images += 1
        _accumulate_target_sums(row, req.target_columns, sums)

    return RegressionSplitStats(
        rows=rows,
        missing_columns=missing_columns,
        missing_images=missing_images,
        checked_images=checked_images,
        targets={
            column: RegressionTargetStats(
                count=count, mean=total / count if count else None, min=lo, max=hi
            )
            for column, (count, total, lo, hi) in sums.items()
        },
    )


def _accumulate_target_sums(
    row: dict[str, str | None],
    columns: list[str],
    sums: dict[str, tuple[int, float, float, float]],
) -> None:
    """Accumulate numeric target values in one CSV row."""
    for column in columns:
        if column not in row:
            continue
        cell = row[column]
        if cell is None:
            continue
        try:
            value = float(cell)
        except (TypeError, ValueError):
            continue
        count, total, lo, hi = sums.get(column, (0, 0.0, value, value))
        sums[column] = (
            count + 1,
            total + value,
            min(lo, value),
            max(hi, value),
        )


def _open_native_checkpoint_dialog() -> CheckpointPickResponse:
    """Open a tkinter file dialog filtered to .pth/.pt and return the chosen path.

    Returns an empty path + cancelled=True when the dialog is dismissed or
    tkinter is unavailable.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return CheckpointPickResponse(
            path="",
            cancelled=True,
            message=tr(PICK_NO_TKINTER_KEY),
        )

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(
            title=tr("pick.title_checkpoint"),
            filetypes=[
                ("PyTorch checkpoint", "*.pth *.pt"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
    except Exception as exc:  # noqa: BLE001
        return CheckpointPickResponse(
            path="",
            cancelled=True,
            message=tr(PICK_FAILED_KEY, error=exc),
        )

    if not chosen:
        return CheckpointPickResponse(
            path="", cancelled=True, message=CANCELLED_MESSAGE
        )

    return CheckpointPickResponse(path=str(Path(chosen).resolve()), cancelled=False)


def _open_native_yaml_dialog() -> DatasetPickResponse:
    """Open a tkinter file dialog filtered to .yaml/.yml and return the path.

    Returns an empty path + cancelled=True when the dialog is dismissed or
    tkinter is unavailable.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return DatasetPickResponse(
            path="",
            cancelled=True,
            message=tr(PICK_NO_TKINTER_KEY),
        )

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(
            title=tr("pick.title_yaml"),
            filetypes=[
                ("Ultralytics data.yaml", "*.yaml *.yml"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
    except Exception as exc:  # noqa: BLE001
        return DatasetPickResponse(
            path="",
            cancelled=True,
            message=tr(PICK_FAILED_KEY, error=exc),
        )

    if not chosen:
        return DatasetPickResponse(path="", cancelled=True, message=CANCELLED_MESSAGE)

    return DatasetPickResponse(path=str(Path(chosen).resolve()), cancelled=False)


def _open_native_folder_dialog() -> DatasetPickResponse:
    """Open a tkinter folder dialog and return the chosen absolute path.

    Runs in a worker thread because tkinter blocks the calling thread. Returns
    an empty path + cancelled=True when the dialog is dismissed or unavailable.
    """
    # A container has no display, so the dialog can never open there. Saying so
    # up front beats letting Tk fail and reporting its error, which reads like a
    # bug rather than the expected consequence of running headless.
    if os.environ.get("VISIONFORGE_CONTAINER"):
        return DatasetPickResponse(
            path="",
            cancelled=True,
            message=tr("pick.container"),
        )

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return DatasetPickResponse(
            path="",
            cancelled=True,
            message=tr(PICK_NO_TKINTER_KEY),
        )

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(title=tr("pick.title_folder"))
        root.destroy()
    except Exception as exc:  # noqa: BLE001
        return DatasetPickResponse(
            path="",
            cancelled=True,
            message=tr(PICK_FAILED_KEY, error=exc),
        )

    if not chosen:
        return DatasetPickResponse(path="", cancelled=True, message=CANCELLED_MESSAGE)

    resolved = Path(chosen).resolve()
    _remember_dataset_root(resolved)
    return DatasetPickResponse(path=str(resolved), cancelled=False)


def _normalize_split_name(name: str) -> str:
    """Lowercase + strip simple separators so 'Train_Set' matches 'train'."""
    import unicodedata

    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    stripped = stripped.lower()
    for sep in ("-", "_", " ", "."):
        stripped = stripped.replace(sep, "")
    return stripped


def _match_alias(name: str, aliases: set[str]) -> bool:
    """Return True if ``name`` (or its prefix) maps to one of the alias variants."""
    normalized = _normalize_split_name(name)
    return normalized in aliases


def _detect_dataset_layout(base_dir_str: str) -> DatasetDetectResponse:
    """Inspect a directory and try to identify standard train/val/test splits."""
    if not base_dir_str or not base_dir_str.strip():
        return DatasetDetectResponse(
            base_dir=base_dir_str,
            detected=False,
            message=tr("detect.path_empty"),
        )

    base = Path(base_dir_str).expanduser()
    if not base.exists():
        return DatasetDetectResponse(
            base_dir=str(base),
            detected=False,
            message=tr("detect.not_found", path=base),
        )
    if not base.is_dir():
        return DatasetDetectResponse(
            base_dir=str(base),
            detected=False,
            message=tr("detect.not_a_folder", path=base),
        )

    try:
        children = [p for p in base.iterdir() if p.is_dir()]
    except PermissionError:
        return DatasetDetectResponse(
            base_dir=str(base),
            detected=False,
            message=tr("detect.no_permission", path=base),
        )

    if not children:
        return DatasetDetectResponse(
            base_dir=str(base),
            detected=False,
            message=tr("detect.no_subfolders", path=base),
        )

    train_dir, val_dir, test_dir = _find_split_dirs(children)
    candidates = sorted(c.name for c in children)
    return _layout_response(base, candidates, train_dir, val_dir, test_dir)


def _find_split_dirs(children: list[Path]) -> tuple[str | None, str | None, str | None]:
    """Match each split role to its first recognized child directory."""
    train_dir: str | None = None
    val_dir: str | None = None
    test_dir: str | None = None
    for child in children:
        if train_dir is None and _match_alias(child.name, _TRAIN_ALIASES):
            train_dir = child.name
        elif val_dir is None and _match_alias(child.name, _VAL_ALIASES):
            val_dir = child.name
        elif test_dir is None and _match_alias(child.name, _TEST_ALIASES):
            test_dir = child.name
    return train_dir, val_dir, test_dir


def _layout_response(
    base: Path,
    candidates: list[str],
    train_dir: str | None,
    val_dir: str | None,
    test_dir: str | None,
) -> DatasetDetectResponse:
    """Report a complete, partial, or unrecognized split layout."""
    found = [name for name in (train_dir, val_dir, test_dir) if name]
    if len(found) < 2:
        return DatasetDetectResponse(
            base_dir=str(base),
            detected=False,
            candidates=candidates,
            message=tr("detect.unrecognized", found=", ".join(candidates)),
        )
    missing = _missing_split_roles(train_dir, val_dir, test_dir)
    if missing:
        message = tr("detect.partial", missing=", ".join(missing))
    else:
        message = tr("detect.found", train=train_dir, val=val_dir, test=test_dir)
    return DatasetDetectResponse(
        base_dir=str(base),
        detected=not missing,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        candidates=candidates,
        message=message,
    )


def _missing_split_roles(
    train_dir: str | None, val_dir: str | None, test_dir: str | None
) -> list[str]:
    """Name absent roles in train, validation, test order for the message."""
    missing = []
    if not train_dir:
        missing.append(tr("detect.role_train"))
    if not val_dir:
        missing.append(tr("detect.role_val"))
    if not test_dir:
        missing.append(tr("detect.role_test"))
    return missing
