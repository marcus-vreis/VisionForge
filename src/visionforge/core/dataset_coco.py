"""Export a YOLO detection dataset to COCO, as a new dataset (ADR-169).

Phase 4b of the datasets surface (issue #117). The same rule as incorporating
(ADR-166), merging (ADR-167) and re-splitting (ADR-168) applies: the opened dataset
is never touched, because its fingerprint (ADR-129) is what old ``run.json`` files
remember. The result is a sibling folder ``<name>-coco`` written under a temporary
name and published with one rename (see :mod:`visionforge.core.dataset_derived`).

Two steps, the first one cancellable:

* :func:`analyze_coco_export` reads every label and opens every image (header only)
  to learn its size. An image that does not open, or whose label has a malformed
  line, is *excluded and reported*: a half-converted image would be worse than a
  missing one. An image with no label (or an empty one) is a background image and is
  kept with no annotations.
* :func:`build_coco_export` links the images into ``images/<split>/`` and writes one
  ``annotations/instances_<split>.json`` per split the dataset has.

The conversion itself is :func:`yolo_to_coco_bbox`; its inverse, for the import, is
:func:`coco_bbox_to_yolo`. Both are pure so the convention is testable on its own.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image

from visionforge import __version__
from visionforge.core.cancellation import CancellationToken, is_cancelled
from visionforge.core.dataset_derived import (
    REGISTRY_SCHEMA,
    DerivedError,
    ProgressCounter,
    ensure_unchanged,
    link_or_copy,
    prepare_destination,
    publish,
    write_registry,
)
from visionforge.core.dataset_explorer import SPLIT_NAMES, build_index
from visionforge.core.dataset_fingerprint import fingerprint_dataset

MAX_REPORTED = 200
# COCO category ids start at 1 (0 is conventionally "no object"); YOLO ids start at 0.
CATEGORY_ID_OFFSET = 1
ROUND_DIGITS = 6

# Why an image stays out of the export. The codes are the keys the page translates.
UNREADABLE_IMAGE = "unreadable_image"
MALFORMED_LABEL = "malformed_label"

_BAD_DEST = "coco.bad_dest_name"
_DEST_EXISTS = "coco.dest_exists"
_BASE_CHANGED = "coco.base_changed"
_UTF8 = "utf-8"
_IMAGES = "images"
_ANNOTATIONS = "annotations"

# One YOLO row: class id, then the normalized centre x, centre y, width, height.
YoloBox = tuple[int, float, float, float, float]
# A COCO box: absolute pixels, top-left corner, then width and height.
CocoBox = tuple[float, float, float, float]


class CocoError(DerivedError):
    """A refusal with a code the API turns into a translated message."""


class CocoChangedError(CocoError):
    """The source folder changed between the preview and the write."""


# ── the conversion ────────────────────────────────────────────────────────────


def yolo_to_coco_bbox(
    cx: float, cy: float, w: float, h: float, width: int, height: int
) -> CocoBox:
    """A YOLO box as a COCO ``[x, y, w, h]``.

    Convention: YOLO gives the *centre* and the size, all normalized to ``[0, 1]``
    by the image width (x, w) and height (y, h). COCO gives the *top-left corner*
    and the size, in absolute pixels. So ``x = (cx - w/2) * width``,
    ``y = (cy - h/2) * height``, ``bw = w * width``, ``bh = h * height``. Values are
    rounded to six decimals, which is finer than any pixel grid and keeps the file
    free of float noise such as ``0.30000000000000004``.
    """
    return (
        round((cx - w / 2) * width, ROUND_DIGITS),
        round((cy - h / 2) * height, ROUND_DIGITS),
        round(w * width, ROUND_DIGITS),
        round(h * height, ROUND_DIGITS),
    )


def parse_yolo_label(text: str) -> list[YoloBox]:
    """The boxes of a YOLO label file.

    Blank lines are ignored. A line must have exactly five fields: an integer class
    id (``3`` or ``3.0``), then four finite numbers where the width and height are
    positive.

    Raises:
        ValueError: on the first line that is not a box.
    """
    boxes: list[YoloBox] = []
    for number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"line {number}: expected 5 fields, got {len(parts)}")
        try:
            numbers = [float(p) for p in parts]
        except ValueError as exc:
            raise ValueError(f"line {number}: not a number") from exc
        class_id, cx, cy, w, h = numbers
        if not all(math.isfinite(n) for n in numbers):
            raise ValueError(f"line {number}: not finite")
        if class_id < 0 or class_id != int(class_id):
            raise ValueError(f"line {number}: bad class id")
        if w <= 0 or h <= 0:
            raise ValueError(f"line {number}: width and height must be positive")
        boxes.append((int(class_id), cx, cy, w, h))
    return boxes


# ── reading ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExportImage:
    """One image the export keeps."""

    split: str
    rel: str
    width: int
    height: int
    boxes: tuple[YoloBox, ...]

    @property
    def name(self) -> str:
        return Path(self.rel).name


@dataclass(frozen=True)
class Excluded:
    """An image left out of the export, and why."""

    rel: str
    split: str
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        out = {"path": self.rel, "split": self.split, "reason": self.reason}
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass
class ExportAnalysis:
    """What reading the dataset found; input of :func:`build_coco_export`."""

    root: Path
    fingerprint: dict[str, Any]
    classes: list[str]
    present: dict[str, bool]
    images: list[ExportImage]
    excluded: list[Excluded] = field(default_factory=list)

    def per_split(self) -> dict[str, dict[str, int]]:
        """``{split: {images, boxes}}`` of what the export keeps."""
        out = {s: {"images": 0, "boxes": 0} for s in SPLIT_NAMES}
        for image in self.images:
            out[image.split]["images"] += 1
            out[image.split]["boxes"] += len(image.boxes)
        return out

    def summary(self) -> dict[str, Any]:
        """What the page shows once the dataset was read."""
        reasons = Counter(e.reason for e in self.excluded)
        return {
            "base_dir": str(self.root),
            "classes": self.classes,
            "categories": len(self.classes),
            "present": self.present,
            "images": len(self.images),
            "boxes": sum(len(i.boxes) for i in self.images),
            "per_split": self.per_split(),
            "excluded": len(self.excluded),
            "excluded_by_reason": dict(sorted(reasons.items())),
            "excluded_listed": [e.to_dict() for e in self.excluded[:MAX_REPORTED]],
        }


def _read_image(
    root: Path, rel: str, label_rel: str | None, split: str
) -> tuple[ExportImage | None, Excluded | None]:
    """``(image, None)`` or ``(None, Excluded)`` for one image of the index."""
    try:
        # The dataset is the one the user opened in the explorer; it is only read.
        with Image.open(root / rel) as source:
            width, height = source.size
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        logger.warning("Could not open {}", rel)
        return None, Excluded(rel, split, UNREADABLE_IMAGE, type(exc).__name__)
    boxes: list[YoloBox] = []
    if label_rel is not None:
        try:
            text = (root / label_rel).read_text(encoding=_UTF8)
            boxes = parse_yolo_label(text)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            return None, Excluded(rel, split, MALFORMED_LABEL, str(exc))
    return ExportImage(split, rel, width, height, tuple(boxes)), None


def analyze_coco_export(
    root: Path,
    *,
    token: CancellationToken | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> ExportAnalysis | None:
    """Read the labels and the image sizes of a YOLO dataset. ``None`` if cancelled.

    Raises:
        CocoError: if the folder is missing, has no images, or nothing is left to
            export once the unreadable images and malformed labels are out.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise CocoError("coco.base_missing", path=str(root))
    index = build_index(root, "detection")
    if not index.images:
        raise CocoError("coco.base_empty")
    fingerprint = fingerprint_dataset(root).to_dict()

    progress = ProgressCounter(len(index.images), on_progress)
    images: list[ExportImage] = []
    excluded: list[Excluded] = []
    for entry in index.images:
        if is_cancelled(token):
            return None
        image, left_out = _read_image(root, entry.rel, entry.label_rel, entry.split)
        if image is not None:
            images.append(image)
        if left_out is not None:
            excluded.append(left_out)
        progress.tick()
    if not images:
        raise CocoError("coco.nothing_to_export")
    top = max((b[0] for i in images for b in i.boxes), default=-1)
    names = list(index.class_names)
    names += [f"class_{i}" for i in range(len(names), top + 1)]
    return ExportAnalysis(
        root=root,
        fingerprint=fingerprint,
        classes=names,
        present=dict(index.present),
        images=images,
        excluded=excluded,
    )


def check_export_unchanged(analysis: ExportAnalysis) -> None:
    """Raise if the dataset is no longer what was analysed.

    Raises:
        CocoChangedError: on a different fingerprint.
    """

    def changed() -> CocoChangedError:
        return CocoChangedError(_BASE_CHANGED)

    ensure_unchanged(analysis.root, analysis.fingerprint, error=changed)


# ── writing ───────────────────────────────────────────────────────────────────


def build_coco_export(
    analysis: ExportAnalysis,
    dest_name: str,
    *,
    token: CancellationToken | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    link: Callable[[Path, Path], None] = os.link,
) -> dict[str, Any] | None:
    """Write the COCO dataset next to the original. ``None`` if cancelled.

    Raises:
        CocoError: bad or existing destination name; the dataset changed since the
            analysis.
    """
    root = analysis.root
    dest = prepare_destination(
        root.parent,
        dest_name,
        error=CocoError,
        bad_name_code=_BAD_DEST,
        exists_code=_DEST_EXISTS,
    )
    check_export_unchanged(analysis)

    def write(tmp: Path) -> dict[str, Any] | None:
        return _write(analysis, tmp, token, on_progress, link)

    result: dict[str, Any] | None = publish(
        root.parent, dest, write, error=CocoError, exists_code=_DEST_EXISTS
    )
    if result is None:
        return None
    result["path"] = str(dest)
    return result


def _category_list(classes: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {"id": i + CATEGORY_ID_OFFSET, "name": name, "supercategory": ""}
        for i, name in enumerate(classes)
    ]


def _info(analysis: ExportAnalysis, split: str, now: datetime) -> dict[str, Any]:
    return {
        "description": f"{analysis.root.name} ({split}), exported by VisionForge",
        "contributor": f"VisionForge {__version__}",
        "visionforge_version": __version__,
        "date_created": now.isoformat(timespec="seconds"),
        "year": now.year,
        "source_fingerprint": analysis.fingerprint.get("digest"),
    }


class _Ids:
    """Image and annotation ids: unique across the splits, starting at 1."""

    def __init__(self) -> None:
        self.image = 0
        self.annotation = 0

    def next_image(self) -> int:
        self.image += 1
        return self.image

    def next_annotation(self) -> int:
        self.annotation += 1
        return self.annotation


def _coco_entries(
    image: ExportImage, ids: _Ids
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The ``images`` row and the ``annotations`` rows of one image."""
    image_id = ids.next_image()
    row = {
        "id": image_id,
        "file_name": image.name,
        "width": image.width,
        "height": image.height,
    }
    annotations: list[dict[str, Any]] = []
    for class_id, cx, cy, w, h in image.boxes:
        x, y, bw, bh = yolo_to_coco_bbox(cx, cy, w, h, image.width, image.height)
        annotations.append(
            {
                "id": ids.next_annotation(),
                "image_id": image_id,
                "category_id": class_id + CATEGORY_ID_OFFSET,
                "bbox": [x, y, bw, bh],
                "area": round(bw * bh, ROUND_DIGITS),
                "iscrowd": 0,
                "segmentation": [],
            }
        )
    return row, annotations


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding=_UTF8, newline="\n")


def _write(
    analysis: ExportAnalysis,
    tmp: Path,
    token: CancellationToken | None,
    on_progress: Callable[[int, int], None] | None,
    link: Callable[[Path, Path], None],
) -> dict[str, Any] | None:
    progress = ProgressCounter(len(analysis.images), on_progress)
    methods: Counter[str] = Counter()
    ids = _Ids()
    now = datetime.now(UTC)
    categories = _category_list(analysis.classes)
    (tmp / _ANNOTATIONS).mkdir(parents=True)
    for split in (s for s in SPLIT_NAMES if analysis.present.get(s)):
        folder = tmp / _IMAGES / split
        folder.mkdir(parents=True)
        rows: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        for image in (i for i in analysis.images if i.split == split):
            if is_cancelled(token):
                return None
            methods[
                link_or_copy(analysis.root / image.rel, folder / image.name, link=link)
            ] += 1
            row, boxes = _coco_entries(image, ids)
            rows.append(row)
            annotations.extend(boxes)
            progress.tick()
        _write_json(
            tmp / _ANNOTATIONS / f"instances_{split}.json",
            {
                "info": _info(analysis, split, now),
                "licenses": [],
                "images": rows,
                "annotations": annotations,
                "categories": categories,
            },
        )
    registry = _registry(analysis, methods, now)
    write_registry(tmp, registry)
    if is_cancelled(token):
        return None
    return {
        "registry": registry,
        "images": len(analysis.images),
        "boxes": registry["counts"]["boxes"],
        "excluded": len(analysis.excluded),
        "files_linked": methods.get("link", 0),
        "files_copied": methods.get("copy", 0),
    }


def _registry(
    analysis: ExportAnalysis, methods: Counter[str], now: datetime
) -> dict[str, Any]:
    summary = analysis.summary()
    return {
        "schema_version": REGISTRY_SCHEMA,
        "kind": "coco_export",
        "created_at": now.isoformat(timespec="seconds"),
        "task": "detection",
        "derived_from": {
            "path": str(analysis.root),
            "fingerprint": analysis.fingerprint,
        },
        "classes": analysis.classes,
        "category_id_offset": CATEGORY_ID_OFFSET,
        "counts": {
            "images": summary["images"],
            "boxes": summary["boxes"],
            "per_split": summary["per_split"],
            "excluded": summary["excluded"],
            "excluded_by_reason": summary["excluded_by_reason"],
            "excluded_listed": summary["excluded_listed"],
        },
        "files_linked": methods.get("link", 0),
        "files_copied": methods.get("copy", 0),
    }


def suggest_export_name(root: Path) -> str:
    """``<name>-coco``, or the first ``-N`` that does not exist next to it."""
    root = Path(root).resolve()
    stem = f"{root.name}-coco"
    name, n = stem, 2
    while (root.parent / name).exists():
        name = f"{stem}-{n}"
        n += 1
    return name


def coco_bbox_to_yolo(
    bbox: Sequence[float], width: int, height: int
) -> tuple[tuple[float, float, float, float] | None, bool]:
    """A COCO ``[x, y, w, h]`` as YOLO ``(cx, cy, w, h)``, clipped to the image.

    Convention: the inverse of :func:`yolo_to_coco_bbox`. The box is first clipped
    to ``[0, width] x [0, height]``; a box with no area left is dropped.

    Returns:
        ``(yolo box or None when dropped, whether clipping changed the box)``.
    """
    x, y, bw, bh = bbox
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + bw), min(float(height), y + bh)
    clipped = (x1, y1, x2, y2) != (x, y, x + bw, y + bh)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None, clipped
    return (
        (x1 + x2) / 2 / width,
        (y1 + y2) / 2 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ), clipped


__all__ = [
    "CATEGORY_ID_OFFSET",
    "MALFORMED_LABEL",
    "MAX_REPORTED",
    "UNREADABLE_IMAGE",
    "CocoChangedError",
    "CocoError",
    "Excluded",
    "ExportAnalysis",
    "ExportImage",
    "analyze_coco_export",
    "build_coco_export",
    "check_export_unchanged",
    "coco_bbox_to_yolo",
    "parse_yolo_label",
    "suggest_export_name",
    "yolo_to_coco_bbox",
]
