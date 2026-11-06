"""Read-only dataset explorer: index, distributions, thumbnails and integrity scan.

Phase 1 of the datasets surface (ADR-164). Two layouts are understood, the two the
GUI already accepts: an ImageFolder (``<split>/<class>/<image>``) and a YOLO
detection dataset (``images/<split>`` + ``labels/<split>`` or the transposed one,
resolved by :func:`resolve_yolo_split`). Nothing here writes to the dataset.

Two costs are kept apart on purpose:

* :func:`build_index` only lists directories and reads label files. It is cheap
  enough to run on every open, and it feeds the gallery and the class/split counts.
* :func:`scan_integrity` opens and hashes every image. It is the expensive pass, so
  it takes a cancellation token, reports progress and stops at ``max_images``.
"""

from __future__ import annotations

import io
import math
import os
import threading
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw

from visionforge.core.cancellation import CancellationToken, is_cancelled
from visionforge.core.dataset_audit import _verified_image_digest
from visionforge.core.detection_data import resolve_yolo_split

ExplorerTask = Literal["classification", "detection"]

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
SPLIT_NAMES = ("train", "val", "test")

DEFAULT_SCAN_LIMIT = 20_000
MAX_SCAN_LIMIT = 500_000
MAX_REPORTED = 200
MAX_PAGE = 200
THUMB_MIN, THUMB_MAX = 32, 512

# Longest-side buckets and width/height buckets, as (lo, hi) half-open ranges.
_SIDE_EDGES = (0, 64, 128, 256, 512, 1024, 2048, math.inf)
_ASPECT_EDGES = (0.0, 0.5, 0.75, 0.9, 1.1, 1.34, 1.78, 2.5, math.inf)


@dataclass(frozen=True)
class ExplorerImage:
    """One image of the dataset. ``rel`` is POSIX and relative to the dataset root."""

    split: str
    rel: str
    # Classification: the single folder name. Detection: one entry per box, in
    # file order (so a repeated name is a repeated box).
    classes: tuple[str, ...] = ()
    label_rel: str | None = None


@dataclass
class DatasetIndex:
    """Everything the gallery and the counts need, with no image opened."""

    root: Path
    task: ExplorerTask
    class_names: list[str]
    images: list[ExplorerImage] = field(default_factory=list)
    # split -> (is present on disk)
    present: dict[str, bool] = field(default_factory=dict)
    unlabeled: dict[str, int] = field(default_factory=dict)


# ── class names ───────────────────────────────────────────────────────────────


def read_yolo_class_names(base: Path) -> list[str]:
    """Names from ``classes.txt``/``names.txt``, else the ``names`` of ``data.yaml``."""
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
    for candidate in ("data.yaml", "data.yml"):
        path = base / candidate
        if path.is_file():
            names = _names_from_yaml(path)
            if names:
                return names
    return []


def _names_from_yaml(path: Path) -> list[str]:
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, list):
        return [str(n) for n in names]
    if isinstance(names, dict):
        try:
            ordered = sorted(names.items(), key=lambda kv: int(kv[0]))
        except (TypeError, ValueError):
            return []
        return [str(v) for _, v in ordered]
    return []


# ── index ─────────────────────────────────────────────────────────────────────


def build_index(
    root: Path,
    task: ExplorerTask,
    split_dirs: Mapping[str, str] | None = None,
) -> DatasetIndex:
    """List the dataset without opening a single image.

    ``split_dirs`` maps ``train``/``val``/``test`` to the folder name of an
    ImageFolder dataset (defaults to the same names). YOLO layouts ignore it.

    Raises:
        ValueError: if ``root`` is not a directory or ``task`` is unknown.
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {root}")
    if task == "classification":
        return _index_classification(root, split_dirs or {})
    if task == "detection":
        return _index_detection(root)
    raise ValueError(f"Unknown explorer task: {task}")


def _image_files(directory: Path, recursive: bool) -> list[Path]:
    """Image files of a directory, sorted; recursive like torchvision's ImageFolder."""
    found: list[Path] = []
    if recursive:
        for current, dirs, files in os.walk(directory, followlinks=False):
            dirs.sort()
            found.extend(
                Path(current) / n
                for n in sorted(files)
                if Path(n).suffix.lower() in IMAGE_EXTS
            )
        return found
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return []
    return [p for p in entries if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def _index_classification(root: Path, split_dirs: Mapping[str, str]) -> DatasetIndex:
    index = DatasetIndex(root=root, task="classification", class_names=[])
    names: set[str] = set()
    for split in SPLIT_NAMES:
        directory = root / split_dirs.get(split, split)
        index.present[split] = directory.is_dir()
        if not directory.is_dir():
            continue
        for class_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
            names.add(class_dir.name)
            for path in _image_files(class_dir, recursive=True):
                index.images.append(
                    ExplorerImage(
                        split=split,
                        rel=path.relative_to(root).as_posix(),
                        classes=(class_dir.name,),
                    )
                )
    index.class_names = sorted(names)
    return index


def _read_label_ids(label_path: Path) -> list[int]:
    """Class ids of the valid five-field rows (same rule as the trainer's reader)."""
    if not label_path.is_file():
        return []
    ids: list[int] = []
    try:
        text = label_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            ids.append(int(float(parts[0])))
        except ValueError:
            continue
    return ids


def _index_detection(root: Path) -> DatasetIndex:
    declared = read_yolo_class_names(root)
    index = DatasetIndex(root=root, task="detection", class_names=[])
    raw: list[tuple[str, Path, Path, list[int]]] = []
    max_id = -1
    for split in SPLIT_NAMES:
        resolved = resolve_yolo_split(root, split)
        index.present[split] = resolved is not None
        if resolved is None:
            continue
        images_dir, labels_dir = resolved
        index.unlabeled[split] = 0
        for path in _image_files(images_dir, recursive=False):
            label = labels_dir / f"{path.stem}.txt"
            ids = _read_label_ids(label)
            if not ids:
                index.unlabeled[split] += 1
            else:
                max_id = max(max_id, *ids)
            raw.append((split, path, label, ids))
    count = max(len(declared), max_id + 1)
    index.class_names = [
        declared[i] if i < len(declared) else f"class_{i}" for i in range(count)
    ]
    for split, path, label, ids in raw:
        index.images.append(
            ExplorerImage(
                split=split,
                rel=path.relative_to(root).as_posix(),
                classes=tuple(
                    index.class_names[i] if 0 <= i < count else f"class_{i}"
                    for i in ids
                ),
                label_rel=label.relative_to(root).as_posix()
                if label.is_file()
                else None,
            )
        )
    return index


# ── counts and gallery ────────────────────────────────────────────────────────


def summarize(index: DatasetIndex) -> dict[str, Any]:
    """Images per class and split; for detection, boxes per class as well."""
    splits: dict[str, dict[str, Any]] = {}
    for split in SPLIT_NAMES:
        in_split = [im for im in index.images if im.split == split]
        per_class: Counter[str] = Counter()
        boxes: Counter[str] = Counter()
        for im in in_split:
            if index.task == "classification":
                per_class.update(im.classes)
            else:
                per_class.update(set(im.classes))
                boxes.update(im.classes)
        splits[split] = {
            "present": index.present.get(split, False),
            "images": len(in_split),
            "per_class": dict(sorted(per_class.items())),
            "boxes": dict(sorted(boxes.items())),
            "unlabeled": index.unlabeled.get(split, 0),
        }
    return {
        "task": index.task,
        "class_names": index.class_names,
        "total_images": len(index.images),
        "splits": splits,
    }


def select_images(
    index: DatasetIndex,
    *,
    split: str | None = None,
    class_name: str | None = None,
    offset: int = 0,
    limit: int = 60,
) -> tuple[int, list[ExplorerImage]]:
    """One page of the gallery: ``(total matching, the page)``."""
    limit = max(1, min(limit, MAX_PAGE))
    offset = max(0, offset)
    matches = [
        im
        for im in index.images
        if (split is None or im.split == split)
        and (class_name is None or class_name in im.classes)
    ]
    return len(matches), matches[offset : offset + limit]


# ── thumbnails ────────────────────────────────────────────────────────────────


def contained_path(root: Path, rel: str) -> Path:
    """``root / rel`` resolved, refusing anything that leaves ``root``.

    Raises:
        ValueError: on an absolute path, a ``..`` segment or an escaping link.
    """
    if not rel or Path(rel).is_absolute() or ".." in rel.replace("\\", "/").split("/"):
        raise ValueError("path must be relative to the dataset root")
    resolved = (root / rel).resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("path escapes the dataset root")
    return resolved


def render_thumbnail(
    image_path: Path, size: int, label_path: Path | None = None
) -> bytes:
    """A JPEG no larger than ``size`` px on its longest side, boxes drawn if labeled."""
    size = max(THUMB_MIN, min(size, THUMB_MAX))
    with Image.open(image_path) as source:
        source.draft("RGB", (size, size))  # JPEG: decode at reduced scale
        image = source.convert("RGB")
    image.thumbnail((size, size))
    if label_path is not None and label_path.is_file():
        draw = ImageDraw.Draw(image)
        w, h = image.size
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                _, cx, cy, bw, bh = (float(p) for p in parts)
            except ValueError:
                continue
            draw.rectangle(
                (
                    (cx - bw / 2) * w,
                    (cy - bh / 2) * h,
                    (cx + bw / 2) * w,
                    (cy + bh / 2) * h,
                ),
                outline=(255, 64, 64),
                width=2,
            )
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=80)
    return out.getvalue()


# ── integrity scan ────────────────────────────────────────────────────────────


def _bucket(value: float, edges: tuple[float, ...]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return len(edges) - 2


def _edge_label(lo: float, hi: float) -> dict[str, float | None]:
    return {"lo": lo, "hi": None if math.isinf(hi) else hi}


def scan_integrity(
    index: DatasetIndex,
    *,
    max_images: int = DEFAULT_SCAN_LIMIT,
    token: CancellationToken | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Open and hash images: corrupted files, exact duplicates, leakage, sizes.

    At most ``max_images`` are read. Past that limit the scan takes an evenly
    spaced subset (``sampled`` is true), so every split is still represented. A
    duplicate or a leak found in a sample is real; its absence proves nothing, and
    the report says so.

    Cancelling returns what was gathered so far with ``cancelled`` true.
    """
    max_images = max(1, min(max_images, MAX_SCAN_LIMIT))
    total = len(index.images)
    step = max(1, math.ceil(total / max_images)) if total else 1
    chosen = index.images[::step]
    if on_progress:
        on_progress(0, len(chosen))

    corrupted: list[dict[str, str]] = []
    corrupted_count = 0
    by_hash: dict[str, list[ExplorerImage]] = defaultdict(list)
    widths: list[int] = []
    heights: list[int] = []
    sizes: Counter[tuple[int, int]] = Counter()
    side_hist = [0] * (len(_SIDE_EDGES) - 1)
    aspect_hist = [0] * (len(_ASPECT_EDGES) - 1)
    scanned = 0
    cancelled = False
    current: list[ExplorerImage] = []

    def record(path: Path, reason: str) -> None:
        nonlocal corrupted_count
        corrupted_count += 1
        if len(corrupted) < MAX_REPORTED:
            corrupted.append(
                {
                    "split": current[0].split,
                    "path": current[0].rel,
                    "reason": reason,
                }
            )

    root = index.root.resolve()
    for item in chosen:
        if is_cancelled(token):
            cancelled = True
            break
        current[:] = [item]
        digest = _verified_image_digest(root, root / item.rel, record)
        scanned += 1
        if digest is not None:
            by_hash[digest].append(item)
            try:
                with Image.open(root / item.rel) as im:
                    w, h = im.size
            except (OSError, ValueError, Image.DecompressionBombError):
                w = h = 0
            if w > 0 and h > 0:
                widths.append(w)
                heights.append(h)
                sizes[(w, h)] += 1
                side_hist[_bucket(max(w, h), _SIDE_EDGES)] += 1
                aspect_hist[_bucket(w / h, _ASPECT_EDGES)] += 1
        if on_progress:
            on_progress(scanned, len(chosen))

    duplicate_groups, leakage_groups = _group_report(by_hash)
    sampled = step > 1
    return {
        "task": index.task,
        "total_images": total,
        "scanned": scanned,
        "limit": max_images,
        "sampled": sampled,
        "sample_step": step,
        "cancelled": cancelled,
        "complete": not sampled and not cancelled,
        "hash_algorithm": "sha256",
        "corrupted_count": corrupted_count,
        "corrupted": corrupted,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_file_count": sum(len(g["files"]) - 1 for g in duplicate_groups),
        "duplicate_groups": duplicate_groups[:MAX_REPORTED],
        "leakage_group_count": len(leakage_groups),
        "leakage_groups": leakage_groups[:MAX_REPORTED],
        "sizes": _size_report(widths, heights, sizes, side_hist, aspect_hist),
    }


def _group_report(
    by_hash: Mapping[str, list[ExplorerImage]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Groups of identical content: all of them, and those spanning several splits."""
    duplicates: list[dict[str, Any]] = []
    leaks: list[dict[str, Any]] = []
    for digest, items in sorted(by_hash.items()):
        if len(items) < 2:
            continue
        splits = sorted({it.split for it in items})
        classes = sorted({c for it in items for c in set(it.classes)})
        group = {
            "hash": digest,
            "splits": splits,
            "classes": classes,
            "files": [{"split": it.split, "path": it.rel} for it in items],
            # Same bytes labeled as different classes: worse than a plain copy.
            "label_conflict": _label_conflict(items),
        }
        duplicates.append(group)
        if len(splits) > 1:
            leaks.append(group)
    return duplicates, leaks


def _label_conflict(items: list[ExplorerImage]) -> bool:
    """True if identical images disagree on their class set."""
    return len({tuple(sorted(set(it.classes))) for it in items}) > 1


def _size_report(
    widths: list[int],
    heights: list[int],
    sizes: Counter[tuple[int, int]],
    side_hist: list[int],
    aspect_hist: list[int],
) -> dict[str, Any]:
    if not widths:
        return {"count": 0}

    def stats(values: list[int]) -> dict[str, float]:
        ordered = sorted(values)
        mid = len(ordered) // 2
        median = (
            ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        )
        return {
            "min": ordered[0],
            "max": ordered[-1],
            "mean": sum(ordered) / len(ordered),
            "median": median,
        }

    return {
        "count": len(widths),
        "width": stats(widths),
        "height": stats(heights),
        "distinct_sizes": len(sizes),
        "top_sizes": [
            {"width": w, "height": h, "count": n} for (w, h), n in sizes.most_common(8)
        ],
        "side_bins": [
            {**_edge_label(_SIDE_EDGES[i], _SIDE_EDGES[i + 1]), "count": n}
            for i, n in enumerate(side_hist)
        ],
        "aspect_bins": [
            {**_edge_label(_ASPECT_EDGES[i], _ASPECT_EDGES[i + 1]), "count": n}
            for i, n in enumerate(aspect_hist)
        ],
    }


# ── small index cache ─────────────────────────────────────────────────────────


class IndexCache:
    """The last few indexes, so paging the gallery does not re-walk the disk.

    ``put`` is called by the summary (which always rebuilds), ``get`` by the
    gallery. A miss means the caller rebuilds.
    """

    def __init__(self, capacity: int = 4) -> None:
        self._capacity = capacity
        self._items: OrderedDict[tuple[Any, ...], DatasetIndex] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple[Any, ...]) -> DatasetIndex | None:
        with self._lock:
            index = self._items.get(key)
            if index is not None:
                self._items.move_to_end(key)
            return index

    def put(self, key: tuple[Any, ...], index: DatasetIndex) -> None:
        with self._lock:
            self._items[key] = index
            self._items.move_to_end(key)
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)


__all__ = [
    "DEFAULT_SCAN_LIMIT",
    "MAX_SCAN_LIMIT",
    "DatasetIndex",
    "ExplorerImage",
    "ExplorerTask",
    "IndexCache",
    "build_index",
    "contained_path",
    "read_yolo_class_names",
    "render_thumbnail",
    "scan_integrity",
    "select_images",
    "summarize",
]
