"""Import a COCO detection dataset as a YOLO dataset, as a new dataset (ADR-169).

The counterpart of :mod:`visionforge.core.dataset_coco`. The folder the user picks is
only read; the result is a sibling folder ``<name>-yolo`` in the standard layout
(``images/<split>`` + ``labels/<split>``, ``classes.txt``, ``data.yaml``) published
with one rename (see :mod:`visionforge.core.dataset_derived`).

Three steps:

* :func:`analyze_coco_import` finds the annotation JSONs, resolves where each one's
  images live, opens every image (header only) and converts every box. It decides
  nothing about splits or class ids, so the plan can change in milliseconds.
* :func:`plan_import` (cheap, pure) assigns each JSON a split (detected from its
  name, or chosen by the user, or skipped) and numbers the classes.
* :func:`build_coco_import` writes the result.

Only the boxes are imported. COCO segmentation polygons are ignored, and the preview
says so.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml
from loguru import logger
from PIL import Image

from visionforge.core.cancellation import CancellationToken, is_cancelled
from visionforge.core.dataset_coco import (
    MAX_REPORTED,
    UNREADABLE_IMAGE,
    CocoChangedError,
    CocoError,
    coco_bbox_to_yolo,
)
from visionforge.core.dataset_derived import (
    CLASSES_TXT,
    REGISTRY_SCHEMA,
    ProgressCounter,
    claim_name,
    ensure_unchanged,
    link_or_copy,
    prepare_destination,
    publish,
    write_registry,
)
from visionforge.core.dataset_explorer import SPLIT_NAMES
from visionforge.core.dataset_fingerprint import fingerprint_dataset

SKIP = "skip"

# Why an image or a box stays out. The codes are the keys the page translates.
MISSING_IMAGE = "missing_image"
UNSAFE_PATH = "unsafe_path"
BAD_IMAGE_ENTRY = "bad_image_entry"
DUPLICATE_IMAGE = "duplicate_image_id"
CROWD = "crowd"
BAD_BBOX = "bad_bbox"
DEGENERATE_BOX = "degenerate_box"
UNKNOWN_IMAGE_ID = "unknown_image_id"
UNKNOWN_CATEGORY = "unknown_category"
BAD_CATEGORY = "bad_category"
# Not left out, only changed: the image stays and the box stays.
SIZE_MISMATCH = "size_mismatch"
CLIPPED_BOX = "clipped_box"

_BAD_DEST = "coco.bad_dest_name"
_DEST_EXISTS = "coco.dest_exists"
_UTF8 = "utf-8"
_SAMPLE = 20
_LABEL_DECIMALS = 6
_SPLIT_TOKEN = re.compile(r"^(train|validation|valid|val|test)\d*$")
_CANONICAL_SPLIT = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}

# A converted box: COCO category id, then YOLO centre x, centre y, width, height.
ImportBox = tuple[int, float, float, float, float]


# ── split detection ───────────────────────────────────────────────────────────


def detect_split(json_name: str) -> tuple[str | None, str | None]:
    """``(split, token)`` named by an annotation file, e.g. ``instances_val2017.json``.

    The token is the word of the file name that names the split as it was written
    (``val2017``), because COCO keeps the images in a folder of that same name. The
    split is ``train``, ``val`` or ``test`` (``valid`` and ``validation`` are
    ``val``); ``(None, None)`` if the name says nothing.
    """
    stem = PurePosixPath(json_name).stem.lower()
    for word in re.split(r"[^a-z0-9]+", stem):
        match = _SPLIT_TOKEN.match(word)
        if match:
            return _CANONICAL_SPLIT[match.group(1)], word
    return None, None


# ── reading ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ImportImage:
    """One image the import keeps, with its boxes already converted."""

    src_rel: str
    file_name: str
    boxes: tuple[ImportBox, ...]


class _Report:
    """What was left out or changed in one JSON, by reason, plus the first few."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.skipped: Counter[str] = Counter()
        self.adjusted: Counter[str] = Counter()
        self.listed: list[dict[str, str]] = []

    def _list(self, reason: str, item: str) -> None:
        if len(self.listed) < MAX_REPORTED:
            self.listed.append({"json": self.source, "item": item, "reason": reason})

    def skip(self, reason: str, item: str) -> None:
        self.skipped[reason] += 1
        self._list(reason, item)

    def adjust(self, reason: str, item: str) -> None:
        self.adjusted[reason] += 1
        self._list(reason, item)


@dataclass
class CocoSource:
    """One annotation JSON, read and converted."""

    name: str
    detected_split: str | None
    images_dir: str | None
    categories: dict[int, str]
    images: list[ImportImage]
    images_listed: int
    annotations: int
    report: _Report

    @property
    def boxes(self) -> int:
        return sum(len(i.boxes) for i in self.images)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "detected_split": self.detected_split,
            "images_dir": self.images_dir,
            "images_listed": self.images_listed,
            "images_kept": len(self.images),
            "annotations": self.annotations,
            "boxes": self.boxes,
            "categories": len(self.categories),
            "skipped_by_reason": dict(sorted(self.report.skipped.items())),
            "adjusted_by_reason": dict(sorted(self.report.adjusted.items())),
            "listed": self.report.listed,
        }


@dataclass
class ImportAnalysis:
    """What reading the folder found; input of :func:`plan_import`."""

    root: Path
    fingerprint: dict[str, Any]
    sources: list[CocoSource]
    # JSON files that are not COCO annotation files: ``{"name", "reason"}``.
    ignored: list[dict[str, str]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "base_dir": str(self.root),
            "sources": [s.summary() for s in self.sources],
            "ignored": self.ignored,
        }


def _json_files(root: Path) -> list[Path]:
    """``*.json`` at the top of the folder and in ``annotations/``, sorted."""
    found: list[Path] = []
    for folder in (root, root / "annotations"):
        if folder.is_dir():
            found.extend(sorted(p for p in folder.glob("*.json") if p.is_file()))
    return found


def _load_coco(path: Path) -> tuple[dict[str, Any] | None, str]:
    """``(data, "")`` for a COCO annotation file, else ``(None, why not)``."""
    try:
        # The folder is the user's own pick in the local GUI; it is only read.
        raw = path.read_bytes()  # NOSONAR pythonsecurity:S6549
        data = json.loads(raw.decode("utf-8-sig"))
    except OSError:
        return None, "unreadable"
    except ValueError:
        return None, "not_json"
    if not isinstance(data, dict):
        return None, "not_coco"
    if not all(
        isinstance(data.get(k), list) for k in ("images", "annotations", "categories")
    ):
        return None, "not_coco"
    return data, ""


def _int_id(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _categories(entries: list[Any], report: _Report) -> dict[int, str]:
    found: dict[int, str] = {}
    for entry in entries:
        cid = _int_id(entry.get("id")) if isinstance(entry, dict) else None
        name = entry.get("name") if isinstance(entry, dict) else None
        if cid is None or not isinstance(name, str) or cid in found:
            report.skip(BAD_CATEGORY, str(entry)[:80])
            continue
        found[cid] = name
    return found


def _safe_relative(file_name: str) -> bool:
    """True for a plain relative path that stays inside the folder it is joined to."""
    posix = PurePosixPath(file_name.replace("\\", "/"))
    if PureWindowsPath(file_name).drive or posix.is_absolute() or not posix.parts:
        return False
    return ".." not in posix.parts


def _split_like_folders(root: Path) -> list[str]:
    """Folders of the root (and of ``images/``) named like a split, e.g. ``val2017``."""
    found: list[str] = []
    for parent in (root / "images", root):
        if not parent.is_dir():
            continue
        for child in sorted(parent.iterdir()):
            if child.is_dir() and _SPLIT_TOKEN.match(child.name.lower()):
                found.append(child.relative_to(root).as_posix())
    return found


def _dir_candidates(
    root: Path, json_name: str, split: str | None, token: str | None
) -> list[str]:
    """Folders that may hold the images, in the order they are tried.

    ``images/<token>``, ``<token>`` (``val2017`` as written, then the canonical
    split), the same with the stem of the file; for a name that says no split, the
    folders named like a split too (the user picks the split afterwards); then
    ``images`` and the folder itself.
    """
    stem = PurePosixPath(json_name).stem
    words = [w for w in re.split(r"[^A-Za-z0-9]+", stem) if w]
    names = [n for n in (token, split, stem, words[-1] if words else None) if n]
    candidates: list[str] = []
    for name in dict.fromkeys(names):
        candidates += [f"images/{name}", name]
    if split is None:
        candidates += _split_like_folders(root)
    return [*candidates, "images", "."]


def _resolve_images_dir(
    root: Path, candidates: list[str], file_names: list[str]
) -> str | None:
    """The first candidate that holds any of the first images; ``None`` if none."""
    sample = [f for f in file_names if _safe_relative(f)][:_SAMPLE]
    for candidate in candidates:
        base = PurePosixPath(candidate)
        # The folder is the user's own pick in the local GUI; it is only read.
        if any(
            (root / (base / f)).is_file() for f in sample
        ):  # NOSONAR pythonsecurity:S6549
            return candidate
    return None


def _valid_bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in value):
        return None
    numbers = [float(v) for v in value]
    if not all(math.isfinite(n) for n in numbers):
        return None
    return numbers[0], numbers[1], numbers[2], numbers[3]


def _one_box(
    ann: dict[str, Any],
    width: int,
    height: int,
    categories: Mapping[int, str],
    report: _Report,
) -> ImportBox | None:
    label = f"annotation {ann.get('id')}"
    if ann.get("iscrowd"):
        report.skip(CROWD, label)
        return None
    cid = _int_id(ann.get("category_id"))
    if cid is None or cid not in categories:
        report.skip(UNKNOWN_CATEGORY, label)
        return None
    bbox = _valid_bbox(ann.get("bbox"))
    if bbox is None:
        report.skip(BAD_BBOX, label)
        return None
    box, clipped = coco_bbox_to_yolo(bbox, width, height)
    if box is None:
        report.skip(DEGENERATE_BOX, label)
        return None
    if clipped:
        report.adjust(CLIPPED_BOX, label)
    return (cid, *box)


@dataclass
class _Entry:
    """An ``images`` row that passed the checks that need no disk."""

    image_id: int
    file_name: str
    declared: tuple[int | None, int | None]


def _image_entries(rows: list[Any], report: _Report) -> list[_Entry]:
    entries: list[_Entry] = []
    seen: set[int] = set()
    for row in rows:
        image_id = _int_id(row.get("id")) if isinstance(row, dict) else None
        file_name = row.get("file_name") if isinstance(row, dict) else None
        if image_id is None or not isinstance(file_name, str) or not file_name:
            report.skip(BAD_IMAGE_ENTRY, str(row)[:80])
            continue
        if image_id in seen:
            report.skip(DUPLICATE_IMAGE, file_name)
            continue
        seen.add(image_id)
        entries.append(
            _Entry(
                image_id,
                file_name,
                (_int_id(row.get("width")), _int_id(row.get("height"))),
            )
        )
    return entries


def _group_annotations(
    rows: list[Any], known: set[int], report: _Report
) -> dict[int, list[dict[str, Any]]]:
    by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in rows:
        image_id = _int_id(ann.get("image_id")) if isinstance(ann, dict) else None
        if image_id is None or image_id not in known:
            report.skip(
                UNKNOWN_IMAGE_ID,
                f"annotation {ann.get('id') if isinstance(ann, dict) else '?'}",
            )
            continue
        by_image.setdefault(image_id, []).append(ann)
    return by_image


def _open_size(path: Path) -> tuple[int, int] | None:
    try:
        # The folder is the user's own pick in the local GUI; it is only read.
        with Image.open(path) as source:  # NOSONAR pythonsecurity:S6549
            width, height = source.size
            return int(width), int(height)
    except (OSError, ValueError, Image.DecompressionBombError):
        logger.warning("Could not open {}", path.name)
        return None


def _convert_image(
    root: Path,
    images_dir: str | None,
    entry: _Entry,
    annotations: list[dict[str, Any]],
    categories: Mapping[int, str],
    report: _Report,
) -> ImportImage | None:
    if not _safe_relative(entry.file_name):
        report.skip(UNSAFE_PATH, entry.file_name)
        return None
    if images_dir is None:
        report.skip(MISSING_IMAGE, entry.file_name)
        return None
    rel = (PurePosixPath(images_dir) / entry.file_name.replace("\\", "/")).as_posix()
    # The folder is the user's own pick in the local GUI; it is only read.
    if not (root / rel).is_file():  # NOSONAR pythonsecurity:S6549
        report.skip(MISSING_IMAGE, entry.file_name)
        return None
    size = _open_size(root / rel)
    if size is None:
        report.skip(UNREADABLE_IMAGE, entry.file_name)
        return None
    if entry.declared != (None, None) and entry.declared != size:
        report.adjust(SIZE_MISMATCH, entry.file_name)
    boxes = [
        box
        for ann in annotations
        if (box := _one_box(ann, size[0], size[1], categories, report)) is not None
    ]
    return ImportImage(rel, entry.file_name, tuple(boxes))


@dataclass
class _Loaded:
    """A COCO file parsed, before any image is opened."""

    name: str
    data: dict[str, Any]
    report: _Report
    entries: list[_Entry]


def _load_sources(
    root: Path, token: CancellationToken | None
) -> tuple[list[_Loaded], list[dict[str, str]]] | None:
    loaded: list[_Loaded] = []
    ignored: list[dict[str, str]] = []
    for path in _json_files(root):
        if is_cancelled(token):
            return None
        name = path.relative_to(root).as_posix()
        data, why = _load_coco(path)
        if data is None:
            ignored.append({"name": name, "reason": why})
            continue
        report = _Report(name)
        loaded.append(
            _Loaded(name, data, report, _image_entries(data["images"], report))
        )
    return loaded, ignored


def _convert_source(
    root: Path,
    item: _Loaded,
    token: CancellationToken | None,
    progress: ProgressCounter,
) -> CocoSource | None:
    split, word = detect_split(item.name)
    categories = _categories(item.data["categories"], item.report)
    by_image = _group_annotations(
        item.data["annotations"], {e.image_id for e in item.entries}, item.report
    )
    images_dir = _resolve_images_dir(
        root,
        _dir_candidates(root, item.name, split, word),
        [e.file_name for e in item.entries],
    )
    images: list[ImportImage] = []
    for entry in item.entries:
        if is_cancelled(token):
            return None
        image = _convert_image(
            root,
            images_dir,
            entry,
            by_image.get(entry.image_id, []),
            categories,
            item.report,
        )
        if image is not None:
            images.append(image)
        progress.tick()
    return CocoSource(
        name=item.name,
        detected_split=split,
        images_dir=images_dir,
        categories=categories,
        images=images,
        images_listed=len(item.entries),
        annotations=len(item.data["annotations"]),
        report=item.report,
    )


def analyze_coco_import(
    root: Path,
    *,
    token: CancellationToken | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> ImportAnalysis | None:
    """Read and convert the COCO annotation files of a folder. ``None`` if cancelled.

    Raises:
        CocoError: if the folder is missing or holds no COCO annotation file.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise CocoError("coco.source_missing", path=str(root))
    fingerprint = fingerprint_dataset(root).to_dict()
    found = _load_sources(root, token)
    if found is None:
        return None
    loaded, ignored = found
    if not loaded:
        raise CocoError("coco.no_annotations")
    progress = ProgressCounter(sum(len(item.entries) for item in loaded), on_progress)
    sources: list[CocoSource] = []
    for item in loaded:
        source = _convert_source(root, item, token, progress)
        if source is None:
            return None
        sources.append(source)
    return ImportAnalysis(root, fingerprint, sources, ignored)


def check_import_unchanged(analysis: ImportAnalysis) -> None:
    """Raise if the folder is no longer what was analysed.

    Raises:
        CocoChangedError: on a different fingerprint.
    """

    def changed() -> CocoChangedError:
        return CocoChangedError("coco.base_changed")

    ensure_unchanged(analysis.root, analysis.fingerprint, error=changed)


# ── the plan ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClassMapping:
    """A COCO category and the YOLO class it becomes."""

    coco_id: int
    yolo_id: int
    name: str
    original_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "coco_id": self.coco_id,
            "yolo_id": self.yolo_id,
            "name": self.name,
            "original_name": self.original_name,
        }


@dataclass
class SourcePlan:
    """What one annotation JSON becomes: a split, nothing yet, or nothing at all."""

    source: CocoSource
    split: str | None
    skipped: bool = False

    @property
    def status(self) -> str:
        if self.skipped:
            return "skipped"
        return "ready" if self.split else "unassigned"


@dataclass
class ImportPlan:
    """Which JSON feeds which split, and the numbering of the classes."""

    entries: list[SourcePlan]
    mapping: list[ClassMapping]

    @property
    def active(self) -> list[SourcePlan]:
        return [e for e in self.entries if e.status == "ready"]

    @property
    def unassigned(self) -> list[str]:
        return [e.source.name for e in self.entries if e.status == "unassigned"]

    @property
    def ready(self) -> bool:
        return bool(self.active) and not self.unassigned

    @property
    def classes(self) -> list[str]:
        return [m.name for m in self.mapping]

    def yolo_ids(self) -> dict[int, int]:
        return {m.coco_id: m.yolo_id for m in self.mapping}

    def require_ready(self) -> None:
        """Raise unless every JSON has a split (or is skipped) and one is used."""
        if self.unassigned:
            raise CocoError("coco.unassigned_split", name=self.unassigned[0])
        if not self.active:
            raise CocoError("coco.nothing_selected")

    def summary(self) -> dict[str, Any]:
        totals = {s: {"images": 0, "boxes": 0} for s in SPLIT_NAMES}
        for entry in self.active:
            totals[str(entry.split)]["images"] += len(entry.source.images)
            totals[str(entry.split)]["boxes"] += entry.source.boxes
        return {
            "ready": self.ready,
            "sources": [
                {
                    "name": e.source.name,
                    "split": e.split,
                    "detected_split": e.source.detected_split,
                    "status": e.status,
                }
                for e in self.entries
            ],
            "unassigned": self.unassigned,
            "per_split": totals,
            "images": sum(t["images"] for t in totals.values()),
            "boxes": sum(t["boxes"] for t in totals.values()),
            "classes": self.classes,
            "mapping": [m.to_dict() for m in self.mapping],
            "renamed": [m.to_dict() for m in self.mapping if m.name != m.original_name],
        }


def _assign(
    analysis: ImportAnalysis, splits: Mapping[str, str] | None
) -> list[SourcePlan]:
    chosen = dict(splits or {})
    names = {s.name for s in analysis.sources}
    for name in chosen:
        if name not in names:
            raise CocoError("coco.unknown_json", name=name)
    entries: list[SourcePlan] = []
    for source in analysis.sources:
        value = chosen.get(source.name, source.detected_split)
        if value is not None and value != SKIP and value not in SPLIT_NAMES:
            raise CocoError("coco.bad_split", split=value)
        entries.append(
            SourcePlan(source, None if value == SKIP else value, value == SKIP)
        )
    return entries


def _check_splits_free(entries: list[SourcePlan]) -> None:
    taken: dict[str, str] = {}
    for entry in entries:
        if entry.split is None:
            continue
        if entry.split in taken:
            raise CocoError(
                "coco.split_taken",
                split=entry.split,
                first=taken[entry.split],
                second=entry.source.name,
            )
        taken[entry.split] = entry.source.name


def _clean_name(name: str, cid: int) -> str:
    """One line with no stray blanks, so ``classes.txt`` keeps every id in place."""
    return " ".join(name.split()) or f"class_{cid}"


def _union_categories(entries: list[SourcePlan]) -> dict[int, str]:
    union: dict[int, str] = {}
    for entry in entries:
        if entry.skipped:
            continue
        for cid, name in entry.source.categories.items():
            cleaned = _clean_name(name, cid)
            if union.setdefault(cid, cleaned) != cleaned:
                raise CocoError(
                    "coco.category_conflict", id=cid, first=union[cid], second=cleaned
                )
    return union


def _numbered(union: Mapping[int, str]) -> list[ClassMapping]:
    """Ids 0..N-1 in ascending COCO id; a repeated name gets ``_<coco id>``."""
    used: set[str] = set()
    mapping: list[ClassMapping] = []
    for yolo_id, cid in enumerate(sorted(union)):
        original = union[cid]
        name = original
        if name in used:
            name = f"{original}_{cid}"
            while name in used:
                name += "_"
        used.add(name)
        mapping.append(ClassMapping(cid, yolo_id, name, original))
    return mapping


def plan_import(
    analysis: ImportAnalysis, splits: Mapping[str, str] | None = None
) -> ImportPlan:
    """Assign splits and number the classes.

    ``splits`` maps a JSON name to ``train``/``val``/``test`` or ``skip``; a JSON
    not in it uses the split its name says, and one that says none stays
    unassigned (the plan is then not ready).

    Raises:
        CocoError: unknown JSON or split; two JSONs on one split; two JSONs that
            give one category id two different names.
    """
    entries = _assign(analysis, splits)
    _check_splits_free(entries)
    return ImportPlan(entries, _numbered(_union_categories(entries)))


# ── writing ───────────────────────────────────────────────────────────────────


def build_coco_import(
    analysis: ImportAnalysis,
    plan: ImportPlan,
    dest_name: str,
    *,
    token: CancellationToken | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    link: Callable[[Path, Path], None] = os.link,
) -> dict[str, Any] | None:
    """Write the YOLO dataset next to the picked folder. ``None`` if cancelled.

    Raises:
        CocoError: the plan is not ready; bad or existing destination name; the
            folder changed since the analysis.
    """
    plan.require_ready()
    root = analysis.root
    dest = prepare_destination(
        root.parent,
        dest_name,
        error=CocoError,
        bad_name_code=_BAD_DEST,
        exists_code=_DEST_EXISTS,
    )
    check_import_unchanged(analysis)

    def write(tmp: Path) -> dict[str, Any] | None:
        return _write(analysis, plan, tmp, dest, token, on_progress, link)

    result: dict[str, Any] | None = publish(
        root.parent, dest, write, error=CocoError, exists_code=_DEST_EXISTS
    )
    if result is None:
        return None
    result["path"] = str(dest)
    return result


def _label_text(boxes: tuple[ImportBox, ...], ids: Mapping[int, int]) -> str:
    d = _LABEL_DECIMALS
    lines = [
        f"{ids[cid]} {cx:.{d}f} {cy:.{d}f} {w:.{d}f} {h:.{d}f}"
        for cid, cx, cy, w, h in boxes
    ]
    return "".join(f"{line}\n" for line in lines)


def _write_split(
    analysis: ImportAnalysis,
    entry: SourcePlan,
    ids: Mapping[int, int],
    tmp: Path,
    methods: Counter[str],
    step: Callable[[], bool],
    link: Callable[[Path, Path], None],
) -> bool:
    """Link one JSON's images and write their labels. ``False`` if cancelled."""
    images_dir = tmp / "images" / str(entry.split)
    labels_dir = tmp / "labels" / str(entry.split)
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    taken: set[str] = set()
    for image in entry.source.images:
        if not step():
            return False
        digest = hashlib.sha256(image.file_name.encode(_UTF8)).hexdigest()
        name = claim_name(taken, Path(image.file_name).name, digest, with_label=True)
        methods[
            link_or_copy(analysis.root / image.src_rel, images_dir / name, link=link)
        ] += 1
        (labels_dir / f"{Path(name).stem}.txt").write_text(
            _label_text(image.boxes, ids), encoding=_UTF8, newline="\n"
        )
    return True


def _write(
    analysis: ImportAnalysis,
    plan: ImportPlan,
    tmp: Path,
    dest: Path,
    token: CancellationToken | None,
    on_progress: Callable[[int, int], None] | None,
    link: Callable[[Path, Path], None],
) -> dict[str, Any] | None:
    progress = ProgressCounter(
        sum(len(e.source.images) for e in plan.active), on_progress
    )
    methods: Counter[str] = Counter()
    ids = plan.yolo_ids()

    def step() -> bool:
        if is_cancelled(token):
            return False
        progress.tick()
        return True

    tmp.mkdir()
    for entry in plan.active:
        if not _write_split(analysis, entry, ids, tmp, methods, step, link):
            return None
    _write_class_files(plan, tmp, dest)
    registry = _registry(analysis, plan, methods)
    write_registry(tmp, registry)
    if is_cancelled(token):
        return None
    return {
        "registry": registry,
        "images": registry["counts"]["images"],
        "boxes": registry["counts"]["boxes"],
        "classes": plan.classes,
        "files_linked": methods.get("link", 0),
        "files_copied": methods.get("copy", 0),
    }


def _write_class_files(plan: ImportPlan, tmp: Path, dest: Path) -> None:
    classes = plan.classes
    (tmp / CLASSES_TXT).write_text(
        "\n".join(classes) + "\n", encoding=_UTF8, newline="\n"
    )
    data: dict[str, Any] = {"path": str(dest)}
    for entry in plan.active:
        data[str(entry.split)] = f"images/{entry.split}"
    data["names"] = dict(enumerate(classes))
    data["nc"] = len(classes)
    (tmp / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding=_UTF8,
        newline="\n",
    )


def _registry(
    analysis: ImportAnalysis, plan: ImportPlan, methods: Counter[str]
) -> dict[str, Any]:
    skipped: Counter[str] = Counter()
    adjusted: Counter[str] = Counter()
    listed: list[dict[str, str]] = []
    for entry in plan.active:
        skipped.update(entry.source.report.skipped)
        adjusted.update(entry.source.report.adjusted)
        listed.extend(entry.source.report.listed)
    summary = plan.summary()
    return {
        "schema_version": REGISTRY_SCHEMA,
        "kind": "coco_import",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "task": "detection",
        "derived_from": {
            "path": str(analysis.root),
            "fingerprint": analysis.fingerprint,
        },
        "classes": plan.classes,
        "mapping": summary["mapping"],
        "renamed": summary["renamed"],
        "sources": [
            {
                "json": e.source.name,
                "split": e.split,
                "images_dir": e.source.images_dir,
                "images_listed": e.source.images_listed,
                "images_kept": len(e.source.images),
                "annotations": e.source.annotations,
                "boxes": e.source.boxes,
            }
            for e in plan.active
        ],
        "skipped_json": [e.source.name for e in plan.entries if e.skipped],
        "ignored_json": analysis.ignored,
        "segmentation": "ignored",
        "counts": {
            "images": summary["images"],
            "boxes": summary["boxes"],
            "per_split": summary["per_split"],
            "skipped_by_reason": dict(sorted(skipped.items())),
            "adjusted_by_reason": dict(sorted(adjusted.items())),
            "listed": listed[:MAX_REPORTED],
        },
        "files_linked": methods.get("link", 0),
        "files_copied": methods.get("copy", 0),
    }


def suggest_import_name(root: Path) -> str:
    """``<name>-yolo``, or the first ``-N`` that does not exist next to it."""
    root = Path(root).resolve()
    stem = f"{root.name}-yolo"
    name, n = stem, 2
    while (root.parent / name).exists():
        name = f"{stem}-{n}"
        n += 1
    return name


__all__ = [
    "BAD_BBOX",
    "BAD_CATEGORY",
    "BAD_IMAGE_ENTRY",
    "CLIPPED_BOX",
    "CROWD",
    "DEGENERATE_BOX",
    "DUPLICATE_IMAGE",
    "MISSING_IMAGE",
    "SIZE_MISMATCH",
    "SKIP",
    "UNKNOWN_CATEGORY",
    "UNKNOWN_IMAGE_ID",
    "UNSAFE_PATH",
    "ClassMapping",
    "CocoSource",
    "ImportAnalysis",
    "ImportImage",
    "ImportPlan",
    "SourcePlan",
    "analyze_coco_import",
    "build_coco_import",
    "check_import_unchanged",
    "detect_split",
    "plan_import",
    "suggest_import_name",
]
