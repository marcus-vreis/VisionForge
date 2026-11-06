"""Incorporate new images into a dataset by writing a new dataset (ADR-166).

Phase 2 of the datasets surface (issue #117). The original folder is never
touched: its fingerprint (ADR-129) is what old ``run.json`` files remember, so an
edit in place would silently change what those runs trained on. The result is a
sibling folder with the same structure, written next to the original under a
temporary name and published with a single rename, so nothing is ever half done.

Two steps, both cancellable and with progress:

* :func:`analyze_incorporation` reads the new folder, checks its format, opens
  every image, hashes it (SHA-256, the same digest the audit and the explorer
  use) and compares it with the existing dataset and with the other new images.
  It decides nothing about splits.
* :func:`plan_incorporation` (cheap, pure) distributes what survived among the
  splits, stratified by class and seeded; :func:`build_incorporated_dataset`
  writes the result.

Images already in the dataset stay in the split they are in. A split the dataset
does not have is never created.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import random
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from visionforge.core.cancellation import CancellationToken, is_cancelled
from visionforge.core.dataset_audit import _verified_image_digest
from visionforge.core.dataset_explorer import (
    IMAGE_EXTS,
    SPLIT_NAMES,
    ExplorerTask,
    build_index,
    read_yolo_class_names,
)
from visionforge.core.dataset_fingerprint import fingerprint_dataset
from visionforge.core.detection_data import resolve_yolo_split
from visionforge.utils.paths import validate_path_component

REGISTRY_NAME = "visionforge_dataset.json"
REGISTRY_SCHEMA = 1
MAX_REPORTED = 200
DEFAULT_SEED = 0

# Why an image stays out. The codes are the keys the page translates.
UNREADABLE = "unreadable"
DUPLICATE_EXISTING = "duplicate_existing"
DUPLICATE_NEW = "duplicate_new"
NO_LABEL = "no_label"
BAD_LABEL = "bad_label"
NO_CLASS_FOLDER = "no_class_folder"
UNKNOWN_CLASS = "unknown_class"

_CHUNK = 1024 * 1024
_CLASSES_TXT = "classes.txt"
_CLASS_FILES = (_CLASSES_TXT, "names.txt", "data.yaml", "data.yml")


class IncorporateError(Exception):
    """A refusal with a code the API turns into a translated message."""

    def __init__(self, code: str, **params: object) -> None:
        super().__init__(code)
        self.code = code
        self.params = params


class DatasetChangedError(IncorporateError):
    """The base dataset or a new image changed between the preview and the write."""


@dataclass(frozen=True)
class Excluded:
    """An image left out, and why."""

    path: str
    reason: str
    detail: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {"path": self.path, "reason": self.reason, "detail": self.detail}


@dataclass(frozen=True)
class Candidate:
    """One new image that passed every check, ready to be placed in a split."""

    rel: str
    path: Path
    sha256: str
    size: int
    mtime_ns: int
    # Classification: ``(folder,)``. Detection: the class name of every box.
    names: tuple[str, ...]
    label_path: Path | None = None
    # Detection only: ``(id in the new folder, "cx cy w h")`` per row.
    rows: tuple[tuple[int, str], ...] = ()

    @property
    def stratum(self) -> str:
        """The class this image stands for when splits are stratified.

        The most frequent class of the image (ties by name), or ``""`` for a
        detection image without boxes.
        """
        if not self.names:
            return ""
        counts = Counter(self.names)
        return min(counts, key=lambda n: (-counts[n], n))


@dataclass(frozen=True)
class NewClass:
    name: str
    images: int
    # Id the class has in the folder of new images, when it has one.
    source_id: int | None = None


@dataclass
class Analysis:
    """What reading the new folder found; input of :func:`plan_incorporation`."""

    task: ExplorerTask
    base_root: Path
    source_root: Path
    split_dirs: dict[str, str]
    base_fingerprint: dict[str, Any]
    base_classes: list[str]
    present: dict[str, bool]
    base_counts: dict[str, int]
    source_has_names: bool
    candidates: list[Candidate] = field(default_factory=list)
    excluded: list[Excluded] = field(default_factory=list)
    new_classes: list[NewClass] = field(default_factory=list)
    source_images: int = 0


# ── the new folder ────────────────────────────────────────────────────────────


def _list_source_images(root: Path) -> list[Path]:
    found: list[Path] = []
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        found.extend(
            Path(current) / n
            for n in sorted(files)
            if Path(n).suffix.lower() in IMAGE_EXTS
        )
    return found


def _label_for(image: Path) -> Path | None:
    """The YOLO label of an image: next to it, or in the sibling ``labels`` tree."""
    beside = image.with_suffix(".txt")
    if beside.is_file():
        return beside
    parts = list(image.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            swapped = Path(
                *parts[:i], "labels", *parts[i + 1 : -1], f"{image.stem}.txt"
            )
            return swapped if swapped.is_file() else None
    return None


def _parse_label(path: Path) -> tuple[tuple[int, str], ...]:
    """The rows of a YOLO label file.

    Raises:
        ValueError: with the line number, if a row is not ``id cx cy w h``.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(type(exc).__name__) from exc
    rows: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"line {number}")
        try:
            raw_id = float(parts[0])
            coords = [float(p) for p in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"line {number}") from exc
        if (
            not raw_id.is_integer()
            or raw_id < 0
            or not all(math.isfinite(c) for c in coords)
        ):
            raise ValueError(f"line {number}")
        rows.append((int(raw_id), " ".join(parts[1:])))
    return tuple(rows)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def analyze_incorporation(
    base_root: Path,
    task: ExplorerTask,
    source_root: Path,
    split_dirs: Mapping[str, str] | None = None,
    *,
    token: CancellationToken | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> Analysis | None:
    """Read the new folder and compare it with the dataset. ``None`` if cancelled.

    Raises:
        IncorporateError: if either folder is unusable, or one contains the other.
    """
    base_root = Path(base_root).resolve()
    source_root = Path(source_root).resolve()
    if not base_root.is_dir():
        raise IncorporateError("incorporate.base_missing", path=str(base_root))
    if not source_root.is_dir():
        raise IncorporateError("incorporate.source_missing", path=str(source_root))
    if source_root.is_relative_to(base_root) or base_root.is_relative_to(source_root):
        raise IncorporateError("incorporate.source_overlaps_base")

    dirs = {s: (split_dirs or {}).get(s, s) for s in SPLIT_NAMES}
    index = build_index(base_root, task, dirs)
    if not index.images:
        raise IncorporateError("incorporate.base_empty")
    fingerprint = fingerprint_dataset(base_root).to_dict()

    sources = _list_source_images(source_root)
    total = len(index.images) + len(sources)
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        if on_progress and (done % 25 == 0 or done == total):
            on_progress(done, total)

    if on_progress:
        on_progress(0, total)

    existing: dict[str, str] = {}
    for image in index.images:
        if is_cancelled(token):
            return None
        try:
            existing.setdefault(_file_sha256(base_root / image.rel), image.rel)
        except OSError:
            logger.warning("Could not hash {} while comparing", image.rel)
        tick()

    analysis = Analysis(
        task=task,
        base_root=base_root,
        source_root=source_root,
        split_dirs=dirs,
        base_fingerprint=fingerprint,
        base_classes=list(index.class_names),
        present=dict(index.present),
        base_counts={
            s: sum(1 for i in index.images if i.split == s) for s in SPLIT_NAMES
        },
        source_has_names=False,
        source_images=len(sources),
    )
    source_names: list[str] = []
    if task == "detection":
        source_names = read_yolo_class_names(source_root)
        analysis.source_has_names = bool(source_names)

    seen_new: dict[str, str] = {}
    for path in sources:
        if is_cancelled(token):
            return None
        _consider(analysis, path, source_names, existing, seen_new)
        tick()

    _collect_new_classes(analysis)
    return analysis


def _consider(
    analysis: Analysis,
    path: Path,
    source_names: list[str],
    existing: Mapping[str, str],
    seen_new: dict[str, str],
) -> None:
    """Check one new image and either keep it as a candidate or list it as excluded."""
    root = analysis.source_root
    rel = path.relative_to(root).as_posix()

    def exclude(reason: str, detail: str | None = None) -> None:
        analysis.excluded.append(Excluded(rel, reason, detail))

    names: tuple[str, ...]
    rows: tuple[tuple[int, str], ...] = ()
    label: Path | None = None
    if analysis.task == "classification":
        parts = path.relative_to(root).parts
        if len(parts) < 2:
            exclude(NO_CLASS_FOLDER)
            return
        names = (parts[0],)
    else:
        label = _label_for(path)
        if label is None:
            exclude(NO_LABEL)
            return
        try:
            rows = _parse_label(label)
            names = tuple(
                _name_of_id(i, analysis.base_classes, source_names) for i, _ in rows
            )
        except ValueError as exc:
            exclude(BAD_LABEL, str(exc))
            return

    issues: list[str] = []
    digest = _verified_image_digest(root, path, lambda _p, why: issues.append(why))
    if digest is None:
        exclude(UNREADABLE, issues[0] if issues else None)
        return
    if digest in existing:
        exclude(DUPLICATE_EXISTING, existing[digest])
        return
    if digest in seen_new:
        exclude(DUPLICATE_NEW, seen_new[digest])
        return
    seen_new[digest] = rel
    stat = path.stat()
    analysis.candidates.append(
        Candidate(
            rel=rel,
            path=path,
            sha256=digest,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            names=names,
            label_path=label,
            rows=rows,
        )
    )


def _name_of_id(class_id: int, base: list[str], source_names: list[str]) -> str:
    """The class a label id stands for.

    A folder of new images with its own class list is read through it (so its
    ``0`` need not be the dataset's ``0``). Without one the ids are taken as the
    dataset's own, and an id beyond them names a new class ``class_<id>``.
    """
    if source_names:
        if class_id >= len(source_names):
            raise ValueError(f"id {class_id} outside the class list")
        return source_names[class_id]
    return base[class_id] if class_id < len(base) else f"class_{class_id}"


def _collect_new_classes(analysis: Analysis) -> None:
    known = set(analysis.base_classes)
    counts: Counter[str] = Counter()
    ids: dict[str, int] = {}
    for cand in analysis.candidates:
        for name in set(cand.names):
            if name not in known:
                counts[name] += 1
        # Detection: the smallest id a new class has in the new folder, so the
        # order of the appended classes is stable.
        for (source_id, _), name in zip(cand.rows, cand.names, strict=False):
            if name not in known:
                ids[name] = min(source_id, ids.get(name, source_id))
    analysis.new_classes = [
        NewClass(name, counts[name], ids.get(name))
        for name in sorted(counts, key=lambda n: (ids.get(n, 0), n))
    ]


# ── the plan ──────────────────────────────────────────────────────────────────


@dataclass
class Plan:
    """Which new image goes to which split, with the numbers the preview shows."""

    task: ExplorerTask
    seed: int
    ratios: dict[str, float]
    accepted_classes: list[str]
    classes: list[str]
    assignments: list[tuple[Candidate, str]]
    excluded: list[Excluded]

    def counts(self) -> dict[str, dict[str, int]]:
        """``{split: {class: images}}`` of what will be added (stratum counts)."""
        table: dict[str, Counter[str]] = {s: Counter() for s in SPLIT_NAMES}
        for cand, split in self.assignments:
            table[split][cand.stratum or "(background)"] += 1
        return {s: dict(sorted(table[s].items())) for s in SPLIT_NAMES}

    def summary(self) -> dict[str, Any]:
        reasons = Counter(e.reason for e in self.excluded)
        return {
            "seed": self.seed,
            "ratios": self.ratios,
            "accepted_classes": self.accepted_classes,
            "classes": self.classes,
            "added": len(self.assignments),
            "added_per_split": {
                s: sum(1 for _, sp in self.assignments if sp == s) for s in SPLIT_NAMES
            },
            "added_per_class": self.counts(),
            "excluded_count": len(self.excluded),
            "excluded_by_reason": dict(sorted(reasons.items())),
            "excluded": [e.to_dict() for e in self.excluded[:MAX_REPORTED]],
        }


def default_ratios(analysis: Analysis) -> dict[str, float]:
    """The dataset's own proportion: images per split over the total."""
    total = sum(analysis.base_counts.values())
    if total == 0:
        raise IncorporateError("incorporate.base_empty")
    return {s: analysis.base_counts[s] / total for s in SPLIT_NAMES}


def resolve_ratios(
    analysis: Analysis, requested: Mapping[str, float] | None
) -> dict[str, float]:
    """Normalised ratios for the splits the dataset has.

    Raises:
        IncorporateError: on a negative or all-zero ratio, an unknown split, or a
            positive ratio for a split the dataset does not have.
    """
    if requested is None:
        raw = default_ratios(analysis)
    else:
        raw = dict.fromkeys(SPLIT_NAMES, 0.0)
        for name, value in requested.items():
            if name not in raw or not math.isfinite(value) or value < 0:
                raise IncorporateError("incorporate.bad_ratios")
            raw[name] = float(value)
        for name, value in raw.items():
            if value > 0 and not analysis.present.get(name, False):
                raise IncorporateError("incorporate.ratio_split_absent", split=name)
    total = sum(raw.values())
    if total <= 0:
        raise IncorporateError("incorporate.bad_ratios")
    return {s: raw[s] / total for s in SPLIT_NAMES}


def _allocate(
    n: int, ratios: Mapping[str, float], rng: random.Random
) -> dict[str, int]:
    """Split ``n`` items by ``ratios`` with the largest-remainder method."""
    quotas = {s: n * ratios[s] for s in SPLIT_NAMES}
    counts = {s: math.floor(q) for s, q in quotas.items()}
    leftover = n - sum(counts.values())
    order = sorted(
        (s for s in SPLIT_NAMES if ratios[s] > 0),
        key=lambda s: (-(quotas[s] - counts[s]), rng.random()),
    )
    for s in order[:leftover]:
        counts[s] += 1
    return counts


def plan_incorporation(
    analysis: Analysis,
    *,
    accept_classes: Sequence[str] = (),
    ratios: Mapping[str, float] | None = None,
    seed: int = DEFAULT_SEED,
) -> Plan:
    """Distribute the surviving new images among the splits.

    Deterministic: the same analysis, classes, ratios and seed give the same plan,
    whatever the order the files came in. Each class is split on its own, so the
    proportion holds per class and not only overall.

    Raises:
        IncorporateError: on a class that is not new, or bad ratios.
    """
    new_names = {c.name for c in analysis.new_classes}
    unknown = [c for c in accept_classes if c not in new_names]
    if unknown:
        raise IncorporateError("incorporate.not_a_new_class", name=unknown[0])
    accepted = sorted(set(accept_classes))
    effective = resolve_ratios(analysis, ratios)
    allowed = set(analysis.base_classes) | set(accepted)

    excluded = list(analysis.excluded)
    kept: list[Candidate] = []
    for cand in analysis.candidates:
        missing = sorted({n for n in cand.names if n not in allowed})
        if missing:
            excluded.append(Excluded(cand.rel, UNKNOWN_CLASS, ", ".join(missing)))
        else:
            kept.append(cand)

    by_stratum: dict[str, list[Candidate]] = defaultdict(list)
    for cand in kept:
        by_stratum[cand.stratum].append(cand)

    assignments: list[tuple[Candidate, str]] = []
    for stratum in sorted(by_stratum):
        group = sorted(by_stratum[stratum], key=lambda c: (c.sha256, c.rel))
        # A reproducible split, not a secret: the seed is recorded on purpose.
        rng = random.Random(f"{seed}|{stratum}")  # NOSONAR python:S2245
        rng.shuffle(group)
        quota = _allocate(len(group), effective, rng)
        start = 0
        for split in SPLIT_NAMES:
            for cand in group[start : start + quota[split]]:
                assignments.append((cand, split))
            start += quota[split]
    assignments.sort(key=lambda pair: pair[0].rel)
    return Plan(
        task=analysis.task,
        seed=seed,
        ratios=effective,
        accepted_classes=accepted,
        classes=final_classes(analysis, accepted),
        assignments=assignments,
        excluded=excluded,
    )


def final_classes(analysis: Analysis, accepted: Sequence[str]) -> list[str]:
    """The class list of the result: the dataset's, then the accepted new ones.

    Existing ids are never renumbered. A new class that came with its own class
    list gets the next free id. One that came as a bare id keeps that id, and a
    gap before it is filled with ``class_<id>`` placeholders.
    """
    classes = list(analysis.base_classes)
    if analysis.task != "detection" or not accepted:
        return classes if analysis.task == "detection" else classes + list(accepted)
    info = {c.name: c for c in analysis.new_classes}
    if analysis.source_has_names:
        ordered = sorted(accepted, key=lambda n: (info[n].source_id or 0, n))
        return classes + ordered
    top = max(info[n].source_id or 0 for n in accepted)
    return classes + [f"class_{i}" for i in range(len(classes), top + 1)]


# ── writing ───────────────────────────────────────────────────────────────────


def link_or_copy(
    src: Path,
    dst: Path,
    *,
    link: Callable[[Path, Path], None] = os.link,
) -> str:
    """Hardlink ``src`` to ``dst``, or copy it when linking is not possible.

    A hardlink needs the same volume and a file system that supports it; any
    ``OSError`` falls back to a copy. Symbolic links are copied (their target's
    bytes), never linked, so the result does not depend on the original tree.

    Returns:
        ``"link"`` or ``"copy"``.
    """
    if not src.is_symlink():
        try:
            link(src, dst)
            return "link"
        except OSError as exc:
            if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK, errno.ENOTSUP):
                logger.debug("hardlink failed ({}), copying {}", exc, src.name)
    shutil.copy2(src, dst)
    return "copy"


def _copy_verified(
    src: Path, dst: Path, expected: str, size: int, mtime_ns: int
) -> None:
    """Copy a new file while hashing it; refuse if it is not what the preview saw."""
    stat = src.stat()
    if stat.st_size != size or stat.st_mtime_ns != mtime_ns:
        raise DatasetChangedError("incorporate.source_changed", path=src.name)
    digest = hashlib.sha256()
    with src.open("rb") as reader, dst.open("wb") as writer:
        while chunk := reader.read(_CHUNK):
            digest.update(chunk)
            writer.write(chunk)
    if digest.hexdigest() != expected:
        raise DatasetChangedError("incorporate.source_changed", path=src.name)
    shutil.copystat(src, dst)


def _free_name(taken: set[str], name: str, sha: str, *, with_label: bool) -> str:
    """``name`` if unused (case-insensitively), else ``stem-<sha8>`` with a counter.

    ``with_label``: the stem's ``.txt`` must be free too (YOLO pairs by stem).
    """

    def free(candidate: str) -> bool:
        if candidate.lower() in taken:
            return False
        return not (with_label and f"{Path(candidate).stem.lower()}.txt" in taken)

    if free(name):
        return name
    stem, ext = os.path.splitext(name)
    candidate = f"{stem}-{sha[:8]}{ext}"
    counter = 2
    while not free(candidate):
        candidate = f"{stem}-{sha[:8]}-{counter}{ext}"
        counter += 1
    return candidate


def _remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def build_incorporated_dataset(
    analysis: Analysis,
    plan: Plan,
    dest_name: str,
    *,
    token: CancellationToken | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    link: Callable[[Path, Path], None] = os.link,
) -> dict[str, Any] | None:
    """Write the new dataset next to the original. ``None`` if cancelled.

    The folder is built under a temporary name in the same parent and renamed to
    ``dest_name`` at the very end; on cancellation or any error it is removed and
    nothing is left at the destination.

    Raises:
        IncorporateError: bad or existing destination name; the original or a new
            image changed since the analysis.
    """
    try:
        validate_path_component(dest_name)
    except ValueError as exc:
        raise IncorporateError("incorporate.bad_dest_name") from exc
    base = analysis.base_root
    dest = base.parent / dest_name
    if dest.exists():
        raise IncorporateError("incorporate.dest_exists", name=dest_name)
    now = fingerprint_dataset(base).to_dict()
    if now.get("digest") != analysis.base_fingerprint.get("digest"):
        raise DatasetChangedError("incorporate.base_changed")

    tmp = base.parent / f".{dest_name}.tmp-{uuid.uuid4().hex[:8]}"
    try:
        result = _write(analysis, plan, tmp, dest, token, on_progress, link)
    except BaseException:
        _remove_tree(tmp)
        raise
    if result is None:
        _remove_tree(tmp)
        return None
    try:
        # Re-checked: a folder created meanwhile must not be replaced or merged.
        if dest.exists():
            raise IncorporateError("incorporate.dest_exists", name=dest_name)
        os.rename(tmp, dest)
    except BaseException:
        _remove_tree(tmp)
        raise
    result["path"] = str(dest)
    return result


def _write(
    analysis: Analysis,
    plan: Plan,
    tmp: Path,
    dest: Path,
    token: CancellationToken | None,
    on_progress: Callable[[int, int], None] | None,
    link: Callable[[Path, Path], None],
) -> dict[str, Any] | None:
    base = analysis.base_root
    rewritten = _rewritten_files(analysis, plan)
    files = _base_files(base)
    total = len(files) + len(plan.assignments)
    done = 0
    methods: Counter[str] = Counter()

    def tick() -> None:
        nonlocal done
        done += 1
        if on_progress and (done % 25 == 0 or done == total):
            on_progress(done, total)

    if on_progress:
        on_progress(0, total)
    tmp.mkdir()
    for rel in files:
        if is_cancelled(token):
            return None
        if rel in rewritten or rel == REGISTRY_NAME:
            tick()
            continue
        target = tmp / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        methods[link_or_copy(base / rel, target, link=link)] += 1
        tick()

    final_ids = {name: i for i, name in enumerate(plan.classes)}
    layout = _layout(analysis)
    taken: dict[Path, set[str]] = {}
    placed = 0
    for cand, split in plan.assignments:
        if is_cancelled(token):
            return None
        images_dir, labels_dir = layout[split]
        folder = (
            images_dir / cand.names[0]
            if analysis.task == "classification"
            else images_dir
        )
        (tmp / folder).mkdir(parents=True, exist_ok=True)
        if folder not in taken:
            taken[folder] = _occupied(tmp, folder, labels_dir, analysis.task)
        detection = analysis.task == "detection"
        name = _free_name(
            taken[folder], Path(cand.rel).name, cand.sha256, with_label=detection
        )
        taken[folder].add(name.lower())
        if detection:
            taken[folder].add(f"{Path(name).stem.lower()}.txt")
        _copy_verified(
            cand.path, tmp / folder / name, cand.sha256, cand.size, cand.mtime_ns
        )
        if analysis.task == "detection":
            assert labels_dir is not None and cand.label_path is not None
            (tmp / labels_dir).mkdir(parents=True, exist_ok=True)
            _write_label(
                cand,
                final_ids,
                tmp / labels_dir / f"{Path(name).stem}.txt",
            )
        placed += 1
        tick()

    _write_class_files(analysis, plan, tmp, dest)
    registry = _registry(analysis, plan, methods, placed)
    (tmp / REGISTRY_NAME).write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if is_cancelled(token):
        return None
    return {"registry": registry, "added": placed, "link_methods": dict(methods)}


def _base_files(base: Path) -> list[str]:
    found: list[str] = []
    for current, dirs, names in os.walk(base, followlinks=False):
        dirs.sort()
        found.extend(
            (Path(current) / n).relative_to(base).as_posix() for n in sorted(names)
        )
    return found


def _layout(analysis: Analysis) -> dict[str, tuple[Path, Path | None]]:
    """Where new images of each split go, relative to the dataset root."""
    layout: dict[str, tuple[Path, Path | None]] = {}
    for split in SPLIT_NAMES:
        if not analysis.present.get(split):
            continue
        if analysis.task == "classification":
            layout[split] = (Path(analysis.split_dirs[split]), None)
            continue
        resolved = resolve_yolo_split(analysis.base_root, split)
        assert resolved is not None
        images_dir, labels_dir = resolved
        layout[split] = (
            images_dir.relative_to(analysis.base_root),
            labels_dir.relative_to(analysis.base_root),
        )
    return layout


def _occupied(tmp: Path, folder: Path, labels_dir: Path | None, task: str) -> set[str]:
    names = {p.name.lower() for p in (tmp / folder).iterdir()}
    if task == "detection" and labels_dir is not None and (tmp / labels_dir).is_dir():
        names |= {p.name.lower() for p in (tmp / labels_dir).iterdir()}
    return names


def _write_label(
    cand: Candidate,
    final_ids: Mapping[str, int],
    target: Path,
) -> None:
    """Copy the label byte for byte when no id moves, else write it with new ids."""
    assert cand.label_path is not None
    mapped = [final_ids[name] for name in cand.names]
    if mapped == [i for i, _ in cand.rows]:
        shutil.copyfile(cand.label_path, target)
        return
    lines = [f"{cid} {rest}" for cid, (_, rest) in zip(mapped, cand.rows, strict=True)]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _rewritten_files(analysis: Analysis, plan: Plan) -> set[str]:
    """Base files that the result writes itself instead of linking.

    A linked file shares its bytes with the original, so a file that changes must
    be written anew, never through the link.
    """
    if analysis.task != "detection" or len(plan.classes) == len(analysis.base_classes):
        return set()
    present = {n for n in _CLASS_FILES if (analysis.base_root / n).is_file()}
    return present


def _write_class_files(analysis: Analysis, plan: Plan, tmp: Path, dest: Path) -> None:
    if analysis.task != "detection" or len(plan.classes) == len(analysis.base_classes):
        return
    base = analysis.base_root
    wrote = False
    for name in (_CLASSES_TXT, "names.txt"):
        if (base / name).is_file():
            (tmp / name).write_text("\n".join(plan.classes) + "\n", encoding="utf-8")
            wrote = True
    for name in ("data.yaml", "data.yml"):
        if (base / name).is_file():
            _write_yaml(base / name, tmp / name, plan.classes, base, dest)
            wrote = True
    if not wrote:
        (tmp / _CLASSES_TXT).write_text(
            "\n".join(plan.classes) + "\n", encoding="utf-8"
        )


def _write_yaml(
    src: Path, out: Path, classes: list[str], base: Path, dest: Path
) -> None:
    import yaml

    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        data = {}
    if isinstance(data.get("names"), dict):
        data["names"] = dict(enumerate(classes))
    else:
        data["names"] = list(classes)
    if "nc" in data:
        data["nc"] = len(classes)
    path_value = data.get("path")
    if isinstance(path_value, str) and Path(path_value).is_absolute():
        try:
            if Path(path_value).resolve() == base:
                data["path"] = str(dest)
        except OSError:
            pass
    out.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _registry(
    analysis: Analysis, plan: Plan, methods: Counter[str], placed: int
) -> dict[str, Any]:
    reasons = Counter(e.reason for e in plan.excluded)
    return {
        "schema_version": REGISTRY_SCHEMA,
        "kind": "incorporated",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "task": analysis.task,
        "derived_from": {
            "path": str(analysis.base_root),
            "fingerprint": analysis.base_fingerprint,
        },
        "incorporated_from": str(analysis.source_root),
        "seed": plan.seed,
        "ratios": plan.ratios,
        "split_dirs": analysis.split_dirs,
        "classes": plan.classes,
        "accepted_new_classes": plan.accepted_classes,
        "counts": {
            "base": analysis.base_counts,
            "found_in_source": analysis.source_images,
            "added": placed,
            "added_per_split": {
                s: sum(1 for _, sp in plan.assignments if sp == s) for s in SPLIT_NAMES
            },
            "added_per_class": plan.counts(),
            "excluded_by_reason": dict(sorted(reasons.items())),
        },
        "files_linked": methods.get("link", 0),
        "files_copied": methods.get("copy", 0),
    }


def suggest_name(base_root: Path) -> str:
    """``<name>-v2``, or the first ``-vN`` that does not exist next to the dataset."""
    base_root = Path(base_root).resolve()
    stem = base_root.name
    n = 2
    while (base_root.parent / f"{stem}-v{n}").exists():
        n += 1
    return f"{stem}-v{n}"


__all__ = [
    "BAD_LABEL",
    "DEFAULT_SEED",
    "DUPLICATE_EXISTING",
    "DUPLICATE_NEW",
    "NO_CLASS_FOLDER",
    "NO_LABEL",
    "REGISTRY_NAME",
    "UNKNOWN_CLASS",
    "UNREADABLE",
    "Analysis",
    "Candidate",
    "DatasetChangedError",
    "Excluded",
    "IncorporateError",
    "NewClass",
    "Plan",
    "analyze_incorporation",
    "build_incorporated_dataset",
    "default_ratios",
    "final_classes",
    "link_or_copy",
    "plan_incorporation",
    "resolve_ratios",
    "suggest_name",
]
