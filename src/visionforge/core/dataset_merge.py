"""Merge two datasets into a new one, with a class mapping (ADR-167).

Phase 3 of the datasets surface (issue #117). Like incorporating images (ADR-166),
merging never edits a dataset: the result is a new sibling folder of dataset A,
built under a temporary name and published with one rename (see
:mod:`visionforge.core.dataset_derived`). Neither source is touched.

Two steps, both cancellable and with progress:

* :func:`analyze_merge` indexes both datasets, hashes every image (SHA-256, the
  same digest the audit and the explorer use) and finds the images that are in
  both. It decides nothing about classes.
* :func:`plan_merge` (cheap, pure) applies a class mapping (rename, merge,
  discard) and says what the result holds; :func:`build_merged_dataset` writes it.

Rules worth knowing:

* Every image keeps its own split. No re-splitting; that is another phase.
* An image whose bytes are in both datasets is kept once, from A. If A's copy does
  not survive the mapping, B's copy is kept instead, so nothing is lost to a
  duplicate of something the user discarded.
* The class order is A's surviving targets in A's order, then the new targets of
  B in B's order, ids renumbered from 0.
* Only dataset content is carried (images, YOLO labels, the class list and the
  registry). Other files of the sources are not copied: a renamed class folder
  would otherwise drag stray files into a folder that looks like a class.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any

from visionforge.core.cancellation import CancellationToken, is_cancelled
from visionforge.core.dataset_derived import (
    REGISTRY_SCHEMA,
    DerivedError,
    ProgressCounter,
    claim_name,
    ensure_unchanged,
    file_sha256,
    link_or_copy,
    prepare_destination,
    publish,
    write_class_files,
    write_registry,
)
from visionforge.core.dataset_explorer import (
    SPLIT_NAMES,
    DatasetIndex,
    ExplorerImage,
    ExplorerTask,
    build_index,
)
from visionforge.core.dataset_fingerprint import fingerprint_dataset
from visionforge.core.detection_data import resolve_yolo_split
from visionforge.utils.paths import validate_path_component

MAX_REPORTED = 200

_CLASSIFICATION = "classification"
_DETECTION = "detection"
_IMAGES = "images"
_BOXES = "boxes"

SIDES = ("a", "b")
# Why an image of a source stays out. The codes are the keys the page translates.
DISCARDED_CLASS = "discarded_class"
ALL_BOXES_DISCARDED = "all_boxes_discarded"

# ``{"a": {class: target or None}, "b": {...}}``; ``None`` discards the class.
MappingInput = Mapping[str, Mapping[str, str | None]]

_CHANGED = {"a": "merge.a_changed", "b": "merge.b_changed"}
_BAD_DEST = "merge.bad_dest_name"
_DEST_EXISTS = "merge.dest_exists"
_SEPARATORS = re.compile(r"[\s_\-]+")


class MergeError(DerivedError):
    """A refusal with a code the API turns into a translated message."""


class MergeChangedError(MergeError):
    """A source changed between the preview and the write."""


@dataclass(frozen=True)
class MergeImage:
    """One image of a source. ``rel`` is POSIX and relative to its dataset root."""

    side: str
    split: str
    rel: str
    sha256: str
    size: int
    mtime_ns: int
    # Classification: ``(folder,)``. Detection: the class name of every box.
    classes: tuple[str, ...]
    label_rel: str | None = None
    # Detection only: ``(id in its own dataset, "cx cy w h")`` per valid row.
    rows: tuple[tuple[int, str], ...] = ()
    # Images of B only: indices (in A's list) of the A images with the same bytes.
    dup_of: tuple[int, ...] = ()


@dataclass(frozen=True)
class SourceClass:
    """A class of one source, as a row of the mapping table."""

    side: str
    name: str
    images: int


@dataclass
class MergeAnalysis:
    """What reading both datasets found; input of :func:`plan_merge`."""

    task: ExplorerTask
    roots: dict[str, Path]
    split_dirs: dict[str, dict[str, str]]
    fingerprints: dict[str, dict[str, Any]]
    images: dict[str, list[MergeImage]]
    classes: dict[str, list[SourceClass]]
    present: dict[str, dict[str, bool]]
    found: dict[str, int]
    # Images that repeat inside one source (kept; only counted).
    repeated_within: dict[str, int] = field(default_factory=dict)
    # Groups of class names that differ only by case or spacing.
    similar_names: list[list[str]] = field(default_factory=list)


# ── reading ───────────────────────────────────────────────────────────────────


def _split_dirs(given: Mapping[str, str] | None) -> dict[str, str]:
    return {s: (given or {}).get(s, s) for s in SPLIT_NAMES}


def check_sources(a_root: Path, b_root: Path) -> None:
    """Raise :class:`MergeError` unless both are folders and neither holds the other."""
    if not a_root.is_dir():
        raise MergeError("merge.a_missing", path=str(a_root))
    if not b_root.is_dir():
        raise MergeError("merge.b_missing", path=str(b_root))
    if a_root.is_relative_to(b_root) or b_root.is_relative_to(a_root):
        raise MergeError("merge.datasets_overlap")


def _read_rows(path: Path) -> tuple[tuple[int, str], ...]:
    """The valid ``id cx cy w h`` rows of a YOLO label (others are ignored)."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ()
    rows: list[tuple[int, str]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            values = [float(p) for p in parts]
        except ValueError:
            continue
        class_id = values[0]
        if not all(math.isfinite(v) for v in values) or class_id < 0:
            continue
        if class_id != int(class_id):
            continue
        rows.append((int(class_id), " ".join(parts[1:])))
    return tuple(rows)


def _make_image(
    side: str,
    index: DatasetIndex,
    image: ExplorerImage,
    sha: str,
    dup_of: tuple[int, ...],
) -> MergeImage:
    path = index.root / image.rel
    stat = path.stat()
    classes = image.classes
    rows: tuple[tuple[int, str], ...] = ()
    if index.task == _DETECTION:
        names = index.class_names
        if image.label_rel is not None:
            rows = _read_rows(index.root / image.label_rel)
        classes = tuple(names[i] if i < len(names) else f"class_{i}" for i, _ in rows)
    return MergeImage(
        side=side,
        split=image.split,
        rel=image.rel,
        sha256=sha,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        classes=classes,
        label_rel=image.label_rel,
        rows=rows,
        dup_of=dup_of,
    )


def _read_side(
    side: str,
    index: DatasetIndex,
    known: Mapping[str, list[int]],
    progress: ProgressCounter,
    token: CancellationToken | None,
) -> tuple[list[MergeImage], dict[str, list[int]], int] | None:
    """Hash and parse every image of a source. ``None`` if cancelled.

    ``known``: hashes of A's images, so B's duplicates of them get ``dup_of``.
    Returns the images, the indices per hash and how many repeat inside the source.
    """
    images: list[MergeImage] = []
    by_sha: dict[str, list[int]] = {}
    repeated = 0
    for position, image in enumerate(index.images):
        if is_cancelled(token):
            return None
        try:
            sha = file_sha256(index.root / image.rel)
        except OSError as exc:
            raise MergeError("merge.unreadable", path=image.rel) from exc
        repeated += sha in by_sha
        by_sha.setdefault(sha, []).append(position)
        images.append(_make_image(side, index, image, sha, tuple(known.get(sha, ()))))
        progress.tick()
    return images, by_sha, repeated


def _source_classes(
    side: str, index: DatasetIndex, images: list[MergeImage]
) -> list[SourceClass]:
    """The classes of a source in its own order, with the images each one has.

    Images that B shares with A are not counted: they will not be written twice.
    """
    counts: Counter[str] = Counter()
    for image in images:
        if not image.dup_of:
            counts.update(set(image.classes))
    names = list(index.class_names)
    names += sorted(n for n in counts if n not in set(names))
    return [SourceClass(side, name, counts.get(name, 0)) for name in names]


def _similar_names(classes: Mapping[str, list[SourceClass]]) -> list[list[str]]:
    """Groups of names that are the same once case, spaces, ``_`` and ``-`` go."""
    groups: dict[str, set[str]] = defaultdict(set)
    for rows in classes.values():
        for row in rows:
            groups[_SEPARATORS.sub("", row.name).casefold()].add(row.name)
    return [sorted(g) for _, g in sorted(groups.items()) if len(g) > 1]


def _open_side(
    side: str, root: Path, task: ExplorerTask, dirs: Mapping[str, str]
) -> DatasetIndex:
    index = build_index(root, task, dirs)
    if not index.images:
        raise MergeError("merge.empty", side=side.upper())
    return index


def analyze_merge(
    a_root: Path,
    task: ExplorerTask,
    b_root: Path,
    a_split_dirs: Mapping[str, str] | None = None,
    b_split_dirs: Mapping[str, str] | None = None,
    *,
    token: CancellationToken | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> MergeAnalysis | None:
    """Read both datasets and find the images they share. ``None`` if cancelled.

    Raises:
        MergeError: if either folder is unusable or empty, or one contains the other.
    """
    roots = {"a": Path(a_root).resolve(), "b": Path(b_root).resolve()}
    check_sources(roots["a"], roots["b"])
    dirs = {"a": _split_dirs(a_split_dirs), "b": _split_dirs(b_split_dirs)}
    indexes = {s: _open_side(s, roots[s], task, dirs[s]) for s in SIDES}
    fingerprints = {s: fingerprint_dataset(roots[s]).to_dict() for s in SIDES}
    progress = ProgressCounter(
        sum(len(i.images) for i in indexes.values()), on_progress
    )

    read_a = _read_side("a", indexes["a"], {}, progress, token)
    if read_a is None:
        return None
    read_b = _read_side("b", indexes["b"], read_a[1], progress, token)
    if read_b is None:
        return None
    images = {"a": read_a[0], "b": read_b[0]}
    classes = {s: _source_classes(s, indexes[s], images[s]) for s in SIDES}
    return MergeAnalysis(
        task=task,
        roots=roots,
        split_dirs=dirs,
        fingerprints=fingerprints,
        images=images,
        classes=classes,
        present={s: dict(indexes[s].present) for s in SIDES},
        found={s: len(images[s]) for s in SIDES},
        repeated_within={"a": read_a[2], "b": read_b[2]},
        similar_names=_similar_names(classes),
    )


# ── the mapping ───────────────────────────────────────────────────────────────


def _clean_target(raw: str, name: str, task: ExplorerTask) -> str:
    target = raw.strip()
    if not target:
        raise MergeError("merge.empty_target", name=name)
    if any(c in target for c in "\r\n"):
        raise MergeError("merge.bad_target", name=target)
    if task == _CLASSIFICATION:
        try:
            validate_path_component(target)
        except ValueError as exc:
            raise MergeError("merge.bad_target", name=target) from exc
    return target


def _check_case_clash(targets: set[str]) -> None:
    """Folders ``Cat`` and ``cat`` are one folder on Windows; refuse to guess."""
    seen: dict[str, str] = {}
    for target in sorted(targets):
        other = seen.setdefault(target.casefold(), target)
        if other != target:
            raise MergeError("merge.targets_differ_by_case", first=other, second=target)


def resolve_mapping(
    analysis: MergeAnalysis, mapping: MappingInput | None
) -> dict[str, dict[str, str | None]]:
    """The target of every class of both sources (``None``: discarded).

    A class the mapping does not mention keeps its own name, so classes with the
    same name in both datasets merge without anyone asking for it.

    Raises:
        MergeError: on an unknown side or class, or a target that is empty, not a
            valid folder name (classification) or differs from another only by case.
    """
    given = mapping or {}
    extra = [s for s in given if s not in SIDES]
    if extra:
        raise MergeError("merge.bad_mapping", name=extra[0])
    resolved: dict[str, dict[str, str | None]] = {}
    for side in SIDES:
        known = {c.name for c in analysis.classes[side]}
        chosen = given.get(side, {})
        unknown = [n for n in chosen if n not in known]
        if unknown:
            raise MergeError("merge.unknown_class", name=unknown[0], side=side.upper())
        resolved[side] = {}
        for row in analysis.classes[side]:
            value = chosen.get(row.name, row.name)
            resolved[side][row.name] = (
                None if value is None else _clean_target(value, row.name, analysis.task)
            )
    if analysis.task == _CLASSIFICATION:
        _check_case_clash({t for side in resolved.values() for t in side.values() if t})
    return resolved


def final_classes(
    analysis: MergeAnalysis, resolved: Mapping[str, Mapping[str, str | None]]
) -> list[str]:
    """A's surviving targets in A's order, then B's new ones in B's order."""
    classes: list[str] = []
    seen: set[str] = set()
    for side in SIDES:
        for row in analysis.classes[side]:
            target = resolved[side][row.name]
            if target is not None and target not in seen:
                seen.add(target)
                classes.append(target)
    return classes


# ── the plan ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Placement:
    """An image of the result and what its label holds there."""

    image: MergeImage
    # Classification: ``(target,)``. Detection: the target of every kept box.
    names: tuple[str, ...]
    # Detection: the kept rows with their new ids.
    rows: tuple[tuple[int, str], ...] = ()
    # The label needs no rewrite: no id moved and no box went.
    same_label: bool = True


@dataclass(frozen=True)
class Dropped:
    """An image of a source left out, and why."""

    side: str
    path: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"side": self.side, "path": self.path, "reason": self.reason}


@dataclass(frozen=True)
class DroppedDuplicate:
    """An image of B left out because A has the same bytes."""

    b_path: str
    a_path: str
    b_split: str
    a_split: str

    @property
    def cross_split(self) -> bool:
        """The two copies sit in different splits: a leak the merge avoids."""
        return self.a_split != self.b_split

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "b_path": self.b_path,
            "a_path": self.a_path,
            "b_split": self.b_split,
            "a_split": self.a_split,
            "cross_split": self.cross_split,
        }


@dataclass
class MergePlan:
    """What the result holds, with the numbers the preview shows."""

    task: ExplorerTask
    mapping: dict[str, dict[str, str | None]]
    classes: list[str]
    placements: list[Placement]
    dropped: list[Dropped]
    duplicates: list[DroppedDuplicate]
    present: dict[str, bool]
    only_in: dict[str, list[str]]

    def per_split(self) -> dict[str, dict[str, int]]:
        table = {s: {"a": 0, "b": 0} for s in SPLIT_NAMES}
        for p in self.placements:
            table[p.image.split][p.image.side] += 1
        return table

    def per_class(self) -> dict[str, dict[str, dict[str, int]]]:
        """``{class: {"images": {split: n}, "boxes": {split: n}}}``."""
        zero = dict.fromkeys(SPLIT_NAMES, 0)
        table = {n: {_IMAGES: dict(zero), _BOXES: dict(zero)} for n in self.classes}
        for p in self.placements:
            for name in set(p.names):
                table[name][_IMAGES][p.image.split] += 1
            if self.task == _DETECTION:
                for name in p.names:
                    table[name][_BOXES][p.image.split] += 1
        return table

    def background(self) -> dict[str, int]:
        count = dict.fromkeys(SPLIT_NAMES, 0)
        for p in self.placements:
            if not p.names:
                count[p.image.split] += 1
        return count

    def summary(self) -> dict[str, Any]:
        reasons = Counter(d.reason for d in self.dropped)
        splits = self.per_split()
        return {
            "classes": [
                {
                    "name": name,
                    _IMAGES: {**row[_IMAGES], "total": sum(row[_IMAGES].values())},
                    _BOXES: sum(row[_BOXES].values()),
                }
                for name, row in self.per_class().items()
            ],
            "background": self.background(),
            "splits": {
                s: {
                    "present": self.present[s],
                    _IMAGES: splits[s]["a"] + splits[s]["b"],
                    "from_a": splits[s]["a"],
                    "from_b": splits[s]["b"],
                }
                for s in SPLIT_NAMES
            },
            "only_in": self.only_in,
            "images_total": len(self.placements),
            "dropped_count": len(self.dropped),
            "dropped_by_reason": dict(sorted(reasons.items())),
            "dropped": [d.to_dict() for d in self.dropped[:MAX_REPORTED]],
            "duplicates_dropped": len(self.duplicates),
            "duplicates_cross_split": sum(d.cross_split for d in self.duplicates),
            "duplicates": [d.to_dict() for d in self.duplicates[:MAX_REPORTED]],
        }


def _project(
    image: MergeImage,
    targets: Mapping[str, str | None],
    ids: Mapping[str, int],
    task: ExplorerTask,
) -> Placement | str:
    """The image as the result has it, or the reason it is left out."""
    if task == _CLASSIFICATION:
        target = targets[image.classes[0]]
        return DISCARDED_CLASS if target is None else Placement(image, (target,))
    names: list[str] = []
    rows: list[tuple[int, str]] = []
    for (_, rest), name in zip(image.rows, image.classes, strict=True):
        target = targets[name]
        if target is not None:
            names.append(target)
            rows.append((ids[target], rest))
    if image.rows and not rows:
        return ALL_BOXES_DISCARDED
    same = len(rows) == len(image.rows) and all(
        new == old for (new, _), (old, _) in zip(rows, image.rows, strict=True)
    )
    return Placement(image, tuple(names), tuple(rows), same)


def _only_in(present: Mapping[str, Mapping[str, bool]]) -> dict[str, list[str]]:
    """Splits that exactly one source has (their images are all from that source)."""
    return {
        s: [n for n in SPLIT_NAMES if present[s].get(n) and not present[o].get(n)]
        for s, o in (("a", "b"), ("b", "a"))
    }


def plan_merge(analysis: MergeAnalysis, mapping: MappingInput | None) -> MergePlan:
    """Apply the class mapping to both datasets.

    Deterministic: A's images first, then B's, each in its dataset's own order.

    Raises:
        MergeError: on a bad mapping, or when nothing would be left.
    """
    resolved = resolve_mapping(analysis, mapping)
    classes = final_classes(analysis, resolved)
    ids = {name: i for i, name in enumerate(classes)}
    placements: list[Placement] = []
    dropped: list[Dropped] = []
    duplicates: list[DroppedDuplicate] = []
    survivors: set[int] = set()
    for side in SIDES:
        for position, image in enumerate(analysis.images[side]):
            twins = [analysis.images["a"][i] for i in image.dup_of if i in survivors]
            if twins:
                duplicates.append(
                    DroppedDuplicate(
                        image.rel, twins[0].rel, image.split, twins[0].split
                    )
                )
                continue
            outcome = _project(image, resolved[side], ids, analysis.task)
            if isinstance(outcome, str):
                dropped.append(Dropped(side, image.rel, outcome))
                continue
            placements.append(outcome)
            if side == "a":
                survivors.add(position)
    if not placements:
        raise MergeError("merge.nothing_left")
    present = {
        s: any(analysis.present[side].get(s) for side in SIDES) for s in SPLIT_NAMES
    }
    return MergePlan(
        task=analysis.task,
        mapping=resolved,
        classes=classes,
        placements=placements,
        dropped=dropped,
        duplicates=duplicates,
        present=present,
        only_in=_only_in(analysis.present),
    )


# ── writing ───────────────────────────────────────────────────────────────────


def _same_file(analysis: MergeAnalysis, image: MergeImage) -> None:
    stat = (analysis.roots[image.side] / image.rel).stat()
    if stat.st_size != image.size or stat.st_mtime_ns != image.mtime_ns:
        raise MergeChangedError("merge.image_changed", path=image.rel)


def _layout(
    analysis: MergeAnalysis, plan: MergePlan
) -> dict[str, tuple[Path, Path | None]]:
    """Where the images (and labels) of each split go, relative to the result.

    Detection follows A's layout for a split A has; a split only B has gets the
    standard ``images/<split>`` + ``labels/<split>``.
    """
    layout: dict[str, tuple[Path, Path | None]] = {}
    root = analysis.roots["a"]
    for split in SPLIT_NAMES:
        if not plan.present[split]:
            continue
        if analysis.task == _CLASSIFICATION:
            layout[split] = (Path(analysis.split_dirs["a"][split]), None)
            continue
        resolved = resolve_yolo_split(root, split)
        layout[split] = (
            (resolved[0].relative_to(root), resolved[1].relative_to(root))
            if resolved is not None
            else (Path(_IMAGES) / split, Path("labels") / split)
        )
    return layout


def _classification_folder(
    analysis: MergeAnalysis,
    placement: Placement,
    layout: Mapping[str, tuple[Path, Path | None]],
) -> Path:
    """``<split dir of A>/<target>/<sub folders the image had in its class>``."""
    image = placement.image
    own = analysis.split_dirs[image.side][image.split]
    inner = PurePosixPath(image.rel).relative_to(f"{own}/{image.classes[0]}")
    return layout[image.split][0] / placement.names[0] / Path(*inner.parent.parts)


def _write_label(
    analysis: MergeAnalysis,
    placement: Placement,
    target: Path,
    link: Callable[[Path, Path], None],
    methods: Counter[str],
) -> None:
    """Link the label when it is unchanged, else write the remapped one fresh."""
    image = placement.image
    if placement.same_label:
        if image.label_rel is not None:
            source = analysis.roots[image.side] / image.label_rel
            methods[link_or_copy(source, target, link=link)] += 1
        return
    lines = [f"{new} {rest}" for new, rest in placement.rows]
    target.write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n"
    )


def _place(
    analysis: MergeAnalysis,
    placement: Placement,
    tmp: Path,
    layout: Mapping[str, tuple[Path, Path | None]],
    taken: dict[Path, set[str]],
    methods: Counter[str],
    link: Callable[[Path, Path], None],
) -> None:
    image = placement.image
    _same_file(analysis, image)
    detection = analysis.task == _DETECTION
    images_dir, labels_dir = layout[image.split]
    folder = (
        images_dir if detection else _classification_folder(analysis, placement, layout)
    )
    (tmp / folder).mkdir(parents=True, exist_ok=True)
    name = claim_name(
        taken.setdefault(folder, set()),
        Path(image.rel).name,
        image.sha256,
        with_label=detection,
    )
    source = analysis.roots[image.side] / image.rel
    methods[link_or_copy(source, tmp / folder / name, link=link)] += 1
    if detection and labels_dir is not None:
        (tmp / labels_dir).mkdir(parents=True, exist_ok=True)
        label = tmp / labels_dir / f"{Path(name).stem}.txt"
        _write_label(analysis, placement, label, link, methods)


def _extra_splits(
    analysis: MergeAnalysis, layout: Mapping[str, tuple[Path, Path | None]]
) -> dict[str, str]:
    """Splits the result has that A's ``data.yaml`` may not name yet."""
    return {
        split: images_dir.as_posix()
        for split, (images_dir, _) in layout.items()
        if not analysis.present["a"].get(split)
    }


def _write(
    analysis: MergeAnalysis,
    plan: MergePlan,
    tmp: Path,
    dest: Path,
    token: CancellationToken | None,
    on_progress: Callable[[int, int], None] | None,
    link: Callable[[Path, Path], None],
) -> dict[str, Any] | None:
    progress = ProgressCounter(len(plan.placements), on_progress)
    methods: Counter[str] = Counter()
    layout = _layout(analysis, plan)
    taken: dict[Path, set[str]] = {}
    tmp.mkdir()
    for placement in plan.placements:
        if is_cancelled(token):
            return None
        _place(analysis, placement, tmp, layout, taken, methods, link)
        progress.tick()
    if analysis.task == _DETECTION:
        write_class_files(
            analysis.roots["a"],
            tmp,
            dest,
            plan.classes,
            extra_splits=_extra_splits(analysis, layout),
        )
    registry = _registry(analysis, plan, methods)
    write_registry(tmp, registry)
    if is_cancelled(token):
        return None
    return {"registry": registry, _IMAGES: len(plan.placements)}


def ensure_sources_unchanged(analysis: MergeAnalysis) -> None:
    """Raise :class:`MergeChangedError` if A or B no longer match the analysis."""
    for side in SIDES:
        ensure_unchanged(
            analysis.roots[side],
            analysis.fingerprints[side],
            error=partial(MergeChangedError, _CHANGED[side]),
        )


def build_merged_dataset(
    analysis: MergeAnalysis,
    plan: MergePlan,
    dest_name: str,
    *,
    token: CancellationToken | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    link: Callable[[Path, Path], None] = os.link,
) -> dict[str, Any] | None:
    """Write the merged dataset next to dataset A. ``None`` if cancelled.

    Built under a temporary name in the same parent and renamed to ``dest_name``
    at the very end; on cancellation or any error it is removed and nothing is
    left at the destination.

    Raises:
        MergeError: bad or existing destination name; either source changed since
            the analysis.
    """
    parent = analysis.roots["a"].parent
    dest = prepare_destination(
        parent,
        dest_name,
        error=MergeError,
        bad_name_code=_BAD_DEST,
        exists_code=_DEST_EXISTS,
    )
    ensure_sources_unchanged(analysis)
    result = publish(
        parent,
        dest,
        lambda tmp: _write(analysis, plan, tmp, dest, token, on_progress, link),
        error=MergeError,
        exists_code=_DEST_EXISTS,
    )
    if result is None:
        return None
    result["path"] = str(dest)
    return result


def _registry(
    analysis: MergeAnalysis, plan: MergePlan, methods: Counter[str]
) -> dict[str, Any]:
    reasons = Counter(d.reason for d in plan.dropped)
    return {
        "schema_version": REGISTRY_SCHEMA,
        "kind": "merged",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "task": analysis.task,
        "sources": {
            side: {
                "path": str(analysis.roots[side]),
                "fingerprint": analysis.fingerprints[side],
            }
            for side in SIDES
        },
        "split_dirs": analysis.split_dirs["a"],
        "mapping": plan.mapping,
        "classes": plan.classes,
        "counts": {
            "found": analysis.found,
            _IMAGES: len(plan.placements),
            "per_split": plan.per_split(),
            "per_class": plan.per_class(),
            "images_dropped": dict(sorted(reasons.items())),
            "duplicates_dropped": len(plan.duplicates),
            "duplicates_cross_split": sum(d.cross_split for d in plan.duplicates),
            "repeated_within_source": analysis.repeated_within,
        },
        "files_linked": methods.get("link", 0),
        "files_copied": methods.get("copy", 0),
    }


def suggest_merge_name(a_root: Path, b_root: Path) -> str:
    """``<A>-<B>``, or the first ``-N`` that is free next to dataset A."""
    a_root = Path(a_root).resolve()
    stem = f"{a_root.name}-{Path(b_root).resolve().name}"
    try:
        validate_path_component(stem)
    except ValueError:
        stem = f"{a_root.name}-merged"
    name, n = stem, 2
    while (a_root.parent / name).exists():
        name = f"{stem}-{n}"
        n += 1
    return name


__all__ = [
    "ALL_BOXES_DISCARDED",
    "DISCARDED_CLASS",
    "SIDES",
    "Dropped",
    "DroppedDuplicate",
    "MergeAnalysis",
    "MergeChangedError",
    "MergeError",
    "MergeImage",
    "MergePlan",
    "Placement",
    "SourceClass",
    "analyze_merge",
    "build_merged_dataset",
    "check_sources",
    "ensure_sources_unchanged",
    "final_classes",
    "plan_merge",
    "resolve_mapping",
    "suggest_merge_name",
]
