"""Re-split a dataset by writing a new dataset (ADR-168).

Phase 4 of the datasets surface (issue #117). The same rule as incorporating
(ADR-166) and merging (ADR-167) applies: the opened dataset is never touched,
because its fingerprint (ADR-129) is what old ``run.json`` files remember. The
result is a sibling folder written under a temporary name and published with one
rename (see :mod:`visionforge.core.dataset_derived`).

Two steps, both cancellable where they take time:

* :func:`analyze_resplit` indexes the dataset, hashes every image (SHA-256) and,
  optionally, reads a group manifest (``path,group`` CSV, ADR-150). It decides
  nothing about splits.
* :func:`plan_resplit` (cheap, pure) pools the images, keeps exact copies and
  groups together, and distributes them over the splits by the requested ratios,
  stratified by class and seeded. :func:`build_resplit_dataset` writes the result.

By default the test split is *locked*: it stays exactly as it is, so a model
evaluated before and after is evaluated on the same images. Anything in the pool
that is an exact copy of a locked image, or that shares a group with one, would
leak into training and is therefore dropped (and reported), never moved.
"""

from __future__ import annotations

import csv
import io
import math
import os
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from visionforge.core.cancellation import CancellationToken, is_cancelled
from visionforge.core.dataset_audit import _relative_path
from visionforge.core.dataset_derived import (
    REGISTRY_NAME,
    REGISTRY_SCHEMA,
    YAML_NAMES,
    DerivedError,
    ProgressCounter,
    allocate_largest_remainder,
    claim_name,
    dominant_class,
    ensure_unchanged,
    file_sha256,
    link_or_copy,
    list_files,
    occupied_names,
    prepare_destination,
    publish,
    write_registry,
    write_yaml,
)
from visionforge.core.dataset_explorer import (
    SPLIT_NAMES,
    ExplorerImage,
    ExplorerTask,
    build_index,
)
from visionforge.core.dataset_fingerprint import fingerprint_dataset
from visionforge.core.detection_data import resolve_yolo_split

MAX_REPORTED = 200
DEFAULT_SEED = 0
# The stratum (and the row of the tables) of a detection image without boxes.
BACKGROUND = "(background)"

# Why an image of the pool stays out. The codes are the keys the page translates.
DUPLICATE_OF_TEST = "duplicate_of_test"
GROUP_IN_TEST = "group_in_test"

_TEST = "test"
_BAD_DEST = "resplit.bad_dest_name"
_DEST_EXISTS = "resplit.dest_exists"
_BAD_RATIOS = "resplit.bad_ratios"
_BAD_CSV = "resplit.bad_group_csv"
_UTF8 = "utf-8"


class ResplitError(DerivedError):
    """A refusal with a code the API turns into a translated message."""


class DatasetChangedError(ResplitError):
    """The dataset or the group file changed between the preview and the write."""


@dataclass(frozen=True)
class Item:
    """One image of the opened dataset."""

    split: str
    rel: str
    sha256: str
    # Classification: the class folder. Detection: the most frequent class of the
    # boxes, ``""`` for an image without boxes.
    stratum: str
    label_rel: str | None = None
    group: str | None = None


@dataclass(frozen=True)
class Dropped:
    """An image of the pool left out of the result, and why."""

    rel: str
    split: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.rel, "split": self.split, "reason": self.reason}


@dataclass
class Analysis:
    """What reading the dataset found; input of :func:`plan_resplit`."""

    task: ExplorerTask
    root: Path
    split_dirs: dict[str, str]
    fingerprint: dict[str, Any]
    classes: list[str]
    present: dict[str, bool]
    items: list[Item]
    group_csv: str | None = None
    group_csv_sha256: str | None = None
    # Groups listed in the file; ``None`` when no file was given.
    group_count: int | None = None
    # Images the file does not list (each is its own group).
    ungrouped: int = 0
    # Rows of the file that name no image of the dataset (ignored).
    unknown_entries: int = 0


# ── reading ───────────────────────────────────────────────────────────────────


def _read_groups(root: Path, csv_path: str) -> tuple[dict[str, str], str]:
    """``({relative path: group}, sha256 of the file)`` of a ``path,group`` CSV.

    Raises:
        ResplitError: if the file cannot be read or is not ``path,group``.
    """
    path = Path(csv_path)
    if not path.is_absolute():
        path = root / path
    try:
        # The file is the user's own pick in the local GUI; it is only read.
        raw = path.read_bytes()  # NOSONAR pythonsecurity:S6549
        text = raw.decode("utf-8-sig")
    except OSError as exc:
        raise ResplitError("resplit.group_csv_missing", path=csv_path) from exc
    except UnicodeDecodeError as exc:
        raise ResplitError(_BAD_CSV, detail="not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != ["path", "group"]:
        raise ResplitError(_BAD_CSV, detail="columns must be path,group")
    groups: dict[str, str] = {}
    for number, row in enumerate(reader, start=2):
        rel, group = _csv_row(row, number)
        if rel in groups:
            raise ResplitError(_BAD_CSV, detail=f"line {number}: repeated {rel}")
        groups[rel] = group
    return groups, file_sha256(path)


def _csv_row(row: Mapping[Any, Any], number: int) -> tuple[str, str]:
    group = (row.get("group") or "").strip()
    try:
        rel = _relative_path(row.get("path") or "").as_posix()
    except ValueError as exc:
        raise ResplitError(_BAD_CSV, detail=f"line {number}: bad path") from exc
    if None in row or not group:
        raise ResplitError(_BAD_CSV, detail=f"line {number}: group required")
    return rel, group


def _stratum(task: ExplorerTask, classes: tuple[str, ...]) -> str:
    return (
        classes[0] if task == "classification" and classes else dominant_class(classes)
    )


def analyze_resplit(
    root: Path,
    task: ExplorerTask,
    split_dirs: Mapping[str, str] | None = None,
    group_csv: str | None = None,
    *,
    token: CancellationToken | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> Analysis | None:
    """Index and hash the dataset. ``None`` if cancelled.

    Raises:
        ResplitError: if the folder is unusable or empty, or the group file is.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ResplitError("resplit.base_missing", path=str(root))
    dirs = {s: (split_dirs or {}).get(s, s) for s in SPLIT_NAMES}
    index = build_index(root, task, dirs)
    if not index.images:
        raise ResplitError("resplit.base_empty")
    groups, csv_hash = _read_groups(root, group_csv) if group_csv else ({}, None)
    fingerprint = fingerprint_dataset(root).to_dict()

    progress = ProgressCounter(len(index.images), on_progress)
    items: list[Item] = []
    for image in index.images:
        if is_cancelled(token):
            return None
        items.append(_item(root, task, image, groups))
        progress.tick()
    items.sort(key=lambda i: (i.sha256, i.rel))
    known = {i.rel for i in items}
    return Analysis(
        task=task,
        root=root,
        split_dirs=dirs,
        fingerprint=fingerprint,
        classes=list(index.class_names),
        present=dict(index.present),
        items=items,
        group_csv=group_csv,
        group_csv_sha256=csv_hash,
        group_count=len(set(groups.values())) if group_csv else None,
        ungrouped=sum(1 for i in items if i.group is None) if group_csv else 0,
        unknown_entries=sum(1 for rel in groups if rel not in known),
    )


def _item(
    root: Path, task: ExplorerTask, image: ExplorerImage, groups: Mapping[str, str]
) -> Item:
    try:
        digest = file_sha256(root / image.rel)
    except OSError as exc:
        logger.warning("Could not hash {}", image.rel)
        raise ResplitError("resplit.unreadable", path=image.rel) from exc
    return Item(
        split=image.split,
        rel=image.rel,
        sha256=digest,
        stratum=_stratum(task, image.classes),
        label_rel=image.label_rel,
        group=groups.get(image.rel),
    )


def check_unchanged(analysis: Analysis) -> None:
    """Raise if the dataset or the group file is no longer what was analysed.

    Raises:
        DatasetChangedError: on a different fingerprint or group-file hash.
    """
    ensure_unchanged(
        analysis.root,
        analysis.fingerprint,
        error=lambda: DatasetChangedError("resplit.base_changed"),
    )
    if analysis.group_csv is None:
        return
    path = Path(analysis.group_csv)
    if not path.is_absolute():
        path = analysis.root / path
    try:
        current = file_sha256(path)  # NOSONAR pythonsecurity:S6549
    except OSError:
        current = None
    if current != analysis.group_csv_sha256:
        raise DatasetChangedError("resplit.group_csv_changed")


# ── the plan ──────────────────────────────────────────────────────────────────


class _DisjointSets:
    """Union-find over item indexes."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, node: int) -> int:
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[node] != root:
            self._parent[node], node = root, self._parent[node]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)


@dataclass
class _Unit:
    """Images that must land in the same split: exact copies and group members."""

    items: list[Item]
    key: str
    stratum: str


@dataclass
class Plan:
    """Which image goes to which split, with the numbers the preview shows."""

    task: ExplorerTask
    seed: int
    ratios: dict[str, float]
    locked_splits: list[str]
    by_group: bool
    # relative path -> split, for every image the result keeps.
    final: dict[str, str]
    dropped: list[Dropped]
    items: list[Item]
    moved: int
    # Exact-copy sets kept together: how many, extra copies, and how many of the
    # sets were spread over several splits before.
    duplicate_sets: int = 0
    duplicate_extras: int = 0
    duplicate_sets_split_before: int = 0
    warnings: list[dict[str, object]] = field(default_factory=list)

    def counts(self, *, after: bool) -> dict[str, dict[str, int]]:
        """``{split: {class: images}}`` before the re-split, or in the result."""
        table: dict[str, Counter[str]] = {s: Counter() for s in SPLIT_NAMES}
        for item in self.items:
            split = self.final.get(item.rel) if after else item.split
            if split is not None:
                table[split][item.stratum or BACKGROUND] += 1
        return {s: dict(sorted(table[s].items())) for s in SPLIT_NAMES}

    def split_totals(self, *, after: bool) -> dict[str, int]:
        return {s: sum(c.values()) for s, c in self.counts(after=after).items()}

    def achieved(self) -> dict[str, float]:
        """Share of the re-split images each split got (locked splits excluded)."""
        pooled = {
            s: n
            for s, n in self.split_totals(after=True).items()
            if s not in self.locked_splits
        }
        total = sum(pooled.values())
        return {s: (pooled[s] / total if total else 0.0) for s in pooled}

    def summary(self) -> dict[str, Any]:
        reasons = Counter(d.reason for d in self.dropped)
        return {
            "seed": self.seed,
            "ratios": self.ratios,
            "achieved": self.achieved(),
            "locked_splits": self.locked_splits,
            "by_group": self.by_group,
            "before": self.counts(after=False),
            "after": self.counts(after=True),
            "before_totals": self.split_totals(after=False),
            "after_totals": self.split_totals(after=True),
            "kept": len(self.final),
            "moved": self.moved,
            "dropped_count": len(self.dropped),
            "dropped_by_reason": dict(sorted(reasons.items())),
            "dropped": [d.to_dict() for d in self.dropped[:MAX_REPORTED]],
            "duplicate_sets": self.duplicate_sets,
            "duplicate_extras": self.duplicate_extras,
            "duplicate_sets_split_before": self.duplicate_sets_split_before,
            "warnings": self.warnings,
        }


def _resolve_ratios(
    pool: list[Item], requested: Mapping[str, float] | None, *, locked: bool
) -> dict[str, float]:
    """Normalised ratios of the splits the pool is distributed over.

    Without a request, the proportion the pooled splits have now. A locked split
    gets 0 (it is not distributed); a split the dataset lacks gets a share only if
    the request names it.

    Raises:
        ResplitError: on a negative, unknown or all-zero ratio, or a positive ratio
            for the locked test split.
    """
    if requested is None:
        counts = Counter(i.split for i in pool)
        raw = {s: float(counts[s]) for s in SPLIT_NAMES}
    else:
        raw = dict.fromkeys(SPLIT_NAMES, 0.0)
        for name, value in requested.items():
            if name not in raw or not math.isfinite(value) or value < 0:
                raise ResplitError(_BAD_RATIOS)
            raw[name] = float(value)
        if locked and raw[_TEST] > 0:
            raise ResplitError("resplit.ratio_test_locked")
    total = sum(raw.values())
    if total <= 0:
        raise ResplitError(_BAD_RATIOS)
    return {s: raw[s] / total for s in SPLIT_NAMES}


def _members(items: list[Item], *, by_group: bool) -> list[list[Item]]:
    """Connected sets of items: the same bytes or (``by_group``) the same group."""
    sets = _DisjointSets(len(items))
    first: dict[tuple[str, str], int] = {}
    for idx, item in enumerate(items):
        keys = [("h", item.sha256)]
        if by_group and item.group is not None:
            keys.append(("g", item.group))
        for key in keys:
            if key in first:
                sets.union(first[key], idx)
            else:
                first[key] = idx
    members: dict[int, list[Item]] = defaultdict(list)
    for idx, item in enumerate(items):
        members[sets.find(idx)].append(item)
    return list(members.values())


def _dropped_by_lock(group: list[Item], *, locked: bool) -> list[Dropped]:
    """The pool images of a set that touches the locked split; empty if it does not."""
    held = [i for i in group if locked and i.split == _TEST]
    if not held:
        return []
    locked_hashes = {i.sha256 for i in held}
    return [
        Dropped(
            i.rel,
            i.split,
            DUPLICATE_OF_TEST if i.sha256 in locked_hashes else GROUP_IN_TEST,
        )
        for i in group
        if i not in held
    ]


def _units(
    items: list[Item], *, locked: bool, by_group: bool
) -> tuple[list[_Unit], list[Dropped]]:
    """Connected sets of items (same bytes, or same group); drop the ones that
    touch the locked split, apart from the locked images themselves."""
    units: list[_Unit] = []
    dropped: list[Dropped] = []
    for group in _members(items, by_group=by_group):
        if any(locked and i.split == _TEST for i in group):
            dropped.extend(_dropped_by_lock(group, locked=locked))
        else:
            units.append(_unit(group))
    dropped.sort(key=lambda d: d.rel)
    return units, dropped


def _unit(group: list[Item]) -> _Unit:
    names = sorted({i.group for i in group if i.group is not None})
    key = f"g:{names[0]}" if names else f"h:{min(i.sha256 for i in group)}"
    return _Unit(
        items=group, key=key, stratum=dominant_class([i.stratum for i in group])
    )


def _ordered(units: list[_Unit], seed: int) -> list[_Unit]:
    """Largest units first; equal ones in a seeded order that ignores file order."""
    buckets: dict[tuple[int, str], list[_Unit]] = defaultdict(list)
    for unit in units:
        buckets[(len(unit.items), unit.stratum)].append(unit)
    ordered: list[_Unit] = []
    for size, stratum in sorted(buckets, key=lambda k: (-k[0], k[1])):
        bucket = sorted(buckets[(size, stratum)], key=lambda u: u.key)
        # A reproducible split, not a secret: the seed is recorded on purpose.
        rng = random.Random(f"{seed}|{stratum}|{size}")  # NOSONAR python:S2245
        rng.shuffle(bucket)  # NOSONAR python:S2245
        ordered.extend(bucket)
    return ordered


def _assign(
    units: list[_Unit], ratios: Mapping[str, float], seed: int
) -> dict[str, str]:
    """Greedy, stratified: each unit goes to the split most short of its class.

    The quota of every class per split comes from the largest-remainder method;
    a unit takes the split with the largest remaining deficit for its dominant
    class and, when that class is full everywhere, the one short of images overall.
    """
    sizes: Counter[str] = Counter(i.stratum for u in units for i in u.items)
    targets = {
        stratum: allocate_largest_remainder(
            sizes[stratum],
            ratios,
            # A reproducible split, not a secret.
            random.Random(f"{seed}|{stratum}"),  # NOSONAR python:S2245
        )
        for stratum in sorted(sizes)
    }
    active = [s for s in SPLIT_NAMES if ratios[s] > 0]
    given = {st: dict.fromkeys(SPLIT_NAMES, 0) for st in targets}
    overall = dict.fromkeys(SPLIT_NAMES, 0)
    overall_target = {s: sum(t[s] for t in targets.values()) for s in SPLIT_NAMES}
    final: dict[str, str] = {}
    for unit in _ordered(units, seed):
        dom = targets[unit.stratum]
        short = {s: dom[s] - given[unit.stratum][s] for s in active}
        total_short = {s: overall_target[s] - overall[s] for s in active}
        use_class = max(short.values()) > 0
        split = max(
            active,
            key=lambda s: (
                short[s] if use_class else 0,
                total_short[s],
                ratios[s],
                -SPLIT_NAMES.index(s),
            ),
        )
        for item in unit.items:
            given[item.stratum][split] += 1
            overall[split] += 1
            final[item.rel] = split
    return final


def plan_resplit(
    analysis: Analysis,
    *,
    ratios: Mapping[str, float] | None = None,
    seed: int = DEFAULT_SEED,
    lock_test: bool = True,
    by_group: bool | None = None,
) -> Plan:
    """Distribute the pooled images among the splits.

    Deterministic: the same analysis, ratios, seed and options give the same plan,
    whatever the order the files came in.

    Args:
        ratios: relative weight per split; ``None`` keeps the pooled splits'
            current proportion. 0 removes a split.
        lock_test: keep the test split as it is (only if the dataset has one).
        by_group: use the group file; ``None`` means yes when one was given.

    Raises:
        ResplitError: on bad ratios, groups requested without a file, or nothing
            to distribute.
    """
    use_groups = analysis.group_csv is not None if by_group is None else by_group
    if use_groups and analysis.group_csv is None:
        raise ResplitError("resplit.no_group_csv")
    locked = lock_test and analysis.present.get(_TEST, False)
    pool = [i for i in analysis.items if not (locked and i.split == _TEST)]
    if not pool:
        raise ResplitError("resplit.nothing_to_split")
    effective = _resolve_ratios(pool, ratios, locked=locked)
    units, dropped = _units(analysis.items, locked=locked, by_group=use_groups)
    assigned = _assign(units, effective, seed)
    final = dict(assigned)
    final.update({i.rel: _TEST for i in analysis.items if locked and i.split == _TEST})
    by_rel = {i.rel: i for i in analysis.items}
    plan = Plan(
        task=analysis.task,
        seed=seed,
        ratios=effective,
        locked_splits=[_TEST] if locked else [],
        by_group=use_groups,
        final=final,
        dropped=dropped,
        items=analysis.items,
        moved=sum(1 for rel, split in assigned.items() if by_rel[rel].split != split),
    )
    _describe_duplicates(plan, units)
    plan.warnings = _warnings(analysis, plan)
    return plan


def _describe_duplicates(plan: Plan, units: list[_Unit]) -> None:
    by_sha: dict[str, list[Item]] = defaultdict(list)
    for unit in units:
        for item in unit.items:
            by_sha[item.sha256].append(item)
    copies = [c for c in by_sha.values() if len(c) > 1]
    plan.duplicate_sets = len(copies)
    plan.duplicate_extras = sum(len(c) - 1 for c in copies)
    plan.duplicate_sets_split_before = sum(
        1 for c in copies if len({i.split for i in c}) > 1
    )


def _warnings(analysis: Analysis, plan: Plan) -> list[dict[str, object]]:
    """What the page should flag before the user applies."""
    out: list[dict[str, object]] = []
    totals = plan.split_totals(after=True)
    after = plan.counts(after=True)
    classes = {c for s in SPLIT_NAMES for c in after[s]} - {BACKGROUND}
    for split in SPLIT_NAMES:
        if split in plan.locked_splits:
            continue
        if plan.ratios[split] <= 0:
            if analysis.present.get(split):
                out.append({"code": "split_removed", "split": split})
            continue
        if not analysis.present.get(split):
            out.append({"code": "new_split", "split": split})
        if totals[split] == 0:
            out.append({"code": "empty_split", "split": split})
            continue
        out.extend(
            {"code": "class_missing", "split": split, "class": name}
            for name in sorted(classes)
            if name not in after[split]
        )
    return out[:MAX_REPORTED]


# ── writing ───────────────────────────────────────────────────────────────────


def build_resplit_dataset(
    analysis: Analysis,
    plan: Plan,
    dest_name: str,
    *,
    token: CancellationToken | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    link: Callable[[Path, Path], None] = os.link,
) -> dict[str, Any] | None:
    """Write the new dataset next to the original. ``None`` if cancelled.

    Built under a temporary name in the same parent and renamed to ``dest_name``
    at the very end; on cancellation or any error it is removed and nothing is
    left at the destination.

    Raises:
        ResplitError: bad or existing destination name; the dataset or the group
            file changed since the analysis.
    """
    root = analysis.root
    dest = prepare_destination(
        root.parent,
        dest_name,
        error=ResplitError,
        bad_name_code=_BAD_DEST,
        exists_code=_DEST_EXISTS,
    )
    check_unchanged(analysis)
    result = publish(
        root.parent,
        dest,
        lambda tmp: _write(analysis, plan, tmp, dest, token, on_progress, link),
        error=ResplitError,
        exists_code=_DEST_EXISTS,
    )
    if result is None:
        return None
    result["path"] = str(dest)
    return result


def _layout(analysis: Analysis) -> dict[str, tuple[Path, Path | None]]:
    """Where the images (and labels) of each split live, relative to the root.

    A split the dataset lacks gets the standard YOLO ``images/<s>`` + ``labels/<s>``
    or, for classification, its configured folder name.
    """
    layout: dict[str, tuple[Path, Path | None]] = {}
    for split in SPLIT_NAMES:
        if analysis.task == "classification":
            layout[split] = (Path(analysis.split_dirs[split]), None)
            continue
        resolved = resolve_yolo_split(analysis.root, split)
        if resolved is None:
            layout[split] = (Path("images") / split, Path("labels") / split)
        else:
            layout[split] = (
                resolved[0].relative_to(analysis.root),
                resolved[1].relative_to(analysis.root),
            )
    return layout


def _removed_splits(analysis: Analysis, plan: Plan) -> list[str]:
    return [
        s
        for s in SPLIT_NAMES
        if analysis.present.get(s)
        and plan.ratios[s] <= 0
        and s not in plan.locked_splits
    ]


def _new_splits(analysis: Analysis, plan: Plan) -> list[str]:
    return [
        s
        for s in SPLIT_NAMES
        if not analysis.present.get(s) and plan.split_totals(after=True)[s] > 0
    ]


class _Writer:
    """Puts files into the temporary folder: the mirror, then the moved images."""

    def __init__(
        self,
        analysis: Analysis,
        plan: Plan,
        tmp: Path,
        link: Callable[[Path, Path], None],
    ) -> None:
        self.analysis = analysis
        self.plan = plan
        self.tmp = tmp
        self.link = link
        self.layout = _layout(analysis)
        self.methods: Counter[str] = Counter()
        self._taken: dict[Path, set[str]] = {}

    def mirror(self, rel: str) -> None:
        target = self.tmp / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        self.methods[
            link_or_copy(self.analysis.root / rel, target, link=self.link)
        ] += 1

    def place(self, item: Item) -> None:
        """Put one image (and its label) in its new split, under a free name."""
        detection = self.analysis.task == "detection"
        images_dir, labels_dir = self.layout[self.plan.final[item.rel]]
        folder = images_dir if detection else images_dir / item.stratum
        (self.tmp / folder).mkdir(parents=True, exist_ok=True)
        if labels_dir is not None:
            (self.tmp / labels_dir).mkdir(parents=True, exist_ok=True)
        if folder not in self._taken:
            self._taken[folder] = occupied_names(self.tmp, folder, labels_dir)
        name = claim_name(
            self._taken[folder], Path(item.rel).name, item.sha256, with_label=detection
        )
        root = self.analysis.root
        self.methods[
            link_or_copy(root / item.rel, self.tmp / folder / name, link=self.link)
        ] += 1
        if detection and item.label_rel and labels_dir is not None:
            label = self.tmp / labels_dir / f"{Path(name).stem}.txt"
            self.methods[
                link_or_copy(root / item.label_rel, label, link=self.link)
            ] += 1


def _write(
    analysis: Analysis,
    plan: Plan,
    tmp: Path,
    dest: Path,
    token: CancellationToken | None,
    on_progress: Callable[[int, int], None] | None,
    link: Callable[[Path, Path], None],
) -> dict[str, Any] | None:
    writer = _Writer(analysis, plan, tmp, link)
    removed = _removed_splits(analysis, plan)
    movers = [i for i in analysis.items if plan.final.get(i.rel, i.split) != i.split]
    skip = _skipped_files(analysis, plan, movers, writer.layout, removed)
    files = list_files(analysis.root)
    progress = ProgressCounter(len(files) + len(movers), on_progress)

    tmp.mkdir()
    for rel in files:
        if is_cancelled(token):
            return None
        if rel not in skip:
            writer.mirror(rel)
        progress.tick()
    for item in sorted(movers, key=lambda i: i.rel):
        if is_cancelled(token):
            return None
        writer.place(item)
        progress.tick()

    new = _new_splits(analysis, plan)
    _write_yaml_files(analysis, tmp, dest, writer.layout, new, removed)
    registry = _registry(analysis, plan, writer.methods)
    write_registry(tmp, registry)
    if is_cancelled(token):
        return None
    return {
        "registry": registry,
        "images": len(plan.final),
        "moved": plan.moved,
        "link_methods": dict(writer.methods),
    }


def _skipped_files(
    analysis: Analysis,
    plan: Plan,
    movers: list[Item],
    layout: Mapping[str, tuple[Path, Path | None]],
    removed: list[str],
) -> set[str]:
    """Files the plain mirror must not link: those that move, those dropped, the
    registry, the YOLO yaml (rewritten) and anything inside a removed split."""
    skip = {REGISTRY_NAME}
    skip.update(n for n in YAML_NAMES if (analysis.root / n).is_file())
    leaving = [*movers, *(i for i in analysis.items if i.rel not in plan.final)]
    for item in leaving:
        skip.add(item.rel)
        if item.label_rel:
            skip.add(item.label_rel)
    gone = [d for s in removed for d in layout[s] if d is not None]
    prefixes = tuple(f"{d.as_posix()}/" for d in gone)
    if prefixes:
        skip.update(
            rel for rel in list_files(analysis.root) if rel.startswith(prefixes)
        )
    return skip


def _write_yaml_files(
    analysis: Analysis,
    tmp: Path,
    dest: Path,
    layout: Mapping[str, tuple[Path, Path | None]],
    new: list[str],
    removed: list[str],
) -> None:
    """The YOLO yaml, re-written: a hardlink would share bytes with the original,
    its absolute ``path`` must point at the new folder, and the split keys follow
    the splits that exist now. The class list is not changed."""
    extra = {s: layout[s][0].as_posix() for s in new}
    for name in YAML_NAMES:
        source = analysis.root / name
        if source.is_file():
            write_yaml(
                source,
                tmp / name,
                analysis.classes,
                analysis.root,
                dest,
                extra_splits=extra,
                drop_splits=removed,
            )


def _registry(analysis: Analysis, plan: Plan, methods: Counter[str]) -> dict[str, Any]:
    reasons = Counter(d.reason for d in plan.dropped)
    group_manifest = None
    if analysis.group_csv is not None:
        group_manifest = {
            "path": analysis.group_csv,
            "sha256": analysis.group_csv_sha256,
            "groups": analysis.group_count,
            "images_without_group": analysis.ungrouped,
            "unknown_entries": analysis.unknown_entries,
            "used": plan.by_group,
        }
    return {
        "schema_version": REGISTRY_SCHEMA,
        "kind": "resplit",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "task": analysis.task,
        "derived_from": {
            "path": str(analysis.root),
            "fingerprint": analysis.fingerprint,
        },
        "seed": plan.seed,
        "requested_ratios": plan.ratios,
        "achieved_ratios": plan.achieved(),
        "lock_test": bool(plan.locked_splits),
        "locked_splits": plan.locked_splits,
        "group_manifest": group_manifest,
        "split_dirs": analysis.split_dirs,
        "classes": analysis.classes,
        "counts": {
            "before": plan.split_totals(after=False),
            "after": plan.split_totals(after=True),
            "before_per_class": plan.counts(after=False),
            "after_per_class": plan.counts(after=True),
            "moved": plan.moved,
            "dropped": len(plan.dropped),
            "dropped_by_reason": dict(sorted(reasons.items())),
            "dropped_listed": [d.to_dict() for d in plan.dropped[:MAX_REPORTED]],
            "duplicate_sets": plan.duplicate_sets,
            "duplicate_extras": plan.duplicate_extras,
            "duplicate_sets_split_before": plan.duplicate_sets_split_before,
        },
        "files_linked": methods.get("link", 0),
        "files_copied": methods.get("copy", 0),
    }


def suggest_name(root: Path) -> str:
    """``<name>-resplit``, or the first ``-N`` that does not exist next to it."""
    root = Path(root).resolve()
    stem = f"{root.name}-resplit"
    name, n = stem, 2
    while (root.parent / name).exists():
        name = f"{stem}-{n}"
        n += 1
    return name


__all__ = [
    "BACKGROUND",
    "DEFAULT_SEED",
    "DUPLICATE_OF_TEST",
    "GROUP_IN_TEST",
    "REGISTRY_NAME",
    "Analysis",
    "DatasetChangedError",
    "Dropped",
    "Item",
    "Plan",
    "ResplitError",
    "analyze_resplit",
    "build_resplit_dataset",
    "check_unchanged",
    "plan_resplit",
    "suggest_name",
]
