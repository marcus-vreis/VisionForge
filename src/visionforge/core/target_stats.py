"""What the targets of a regression dataset look like (ADR-235).

A regression dataset is one CSV per split; each row names an image (relative to
``images_dir``) and one or more numeric targets (ADR-036). This module reads the
CSVs in full and reports, per target and split:

* how many cells are numeric, and how many are blank or not a number (the
  trainer stops on those);
* mean, standard deviation, quartiles, minimum and maximum;
* a histogram on bins shared by every split, so the splits can be compared;
* outliers by Tukey's rule: below ``Q1 - 1.5 IQR`` or above ``Q3 + 1.5 IQR`` of
  the training split, with the row and the image;
* how far the validation and test means sit from training, in training standard
  deviations;

and, across targets, the Pearson correlation of every pair over the rows where
both are numeric. It also checks that every listed image exists, and measures
the brightness, contrast and sharpness of the images that do (ADR-240), on an
evenly spaced sample past ``DEFAULT_IMAGE_LIMIT``. Nothing here writes to the
dataset.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from visionforge.core.cancellation import CancellationToken, is_cancelled
from visionforge.core.image_quality import QualityTally

BINS = 20
TUKEY_K = 1.5
MAX_EXAMPLES = 50
DEFAULT_IMAGE_LIMIT = 20_000
MAX_IMAGE_LIMIT = 500_000


@dataclass
class _Split:
    """The rows of one CSV, as read."""

    present: bool
    rows: int = 0
    values: dict[str, list[float]] = field(default_factory=dict)
    # Per target, (row number, image) of each numeric value, in order.
    where: dict[str, list[tuple[int, str]]] = field(default_factory=dict)
    bad: dict[str, int] = field(default_factory=dict)
    bad_examples: list[dict[str, Any]] = field(default_factory=list)
    # Row number -> value per target, for the correlation.
    by_row: list[dict[str, float]] = field(default_factory=list)
    missing_images: int = 0
    missing_examples: list[dict[str, Any]] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    # The listed images that exist, in CSV order (a repeated name repeats).
    images: list[str] = field(default_factory=list)


def _number(cell: str | None) -> float | None:
    try:
        value = float((cell or "").strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _read_split(
    path: Path,
    images_root: Path,
    image_column: str,
    targets: Sequence[str],
    token: CancellationToken | None,
) -> _Split:
    if not path.is_file():
        return _Split(present=False)
    split = _Split(present=True)
    split.values = {t: [] for t in targets}
    split.where = {t: [] for t in targets}
    split.bad = dict.fromkeys(targets, 0)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        split.missing_columns = [c for c in (image_column, *targets) if c not in header]
        # Row 1 is the header, so the first data row is line 2 of the file.
        for line, row in enumerate(reader, start=2):
            if line % 1024 == 0 and is_cancelled(token):
                break
            split.rows += 1
            _read_row(split, row, line, images_root, image_column, targets)
    return split


def _read_row(
    split: _Split,
    row: Mapping[str, str | None],
    line: int,
    images_root: Path,
    image_column: str,
    targets: Sequence[str],
) -> None:
    image = (row.get(image_column) or "").strip()
    if not image or not (images_root / image).is_file():
        split.missing_images += 1
        if len(split.missing_examples) < MAX_EXAMPLES:
            split.missing_examples.append({"line": line, "image": image})
    else:
        split.images.append(image)
    numbers: dict[str, float] = {}
    for target in targets:
        if target not in row:
            continue
        value = _number(row[target])
        if value is None:
            split.bad[target] += 1
            if len(split.bad_examples) < MAX_EXAMPLES:
                split.bad_examples.append(
                    {"line": line, "target": target, "cell": (row[target] or "")[:40]}
                )
            continue
        numbers[target] = value
        split.values[target].append(value)
        split.where[target].append((line, image))
    split.by_row.append(numbers)


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return dict.fromkeys(("mean", "std", "min", "q1", "median", "q3", "max"))
    ordered = sorted(values)
    q1, median, q3 = (
        statistics.quantiles(ordered, n=4, method="inclusive")
        if len(ordered) > 1
        else (ordered[0],) * 3
    )
    return {
        "mean": statistics.fmean(ordered),
        "std": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "min": ordered[0],
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": ordered[-1],
    }


def histogram(values: list[float], lo: float, hi: float, bins: int = BINS) -> list[int]:
    """Counts on ``bins`` equal bins over [lo, hi]; the top edge is inclusive."""
    counts = [0] * bins
    width = (hi - lo) / bins
    for v in values:
        k = bins - 1 if width == 0 else min(bins - 1, int((v - lo) / width))
        counts[max(0, k)] += 1
    return counts


def pearson(pairs: list[tuple[float, float]]) -> float | None:
    """Pearson's r, or None with fewer than three pairs or a constant side."""
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs, strict=True)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _target_report(target: str, splits: Mapping[str, _Split]) -> dict[str, Any]:
    present = {s: sp for s, sp in splits.items() if sp.present and target in sp.values}
    every = [v for sp in present.values() for v in sp.values[target]]
    lo, hi = (min(every), max(every)) if every else (0.0, 0.0)
    train = present.get("train")
    base = _summary(train.values[target]) if train else _summary([])
    fences = _fences(base)
    report: dict[str, Any] = {
        "name": target,
        "range": [lo, hi],
        "bins": BINS,
        "fences": fences,
        "splits": {},
    }
    for name, sp in present.items():
        values = sp.values[target]
        stats = _summary(values)
        outliers = _outliers(sp, target, fences)
        report["splits"][name] = {
            "count": len(values),
            "bad": sp.bad[target],
            **stats,
            "histogram": histogram(values, lo, hi),
            "outlier_count": len(outliers),
            "outliers": outliers[:MAX_EXAMPLES],
            "shift_sd": _shift(stats["mean"], base),
        }
    return report


def _fences(train: Mapping[str, float | None]) -> list[float] | None:
    q1, q3 = train["q1"], train["q3"]
    if q1 is None or q3 is None:
        return None
    iqr = q3 - q1
    return [q1 - TUKEY_K * iqr, q3 + TUKEY_K * iqr]


def _outliers(
    split: _Split, target: str, fences: list[float] | None
) -> list[dict[str, Any]]:
    if fences is None:
        return []
    low, high = fences
    return [
        {"line": line, "image": image, "value": v}
        for v, (line, image) in zip(
            split.values[target], split.where[target], strict=True
        )
        if v < low or v > high
    ]


def _shift(mean: float | None, train: Mapping[str, float | None]) -> float | None:
    """(split mean - train mean) / train std; None without a spread to divide by."""
    std, train_mean = train["std"], train["mean"]
    if mean is None or train_mean is None or not std:
        return None
    return (mean - train_mean) / std


def _measure_images(
    base: Path,
    images_dir: str,
    splits: Mapping[str, _Split],
    max_images: int,
    token: CancellationToken | None,
    on_progress: Callable[[int, int], None] | None,
    offset: int,
) -> tuple[QualityTally, dict[str, Any]]:
    """Brightness, contrast and sharpness of each split's images (ADR-240).

    An image listed in several rows is measured once per split. Past
    ``max_images`` an evenly spaced sample is measured.
    """
    listed = [
        (name, image)
        for name, sp in splits.items()
        for image in dict.fromkeys(sp.images)
    ]
    limit = max(1, min(max_images, MAX_IMAGE_LIMIT))
    step = max(1, math.ceil(len(listed) / limit)) if listed else 1
    chosen = listed[::step]
    tally = QualityTally()
    unreadable = 0
    for done, (name, image) in enumerate(chosen, start=1):
        if is_cancelled(token):
            break
        rel = f"{images_dir}/{image}" if images_dir else image
        if not tally.measure(name, Path(rel).as_posix(), base / rel):
            unreadable += 1
        if on_progress is not None:
            on_progress(offset + done, offset + len(chosen))
    info = {
        "images_listed": len(listed),
        "images_sampled": step > 1,
        "images_unreadable": unreadable,
    }
    return tally, info


def csv_signature(base: Path, csvs: Mapping[str, str]) -> str:
    """A hash of each CSV's name, size and modification time."""
    digest = hashlib.sha256()
    for split, name in sorted(csvs.items()):
        try:
            stat = os.stat(base / name)
            sized = f"{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            sized = "-"
        digest.update(f"{split}\t{name}\t{sized}\n".encode())
    return digest.hexdigest()


def analyze_targets(
    base: Path,
    csvs: Mapping[str, str],
    *,
    images_dir: str = "images",
    image_column: str = "image",
    targets: Sequence[str] = ("target",),
    max_images: int = DEFAULT_IMAGE_LIMIT,
    token: CancellationToken | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Read every split's CSV and describe each target (see the module doc).

    Progress counts the CSVs, then the CSVs plus the images measured.
    """
    base = Path(base).resolve()
    images_root = base / images_dir
    splits: dict[str, _Split] = {}
    for done, (name, filename) in enumerate(csvs.items(), start=1):
        if is_cancelled(token):
            break
        path = base / filename
        if not path.resolve().is_relative_to(base):
            raise ValueError(f"CSV outside the dataset root: {filename!r}")
        splits[name] = _read_split(path, images_root, image_column, targets, token)
        if on_progress is not None:
            on_progress(done, len(csvs))
    quality, image_info = _measure_images(
        base, images_dir, splits, max_images, token, on_progress, len(csvs)
    )
    cancelled = is_cancelled(token)
    pooled = [row for sp in splits.values() if sp.present for row in sp.by_row]
    return {
        "task": "regression",
        "cancelled": cancelled,
        "quality": quality.report(),
        **image_info,
        "tukey_k": TUKEY_K,
        "splits": {
            name: {
                "present": sp.present,
                "rows": sp.rows,
                "missing_columns": sp.missing_columns,
                "missing_images": sp.missing_images,
                "missing_examples": sp.missing_examples,
                "bad_examples": sp.bad_examples,
            }
            for name, sp in splits.items()
        },
        "targets": [_target_report(t, splits) for t in targets],
        "correlations": [
            {
                "a": a,
                "b": b,
                "r": pearson(
                    [(row[a], row[b]) for row in pooled if a in row and b in row]
                ),
            }
            for a, b in combinations(targets, 2)
        ],
    }
