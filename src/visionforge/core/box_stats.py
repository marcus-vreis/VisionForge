"""What the boxes of a YOLO detection dataset look like (ADR-233).

The integrity scan already opens every image it reads to learn its size; this
module takes that size and the image's label file and tallies, per box:

* how many boxes each image holds, and which images hold none (no label file,
  or an empty one);
* the box size by COCO's rule, on the area in pixels: small below 32², medium
  below 96², large above. These are the buckets ``mAP_small``, ``mAP_medium``
  and ``mAP_large`` use, so the counts say which of those numbers has data;
* the aspect ratio in pixels (width over height) and the position of the box
  centers on a 10×10 grid of the image;
* which classes appear together in the same image;
* boxes the trainer would read wrong: a row that is not five numbers, a class id
  outside ``data.yaml``, zero or negative width or height, or a box that runs
  past the image edge.

A row is judged with the same five-field rule the trainer reads labels with;
nothing here writes to the dataset.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

# COCO's area thresholds, in squared pixels.
SMALL_AREA = 32 * 32
MEDIUM_AREA = 96 * 96
GRID = 10
# A box may poke past the edge by this fraction of the image (rounding of the
# tool that wrote it) before it counts as outside.
EDGE_TOLERANCE = 0.01
MAX_EXAMPLES = 200
TOP_PAIRS = 10

# Boxes per image: 0, 1, 2-3, 4-7, 8-15, 16-31, 32+.
_COUNT_EDGES = (0, 1, 2, 4, 8, 16, 32, math.inf)
# Width over height, in pixels.
_ASPECT_EDGES = (0.0, 0.25, 0.5, 0.8, 1.25, 2.0, 4.0, math.inf)


def _bucket(value: float, edges: tuple[float, ...]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return len(edges) - 2


def _bins(edges: tuple[float, ...], counts: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "lo": edges[i],
            "hi": None if math.isinf(edges[i + 1]) else edges[i + 1],
            "count": n,
        }
        for i, n in enumerate(counts)
    ]


def coco_size(area: float) -> str:
    """COCO's size bucket of a box from its area in squared pixels."""
    if area < SMALL_AREA:
        return "small"
    if area < MEDIUM_AREA:
        return "medium"
    return "large"


def parse_row(line: str) -> tuple[int, float, float, float, float] | None:
    """A YOLO row as (class, x, y, w, h), or None if it is not five numbers."""
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    if not all(math.isfinite(v) for v in values) or values[0] != int(values[0]):
        return None
    return int(values[0]), values[1], values[2], values[3], values[4]


def box_problem(
    row: tuple[int, float, float, float, float], n_classes: int
) -> str | None:
    """Why a parsed row is not a usable box, or None if it is one."""
    cls, x, y, w, h = row
    if cls < 0 or (n_classes and cls >= n_classes):
        return "class_out_of_range"
    if w <= 0 or h <= 0:
        return "zero_area"
    tol = EDGE_TOLERANCE
    if (
        x - w / 2 < -tol
        or y - h / 2 < -tol
        or x + w / 2 > 1 + tol
        or y + h / 2 > 1 + tol
    ):
        return "outside_image"
    return None


@dataclass
class BoxTally:
    """Counts gathered one image at a time; :meth:`report` turns them into JSON."""

    class_names: list[str]
    # Classes data.yaml / classes.txt declares; 0 when it declares none, and then
    # no id is out of range. The explorer pads ``class_names`` past it.
    declared: int = 0
    images: int = 0
    boxes: int = 0
    no_label_file: int = 0
    empty_label: int = 0
    per_image: list[int] = field(default_factory=lambda: [0] * (len(_COUNT_EDGES) - 1))
    max_per_image: int = 0
    sizes: Counter[str] = field(default_factory=Counter)
    class_sizes: dict[str, Counter[str]] = field(default_factory=dict)
    aspect: list[int] = field(default_factory=lambda: [0] * (len(_ASPECT_EDGES) - 1))
    heatmap: list[list[int]] = field(
        default_factory=lambda: [[0] * GRID for _ in range(GRID)]
    )
    pairs: Counter[tuple[str, str]] = field(default_factory=Counter)
    invalid: Counter[str] = field(default_factory=Counter)
    examples: list[dict[str, Any]] = field(default_factory=list)
    unsized: int = 0

    def add(
        self, split: str, rel: str, label: Path | None, size: tuple[int, int] | None
    ) -> None:
        """Tally the boxes of one image; ``size`` is (width, height) if it opened."""
        self.images += 1
        if label is None or not label.is_file():
            self.no_label_file += 1
            self.per_image[0] += 1
            return
        try:
            lines = label.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            lines = []
        held: set[str] = set()
        count = 0
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            name = self._box(split, rel, number, line, size)
            if name is not None:
                count += 1
                held.add(name)
        if count == 0:
            self.empty_label += 1
        self.boxes += count
        self.per_image[_bucket(count, _COUNT_EDGES)] += 1
        self.max_per_image = max(self.max_per_image, count)
        self.pairs.update(combinations(sorted(held), 2))

    def _box(
        self,
        split: str,
        rel: str,
        number: int,
        line: str,
        size: tuple[int, int] | None,
    ) -> str | None:
        """Count one row; the class name of a valid box, None otherwise."""
        row = parse_row(line)
        reason = "malformed" if row is None else box_problem(row, self.declared)
        if reason is not None:
            self.invalid[reason] += 1
            if len(self.examples) < MAX_EXAMPLES:
                self.examples.append(
                    {"split": split, "path": rel, "line": number, "reason": reason}
                )
            return None
        assert row is not None
        cls, x, y, w, h = row
        name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
        col = min(GRID - 1, max(0, int(x * GRID)))
        cell_row = min(GRID - 1, max(0, int(y * GRID)))
        self.heatmap[cell_row][col] += 1
        if size is None:
            self.unsized += 1
            return name
        width, height = w * size[0], h * size[1]
        bucket = coco_size(width * height)
        self.sizes[bucket] += 1
        self.class_sizes.setdefault(name, Counter())[bucket] += 1
        self.aspect[_bucket(width / height, _ASPECT_EDGES)] += 1
        return name

    def report(self) -> dict[str, Any]:
        """The tallies as the scan report's ``boxes`` field."""
        invalid_total = sum(self.invalid.values())
        return {
            "images": self.images,
            "boxes": self.boxes,
            "mean_per_image": self.boxes / self.images if self.images else 0.0,
            "max_per_image": self.max_per_image,
            "per_image_bins": _bins(_COUNT_EDGES, self.per_image),
            "no_label_file": self.no_label_file,
            "empty_label": self.empty_label,
            "size_thresholds": [SMALL_AREA, MEDIUM_AREA],
            "sizes": {k: self.sizes.get(k, 0) for k in ("small", "medium", "large")},
            "class_sizes": [
                {"name": name, **{k: c.get(k, 0) for k in ("small", "medium", "large")}}
                for name, c in sorted(self.class_sizes.items())
            ],
            "unsized": self.unsized,
            "aspect_bins": _bins(_ASPECT_EDGES, self.aspect),
            "heatmap": self.heatmap,
            "pairs": [
                {"a": a, "b": b, "images": n}
                for (a, b), n in self.pairs.most_common(TOP_PAIRS)
            ],
            "invalid_count": invalid_total,
            "invalid": dict(sorted(self.invalid.items())),
            "invalid_examples": self.examples,
        }
