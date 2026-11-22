"""Brightness, contrast and sharpness of each image, and the outliers (ADR-238).

Each image is decoded as 8-bit grayscale and shrunk to at most ``MEASURE_SIDE``
pixels on its longest side, so the numbers compare images of different sizes and
a large dataset stays cheap to read:

- brightness: mean gray level, 0 (black) to 255 (white);
- contrast: standard deviation of the gray levels (0 is a flat image);
- sharpness: variance of the 4-neighbour Laplacian, the usual blur measure
  (Pech-Pacheco et al., ICPR 2000). It spans orders of magnitude, so the fences
  are taken on ``log1p`` of it.

An outlier is an image outside Tukey's fences over all scanned images: too dark
or too bright, too flat, or too blurry. A high contrast or sharpness is not a
defect and is not flagged. The interquartile range is floored by ``MIN_SPREAD``,
so a dataset of near-identical images does not flag every small difference.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

MEASURE_SIDE = 256
TUKEY_K = 1.5
# Smallest interquartile range used for the fences, in each metric's own units
# (gray levels, gray levels, log1p of the Laplacian variance).
MIN_SPREAD: Mapping[str, float] = {"brightness": 8.0, "contrast": 4.0, "sharpness": 0.5}
MAX_REPORTED = 50

Metric = Literal["brightness", "contrast", "sharpness"]
METRICS: tuple[Metric, ...] = ("brightness", "contrast", "sharpness")
# Which side of which metric is a problem.
KINDS: tuple[tuple[str, Metric, Literal["low", "high"]], ...] = (
    ("dark", "brightness", "low"),
    ("bright", "brightness", "high"),
    ("flat", "contrast", "low"),
    ("blurry", "sharpness", "low"),
)


@dataclass(frozen=True)
class ImageQuality:
    """The three numbers of one image; see the module docstring for the units."""

    brightness: float
    contrast: float
    sharpness: float


def measure_quality(path: Path) -> ImageQuality:
    """Measure ``path``; raises what PIL raises for a file that does not decode."""
    with Image.open(path) as im:
        # JPEG decodes straight at a reduced scale; other formats ignore this.
        im.draft("L", (MEASURE_SIDE, MEASURE_SIDE))
        gray = im.convert("L")
    gray.thumbnail((MEASURE_SIDE, MEASURE_SIDE))
    pixels = np.asarray(gray, dtype=np.float64)  # [H, W], 0..255
    return ImageQuality(
        brightness=float(pixels.mean()),
        contrast=float(pixels.std()),
        sharpness=_laplacian_variance(pixels),
    )


def _laplacian_variance(pixels: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian of ``pixels`` [H, W]; 0 below 3×3."""
    if pixels.shape[0] < 3 or pixels.shape[1] < 3:
        return 0.0
    centre = pixels[1:-1, 1:-1]
    lap = (
        4 * centre
        - pixels[:-2, 1:-1]
        - pixels[2:, 1:-1]
        - pixels[1:-1, :-2]
        - pixels[1:-1, 2:]
    )
    return float(lap.var())


def _scaled(metric: Metric, value: float) -> float:
    """The value the fences are taken on."""
    return math.log1p(value) if metric == "sharpness" else value


def _unscaled(metric: Metric, value: float) -> float:
    return math.expm1(value) if metric == "sharpness" else value


def _outside(
    metric: Metric, value: float, side: str, fences: tuple[float, float]
) -> bool:
    scaled = _scaled(metric, value)
    return scaled < fences[0] if side == "low" else scaled > fences[1]


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    if len(values) < 2:
        return values[0], values[0], values[0]
    q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, median, q3


@dataclass
class QualityTally:
    """The measures of the scanned images, by split."""

    rows: list[tuple[str, str, ImageQuality]] = field(default_factory=list)

    def add(self, split: str, rel: str, quality: ImageQuality) -> None:
        self.rows.append((split, rel, quality))

    def report(self) -> dict[str, Any]:
        """Per-metric spread, per-split medians and the outliers, most extreme first."""
        if not self.rows:
            return {"count": 0}
        metrics: dict[str, Any] = {}
        fences: dict[Metric, tuple[float, float]] = {}
        for metric in METRICS:
            metrics[metric], fences[metric] = self._metric(metric)
        found: dict[str, list[dict[str, Any]]] = {}
        for kind, metric, side in KINDS:
            hits = [
                (split, rel, getattr(q, metric))
                for split, rel, q in self.rows
                if _outside(metric, getattr(q, metric), side, fences[metric])
            ]
            hits.sort(key=lambda h: h[2], reverse=side == "high")
            found[kind] = [
                {"split": s, "path": r, "value": round(v, 2)} for s, r, v in hits
            ]
        return {
            "count": len(self.rows),
            "tukey_k": TUKEY_K,
            "measure_side": MEASURE_SIDE,
            "metrics": metrics,
            "outlier_count": {kind: len(items) for kind, items in found.items()},
            "outliers": {kind: items[:MAX_REPORTED] for kind, items in found.items()},
        }

    def _metric(self, metric: Metric) -> tuple[dict[str, Any], tuple[float, float]]:
        raw = sorted(getattr(q, metric) for _, _, q in self.rows)
        q1, _, q3 = _quartiles([_scaled(metric, v) for v in raw])
        spread = max(q3 - q1, MIN_SPREAD[metric])
        low, high = q1 - TUKEY_K * spread, q3 + TUKEY_K * spread
        by_split: dict[str, list[float]] = defaultdict(list)
        for split, _, q in self.rows:
            by_split[split].append(getattr(q, metric))
        r1, median, r3 = _quartiles(raw)
        summary = {
            "min": round(raw[0], 2),
            "q1": round(r1, 2),
            "median": round(median, 2),
            "q3": round(r3, 2),
            "max": round(raw[-1], 2),
            "fences": [
                round(_unscaled(metric, low), 2),
                round(_unscaled(metric, high), 2),
            ],
            "median_by_split": {
                split: round(statistics.median(values), 2)
                for split, values in sorted(by_split.items())
            },
        }
        return summary, (low, high)
