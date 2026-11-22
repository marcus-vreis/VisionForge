"""Brightness, contrast and sharpness outliers (ADR-238)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visionforge.core.image_quality import (
    MEASURE_SIDE,
    ImageQuality,
    QualityTally,
    measure_quality,
)


def _save(path: Path, pixels: np.ndarray, mode: str = "L") -> Path:
    Image.fromarray(pixels.astype(np.uint8), mode=mode).save(path)
    return path


class TestMeasure:
    def test_flat_image(self, tmp_path: Path) -> None:
        q = measure_quality(_save(tmp_path / "g.png", np.full((8, 8), 100)))
        assert q == ImageQuality(brightness=100.0, contrast=0.0, sharpness=0.0)

    def test_half_black_half_white(self, tmp_path: Path) -> None:
        pixels = np.zeros((4, 4))
        pixels[:, 2:] = 200
        q = measure_quality(_save(tmp_path / "h.png", pixels))
        assert q.brightness == 100.0
        assert q.contrast == 100.0
        # Interior 2x2: columns 1 and 2. Column 1 (value 0) has a right neighbour
        # of 200, so the Laplacian is -200; column 2 (200) has a left neighbour of
        # 0, so it is +200. Mean 0, variance 200².
        assert q.sharpness == pytest.approx(40_000.0)

    def test_single_bright_pixel(self, tmp_path: Path) -> None:
        pixels = np.zeros((3, 3))
        pixels[1, 1] = 10
        q = measure_quality(_save(tmp_path / "p.png", pixels))
        # One interior pixel: Laplacian 4 * 10, a single value, variance 0.
        assert q.sharpness == 0.0
        assert q.brightness == pytest.approx(10 / 9)

    def test_colour_is_read_as_luma(self, tmp_path: Path) -> None:
        red = np.zeros((4, 4, 3))
        red[..., 0] = 255
        q = measure_quality(_save(tmp_path / "r.png", red, mode="RGB"))
        # PIL's L = R * 299/1000 + ..., rounded: 255 * 0.299 = 76.2 -> 76.
        assert q.brightness == 76.0

    def test_large_image_is_shrunk(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        big = rng.integers(0, 256, (MEASURE_SIDE * 4, MEASURE_SIDE * 2))
        q = measure_quality(_save(tmp_path / "b.png", big))
        # Averaging 4x4 blocks of noise cuts the spread well below the full-size
        # std of about 74 (uniform 0..255).
        assert q.contrast < 40
        assert math.isfinite(q.sharpness)

    def test_undecodable_file_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "x.png"
        bad.write_bytes(b"not an image")
        with pytest.raises(OSError):
            measure_quality(bad)


def _tally(values: list[tuple[str, str, float, float, float]]) -> QualityTally:
    tally = QualityTally()
    for split, rel, b, c, s in values:
        tally.add(split, rel, ImageQuality(b, c, s))
    return tally


class TestReport:
    def test_empty(self) -> None:
        assert QualityTally().report() == {"count": 0}

    def test_dark_and_blurry_outliers(self) -> None:
        # Eight ordinary images: brightness 100..135 by 5, contrast 50, sharpness
        # e^5 - 1 (log1p = 5). Then one very dark and one very blurry image.
        rows = [
            ("train", f"n{i}.png", 100.0 + 5 * i, 50.0, math.expm1(5.0))
            for i in range(8)
        ]
        rows.append(("val", "dark.png", 10.0, 50.0, math.expm1(5.0)))
        rows.append(("test", "blur.png", 120.0, 50.0, math.expm1(1.0)))
        report = _tally(rows).report()
        b = report["metrics"]["brightness"]
        # Ten values sorted: 10, 100, 105, 110, 115, 120, 120, 125, 130, 135.
        # Inclusive quartiles: q1 at position 2.25 -> 106.25, q3 at 6.75 -> 123.75.
        assert (b["q1"], b["q3"]) == (106.25, 123.75)
        # IQR 17.5; fences 106.25 - 26.25 = 80, 123.75 + 26.25 = 150.
        assert b["fences"] == [80.0, 150.0]
        assert b["median_by_split"] == {"test": 120.0, "train": 117.5, "val": 10.0}
        assert report["outlier_count"] == {
            "dark": 1,
            "bright": 0,
            "flat": 0,
            "blurry": 1,
        }
        assert report["outliers"]["dark"] == [
            {"split": "val", "path": "dark.png", "value": 10.0}
        ]
        assert report["outliers"]["blurry"][0]["path"] == "blur.png"
        # Sharpness quartiles on log1p: all 5 except one 1 -> IQR 0, floored to
        # 0.5; low fence 5 - 0.75 = 4.25, shown as e^4.25 - 1.
        assert report["metrics"]["sharpness"]["fences"][0] == round(math.expm1(4.25), 2)

    def test_spread_floor_spares_near_identical_images(self) -> None:
        # Brightness 100 everywhere but one at 105: the IQR is 0, floored to 8,
        # so the fences are 88 and 112 and nothing is flagged.
        rows = [("train", f"n{i}.png", 100.0, 50.0, 200.0) for i in range(9)]
        rows.append(("train", "slightly.png", 105.0, 50.0, 200.0))
        report = _tally(rows).report()
        assert report["metrics"]["brightness"]["fences"] == [88.0, 112.0]
        assert sum(report["outlier_count"].values()) == 0

    def test_most_extreme_first_and_high_side(self) -> None:
        rows = [("train", f"n{i}.png", 100.0, 50.0, 200.0) for i in range(8)]
        rows += [
            ("train", "glare.png", 200.0, 50.0, 200.0),
            ("train", "white.png", 250.0, 50.0, 200.0),
            ("train", "flat.png", 100.0, 1.0, 200.0),
        ]
        report = _tally(rows).report()
        assert [o["path"] for o in report["outliers"]["bright"]] == [
            "white.png",
            "glare.png",
        ]
        assert report["outliers"]["flat"][0]["path"] == "flat.png"
