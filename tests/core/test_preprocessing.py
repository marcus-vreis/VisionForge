"""Tests for the image preprocessing pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PIL import Image

from visionforge.core.preprocessing import (
    apply_pipeline,
    apply_step,
    available_kinds,
)


def _solid_image(size: int = 32) -> Image.Image:
    arr = np.full((size, size, 3), 128, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


class TestApplyStep:
    def test_gaussian_blur_preserves_shape(self) -> None:
        out = apply_step(_solid_image(), "gaussian_blur", {"radius": 1.5})
        assert out.size == (32, 32)
        assert out.mode == "RGB"

    def test_median_blur_handles_even_size(self) -> None:
        """Even size must be rounded up to odd internally without erroring."""
        out = apply_step(_solid_image(), "median_blur", {"size": 4})
        assert out.size == (32, 32)

    def test_grayscale_returns_rgb_mode(self) -> None:
        out = apply_step(_solid_image(), "grayscale")
        assert out.mode == "RGB"

    def test_equalize_runs(self) -> None:
        # Non-uniform image so equalize has something to do.
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        out = apply_step(img, "equalize")
        assert out.size == (32, 32)

    def test_autocontrast_runs(self) -> None:
        out = apply_step(_solid_image(), "autocontrast", {"cutoff": 2.0})
        assert out.size == (32, 32)

    def test_edges_returns_rgb(self) -> None:
        out = apply_step(_solid_image(), "edges")
        assert out.mode == "RGB"

    @pytest.mark.parametrize("band", ["LL", "LH", "HL", "HH"])
    def test_wavelet_all_bands(self, band: str) -> None:
        out = apply_step(_solid_image(), "wavelet", {"band": band})
        assert out.size == (32, 32)

    def test_wavelet_odd_sized_image(self) -> None:
        """Haar pads odd dimensions internally."""
        img = Image.fromarray(np.full((31, 33, 3), 128, dtype=np.uint8), mode="RGB")
        out = apply_step(img, "wavelet", {"band": "LL"})
        assert out.size == (33, 31)

    def test_unknown_step_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preprocessing step"):
            apply_step(_solid_image(), "nope")


class TestApplyPipeline:
    def test_empty_pipeline_returns_input(self) -> None:
        img = _solid_image()
        empty_steps: list[dict[str, Any]] = []
        final, intermediates = apply_pipeline(img, empty_steps)
        assert intermediates == []
        # Same object reference is fine — caller cannot rely on copy semantics.
        assert final.size == img.size

    def test_chained_steps(self) -> None:
        img = _solid_image()
        steps: list[dict[str, Any]] = [
            {"kind": "gaussian_blur", "radius": 1.0},
            {"kind": "edges"},
            {"kind": "equalize"},
        ]
        final, intermediates = apply_pipeline(img, steps)
        assert len(intermediates) == 3
        assert final.size == img.size
        # Final is the last intermediate (image, not just shape).
        assert np.array_equal(np.asarray(final), np.asarray(intermediates[-1]))


class TestRegistry:
    def test_available_kinds_includes_known_filters(self) -> None:
        kinds = set(available_kinds())
        assert {
            "gaussian_blur",
            "median_blur",
            "edges",
            "grayscale",
            "equalize",
            "wavelet",
        }.issubset(kinds)
