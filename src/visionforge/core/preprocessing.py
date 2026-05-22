"""Image preprocessing pipeline.

Each step takes a PIL Image and returns a PIL Image. Steps run in order before
the standard augmentation/normalization transforms applied by ``DataModule``.

Filters are intentionally implemented with PIL/NumPy only — no opencv or
pywavelets dependency. The wavelet implementation is a Haar decomposition
(2x2 box average + differences) sufficient for academic visualization;
researchers needing tighter wavelet families should adopt pywavelets later.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def _ensure_rgb(img: Image.Image) -> Image.Image:
    """Convert any PIL mode to RGB so downstream tensors have consistent shape."""
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _gaussian_blur(img: Image.Image, params: dict[str, Any]) -> Image.Image:
    radius = float(params.get("radius", 2.0))
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _median_blur(img: Image.Image, params: dict[str, Any]) -> Image.Image:
    size = int(params.get("size", 3))
    if size % 2 == 0:
        size += 1
    return img.filter(ImageFilter.MedianFilter(size=size))


def _unsharp(img: Image.Image, params: dict[str, Any]) -> Image.Image:
    radius = float(params.get("radius", 2.0))
    percent = int(params.get("percent", 150))
    threshold = int(params.get("threshold", 3))
    return img.filter(
        ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
    )


def _edges(img: Image.Image, _params: dict[str, Any]) -> Image.Image:
    """Sobel-style edge detection (PIL FIND_EDGES kernel)."""
    return _ensure_rgb(img.convert("L").filter(ImageFilter.FIND_EDGES))


def _emboss(img: Image.Image, _params: dict[str, Any]) -> Image.Image:
    return img.filter(ImageFilter.EMBOSS)


def _grayscale(img: Image.Image, _params: dict[str, Any]) -> Image.Image:
    """Grayscale but promoted back to RGB so the model input channel count is preserved."""
    return _ensure_rgb(ImageOps.grayscale(img))


def _equalize(img: Image.Image, _params: dict[str, Any]) -> Image.Image:
    """Histogram equalization — simple CLAHE approximation."""
    return ImageOps.equalize(_ensure_rgb(img))


def _autocontrast(img: Image.Image, params: dict[str, Any]) -> Image.Image:
    cutoff = float(params.get("cutoff", 1.0))
    return ImageOps.autocontrast(_ensure_rgb(img), cutoff=cutoff)


def _wavelet_band(img: Image.Image, params: dict[str, Any]) -> Image.Image:
    """Haar wavelet decomposition (1-level), returns the requested sub-band.

    Bands:
    - LL: low-pass approximation (smoothed)
    - LH: horizontal detail
    - HL: vertical detail
    - HH: diagonal detail
    """
    band = str(params.get("band", "LL")).upper()
    arr = np.asarray(_ensure_rgb(img), dtype=np.float32)
    h, w = arr.shape[:2]
    # Pad to even dimensions so the Haar pairs match cleanly.
    if h % 2 == 1:
        arr = np.pad(arr, ((0, 1), (0, 0), (0, 0)), mode="edge")
    if w % 2 == 1:
        arr = np.pad(arr, ((0, 0), (0, 1), (0, 0)), mode="edge")

    a = arr[0::2, 0::2]
    b = arr[0::2, 1::2]
    c = arr[1::2, 0::2]
    d = arr[1::2, 1::2]

    if band == "LL":
        sub = (a + b + c + d) / 4.0
    elif band == "LH":
        sub = (a - b + c - d) / 4.0
    elif band == "HL":
        sub = (a + b - c - d) / 4.0
    else:  # HH
        sub = (a - b - c + d) / 4.0

    if band != "LL":
        # Detail bands are zero-mean; shift + rescale for human visibility.
        sub = sub - sub.min()
        max_v = sub.max() if sub.max() > 0 else 1.0
        sub = (sub / max_v) * 255.0

    out = np.clip(sub, 0, 255).astype(np.uint8)
    # Upsample back to original resolution so the pipeline composes cleanly.
    pil_sub = Image.fromarray(out, mode="RGB")
    return pil_sub.resize(img.size, Image.BILINEAR)


_REGISTRY = {
    "gaussian_blur": _gaussian_blur,
    "median_blur": _median_blur,
    "unsharp": _unsharp,
    "edges": _edges,
    "emboss": _emboss,
    "grayscale": _grayscale,
    "equalize": _equalize,
    "autocontrast": _autocontrast,
    "wavelet": _wavelet_band,
}


def apply_step(
    img: Image.Image, kind: str, params: dict[str, Any] | None = None
) -> Image.Image:
    """Apply a single preprocessing step by ``kind`` name.

    Raises:
        ValueError: if ``kind`` is not a registered filter.
    """
    fn = _REGISTRY.get(kind)
    if fn is None:
        raise ValueError(f"Unknown preprocessing step: {kind}")
    return fn(img, params or {})


def apply_pipeline(
    img: Image.Image, steps: list[dict[str, Any]]
) -> tuple[Image.Image, list[Image.Image]]:
    """Run ``img`` through every step. Returns ``(final_image, intermediates)``.

    ``intermediates`` contains the image after each step (not including the
    original). The frontend uses these to render a strip showing the effect of
    every filter in the pipeline.
    """
    intermediates: list[Image.Image] = []
    current = img
    for step in steps:
        kind = step.get("kind")
        params = {k: v for k, v in step.items() if k != "kind"}
        current = apply_step(current, str(kind), params)
        intermediates.append(current)
    return current, intermediates


def available_kinds() -> list[str]:
    """Names of every registered preprocessing step."""
    return sorted(_REGISTRY.keys())


__all__ = ["apply_step", "apply_pipeline", "available_kinds"]
