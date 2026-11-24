"""Images with a planted brightness, contrast or sharpness problem (ADR-238, ADR-240).

Twenty-four ordinary scenes (noise between 70 and 180 with a few colored squares) and
three that stand out against them: ``dark`` (every level divided by 8), ``flat``
(one gray) and ``blurry`` (a Gaussian blur of radius 4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ORDINARY = [f"n{i}" for i in range(24)]
PLANTED = ("dark", "flat", "blurry")
SIZE = 64


def _scene(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(70, 180, (SIZE, SIZE, 3)).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for _ in range(4):
        x, y = (int(v) for v in rng.integers(0, SIZE - 16, 2))
        color = tuple(int(v) for v in rng.integers(0, 256, 3))
        draw.rectangle([x, y, x + 12, y + 12], fill=color)
    return image


def planted_images() -> dict[str, Image.Image]:
    """Name -> image: the ordinary ones, then the three planted.

    Three in twenty-seven stay well under the quarter of a split at which Tukey's
    quartiles start to include the outliers themselves.
    """
    images = {name: _scene(i) for i, name in enumerate(ORDINARY)}
    images["dark"] = Image.eval(_scene(100), lambda v: v // 8)
    images["flat"] = Image.new("RGB", (SIZE, SIZE), (128, 128, 128))
    images["blurry"] = _scene(101).filter(ImageFilter.GaussianBlur(4))
    return images


def save_planted(folder: Path, suffix: str = ".png") -> list[str]:
    """Write :func:`planted_images` into ``folder``; the file names written."""
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for name, image in planted_images().items():
        image.save(folder / f"{name}{suffix}")
        names.append(f"{name}{suffix}")
    return names


def flagged(quality: dict, kind: str) -> set[str]:
    """The file names (not paths) a quality report puts under ``kind``."""
    return {Path(o["path"]).stem for o in quality["outliers"][kind]}
