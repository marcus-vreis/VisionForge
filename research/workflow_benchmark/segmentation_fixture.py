"""Synthetic segmentation splits for the parity tests; not a benchmark dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .native_segmentation import build_tiny_fcn
from .trimap_labels import convert_trimap_file

SPLIT_SIZES = {"train": 4, "val": 2, "test": 2}


def make_segmentation_fixture(
    root: Path, *, num_classes: int = 2, split_sizes: dict[str, int] | None = None
) -> tuple[Path, Path]:
    """Write trimap-derived masks, images and shared initial weights under ``root``.

    Images are 40x36 RGB so the 32px recipe really resizes; masks come from
    synthetic Oxford-style trimaps through the production conversion.
    """
    dataset = root / "dataset"
    trimaps = root / "trimaps"
    trimaps.mkdir(parents=True)
    for role, (split, count) in enumerate((split_sizes or SPLIT_SIZES).items()):
        (dataset / split / "images").mkdir(parents=True)
        (dataset / split / "masks").mkdir(parents=True)
        for index in range(count):
            rng = np.random.default_rng(1000 * role + index)
            trimap = rng.choice(
                np.array([1, 2, 3], dtype=np.uint8), size=(36, 40), p=[0.4, 0.4, 0.2]
            )
            # Brighter pixels where the trimap says pet, so the task is learnable.
            pixels = rng.integers(0, 128, (36, 40, 3), dtype=np.uint8)
            pixels[trimap == 1] += 100
            stem = f"{split}_{index:02d}"
            Image.fromarray(pixels).save(dataset / split / "images" / f"{stem}.png")
            source = trimaps / f"{stem}.png"
            Image.fromarray(trimap).save(source)
            convert_trimap_file(source, dataset / split / "masks" / f"{stem}.png")
    torch.random.default_generator.manual_seed(5)
    initial = root / "initial.pth"
    torch.save(build_tiny_fcn(num_classes).state_dict(), initial)
    return dataset, initial
