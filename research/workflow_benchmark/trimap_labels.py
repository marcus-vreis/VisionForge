"""Oxford-IIIT Pet trimap to class-id mask conversion (ADR-149); local files only."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

IGNORE_INDEX = 255
# Authors' trimap legend: 1 pet, 2 background, 3 border/not classified.
TRIMAP_TO_MASK = {1: 1, 2: 0, 3: IGNORE_INDEX}


def trimap_to_mask(trimap: np.ndarray) -> np.ndarray:
    """Map trimap values to uint8 ids 0 background, 1 pet, 255 ignored.

    Raises:
        ValueError: a value outside the trimap legend.
    """
    unknown = sorted(set(np.unique(trimap).tolist()) - set(TRIMAP_TO_MASK))
    if unknown:
        raise ValueError(
            f"trimap values {unknown} are outside the legend {sorted(TRIMAP_TO_MASK)}"
        )
    mask = np.empty(trimap.shape, dtype=np.uint8)
    for value, label in TRIMAP_TO_MASK.items():
        mask[trimap == value] = label
    return mask


def convert_trimap_file(source: Path, target: Path) -> None:
    """Write the mode-L class-id mask of a single-channel trimap; never overwrite."""
    with Image.open(source) as trimap:
        if trimap.mode not in ("L", "P"):
            raise ValueError(
                f"trimap {source.name} is {trimap.mode}; expected single-channel L or P"
            )
        mask = trimap_to_mask(np.array(trimap))
    buffer = io.BytesIO()
    Image.fromarray(mask).save(buffer, format="PNG")
    with Path(target).open("xb") as stream:
        stream.write(buffer.getvalue())
