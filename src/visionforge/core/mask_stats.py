"""What the masks of a segmentation dataset hold (ADR-234).

A segmentation split is an ``images`` folder and a ``masks`` folder whose files
pair by filename stem; a mask is a single-channel image whose pixel value is the
class id, with ``ignore_index`` for void pixels (ADR-037). This module opens
every pair (or an evenly spaced sample) and reports, per split:

* images without a mask and masks without an image;
* pairs whose mask and image differ in size, and masks in color (RGB), which
  the trainer refuses;
* the share of pixels of each class id, and of ``ignore_index``;
* ids outside ``0..num_classes-1`` that are not ``ignore_index``: the trainer
  would read them as classes the model does not have;
* the coverage of each mask, the share of its pixels that are neither class 0
  nor void (class 0 is background by the usual convention, stated on screen);
* tiny regions: connected areas of one class smaller than ``TINY_SHARE`` of
  their image, which the resize to the training size can erase.

Nothing here writes to the dataset.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from visionforge.core.cancellation import CancellationToken, is_cancelled
from visionforge.core.image_quality import QualityTally

# Same lists as the trainer's reader (core/segmentation_data.py), kept here so
# the analysis does not import torch.
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
MASK_EXTS = (".png", ".bmp", ".tif", ".tiff")
DEFAULT_LIMIT = 5_000
MAX_LIMIT = 200_000
MAX_EXAMPLES = 50
# A region below this share of its image (0.01%: 26 px at 512 × 512) is "tiny".
TINY_SHARE = 1e-4
# Mask coverage bins, as shares of the image: 0, (0, 1%), [1%, 5%), ... [75%, 100%].
_COVERAGE_EDGES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class MaskPair:
    split: str
    image: Path
    mask: Path


def _files(directory: Path, exts: tuple[str, ...]) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {
        p.stem: p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    }


def list_pairs(
    root: Path, split_dirs: Mapping[str, str], images_subdir: str, masks_subdir: str
) -> tuple[list[MaskPair], dict[str, dict[str, Any]]]:
    """Pairs by stem, and per split what is present and what has no partner."""
    pairs: list[MaskPair] = []
    splits: dict[str, dict[str, Any]] = {}
    for split, folder in split_dirs.items():
        base = root / folder
        images = _files(base / images_subdir, IMAGE_EXTS)
        masks = _files(base / masks_subdir, MASK_EXTS)
        splits[split] = {
            "present": base.is_dir(),
            "images": len(images),
            "masks": len(masks),
            "no_mask": sorted(images.keys() - masks.keys()),
            "no_image": sorted(masks.keys() - images.keys()),
        }
        pairs += [
            MaskPair(split, images[s], masks[s])
            for s in sorted(images.keys() & masks.keys())
        ]
    return pairs, splits


def pairs_signature(pairs: list[MaskPair]) -> str:
    """A hash of every pair's paths, sizes and modification times."""
    digest = hashlib.sha256()
    for pair in pairs:
        for path in (pair.image, pair.mask):
            try:
                stat = os.stat(path)
                sized = f"{stat.st_size}:{stat.st_mtime_ns}"
            except OSError:
                sized = "-"
            digest.update(f"{pair.split}\t{path.as_posix()}\t{sized}\n".encode())
    return digest.hexdigest()


@dataclass
class MaskTally:
    """Counts gathered one pair at a time; :meth:`report` turns them into JSON."""

    root: Path
    num_classes: int
    ignore_index: int
    pairs: int = 0
    pixels: dict[str, dict[int, int]] = field(default_factory=dict)
    coverage: list[int] = field(default_factory=lambda: [0] * (len(_COVERAGE_EDGES)))
    out_of_range_pixels: int = 0
    out_of_range_ids: set[int] = field(default_factory=set)
    size_mismatch: int = 0
    color: int = 0
    unreadable: int = 0
    tiny_regions: dict[int, int] = field(default_factory=dict)
    tiny_images: int = 0
    examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # Brightness, contrast and sharpness of the images, not the masks (ADR-240).
    quality: QualityTally = field(default_factory=lambda: QualityTally())

    def _example(self, kind: str, pair: MaskPair, **extra: Any) -> None:
        items = self.examples.setdefault(kind, [])
        if len(items) < MAX_EXAMPLES:
            rel = pair.mask.relative_to(self.root).as_posix()
            items.append({"split": pair.split, "path": rel, **extra})

    def add(self, pair: MaskPair) -> None:
        """Tally one image/mask pair."""
        self.pairs += 1
        try:
            with Image.open(pair.image) as im:
                image_size = im.size
            with Image.open(pair.mask) as mk:
                mode, mask_size = mk.mode, mk.size
                values = None if mode in ("RGB", "RGBA") else np.asarray(mk)
        except (OSError, ValueError, Image.DecompressionBombError):
            self.unreadable += 1
            self._example("unreadable", pair)
            return
        rel_image = pair.image.relative_to(self.root).as_posix()
        if not self.quality.measure(pair.split, rel_image, pair.image):
            self.unreadable += 1
            self._example("unreadable", pair)
            return
        if values is None:
            self.color += 1
            self._example("color", pair, mode=mode)
            return
        if mask_size != image_size:
            self.size_mismatch += 1
            self._example(
                "size_mismatch", pair, mask=list(mask_size), image=list(image_size)
            )
        if values.ndim != 2:
            values = values[..., 0]
        self._pixels(pair, values.astype(np.int64, copy=False))

    def _pixels(self, pair: MaskPair, values: np.ndarray) -> None:
        ids, counts = np.unique(values, return_counts=True)
        split = self.pixels.setdefault(pair.split, {})
        total = int(values.size)
        labelled = 0
        for i, n in zip(ids.tolist(), counts.tolist(), strict=True):
            split[i] = split.get(i, 0) + n
            if i == self.ignore_index:
                continue
            if not 0 <= i < self.num_classes:
                self.out_of_range_pixels += n
                self.out_of_range_ids.add(i)
            elif i != 0:
                labelled += n
        bad = sorted(
            i
            for i in ids.tolist()
            if i != self.ignore_index and not 0 <= i < self.num_classes
        )
        if bad:
            self._example("out_of_range", pair, ids=bad[:10])
        share = labelled / total if total else 0.0
        self.coverage[_coverage_bin(share)] += 1
        self._tiny(pair, values, ids.tolist(), total)

    def _tiny(
        self, pair: MaskPair, values: np.ndarray, ids: list[int], total: int
    ) -> None:
        limit = TINY_SHARE * total
        found = 0
        for i in ids:
            if i in (0, self.ignore_index):
                continue
            labels, n = ndimage.label(values == i)
            if n == 0:
                continue
            areas = np.bincount(labels.ravel())[1:]
            small = int((areas < limit).sum())
            if small:
                self.tiny_regions[i] = self.tiny_regions.get(i, 0) + small
                found += small
        if found:
            self.tiny_images += 1
            self._example("tiny", pair, regions=found)

    def report(self) -> dict[str, Any]:
        """The tallies, with pixel shares per class and split."""
        ids = sorted({i for counts in self.pixels.values() for i in counts})
        split_totals = {s: sum(c.values()) for s, c in self.pixels.items()}
        everything = sum(split_totals.values())
        classes = []
        for i in ids:
            per_split = {s: c.get(i, 0) for s, c in self.pixels.items()}
            pixels = sum(per_split.values())
            classes.append(
                {
                    "id": i,
                    "ignore": i == self.ignore_index,
                    "out_of_range": i != self.ignore_index
                    and not 0 <= i < self.num_classes,
                    "pixels": pixels,
                    "share": 100.0 * pixels / everything if everything else 0.0,
                    "shares": {
                        s: 100.0 * n / split_totals[s] if split_totals[s] else 0.0
                        for s, n in per_split.items()
                    },
                }
            )
        absent = [
            i
            for i in range(self.num_classes)
            if all(i not in c for c in self.pixels.values())
        ]
        return {
            "pairs": self.pairs,
            "classes": classes,
            "absent_classes": absent,
            "coverage_bins": [
                {
                    "lo": 0.0 if k == 0 else _COVERAGE_EDGES[k - 1],
                    "hi": 0.0 if k == 0 else _COVERAGE_EDGES[k],
                    "count": n,
                }
                for k, n in enumerate(self.coverage)
            ],
            "empty_masks": self.coverage[0],
            "out_of_range_pixels": self.out_of_range_pixels,
            "out_of_range_ids": sorted(self.out_of_range_ids),
            "size_mismatch": self.size_mismatch,
            "color_masks": self.color,
            "unreadable": self.unreadable,
            "tiny_share": TINY_SHARE,
            "tiny_regions": [
                {"id": i, "regions": n} for i, n in sorted(self.tiny_regions.items())
            ],
            "tiny_images": self.tiny_images,
            "examples": self.examples,
            "quality": self.quality.report(),
        }


def _coverage_bin(share: float) -> int:
    """Bin 0 is exactly zero; then one bin per pair of ``_COVERAGE_EDGES``."""
    if share <= 0.0:
        return 0
    for k in range(1, len(_COVERAGE_EDGES)):
        if share < _COVERAGE_EDGES[k]:
            return k
    return len(_COVERAGE_EDGES) - 1


def analyze_masks(
    root: Path,
    split_dirs: Mapping[str, str],
    *,
    images_subdir: str = "images",
    masks_subdir: str = "masks",
    num_classes: int,
    ignore_index: int = 255,
    max_images: int = DEFAULT_LIMIT,
    token: CancellationToken | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Pair, open and tally every image/mask pair (or an evenly spaced sample).

    Cancelling returns what was gathered so far with ``cancelled`` true.
    """
    root = Path(root).resolve()
    pairs, splits = list_pairs(root, split_dirs, images_subdir, masks_subdir)
    max_images = max(1, min(max_images, MAX_LIMIT))
    step = max(1, math.ceil(len(pairs) / max_images)) if pairs else 1
    chosen = pairs[::step]
    tally = MaskTally(root, num_classes, ignore_index)
    cancelled = False
    for done, pair in enumerate(chosen, start=1):
        if is_cancelled(token):
            cancelled = True
            break
        tally.add(pair)
        if on_progress is not None:
            on_progress(done, len(chosen))
    for info in splits.values():
        info["no_mask_count"] = len(info["no_mask"])
        info["no_image_count"] = len(info["no_image"])
        info["no_mask"] = info["no_mask"][:MAX_EXAMPLES]
        info["no_image"] = info["no_image"][:MAX_EXAMPLES]
    return {
        "task": "segmentation",
        "num_classes": num_classes,
        "ignore_index": ignore_index,
        "total_pairs": len(pairs),
        "scanned": tally.pairs,
        "limit": max_images,
        "sampled": step > 1,
        "cancelled": cancelled,
        "splits": splits,
        **tally.report(),
    }
