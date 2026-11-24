"""What an anomaly-detection dataset holds (ADR-236).

An anomaly dataset is laid out as MVTec AD: ``train/<normal>`` holds the normal
images the model learns from, ``test/<normal>`` normal test images and every
other ``test/<defect>`` folder one kind of defect (ADR-038). Some datasets add
``ground_truth/<defect>/<stem>_mask.png`` with the defect's pixels. This module
opens every image (or an evenly spaced sample) and reports:

* how many images each group holds, and how many defect kinds have so few
  images that their score swings with one image;
* folders under ``train`` other than the normal one, which training ignores;
* sizes and color modes per group: a test group whose size or mode differs from
  training is measured on images the model never saw in that form;
* training images that look like a defect image (a dHash within
  ``near_duplicates.MAX_DISTANCE``), the likeliest sign of an anomalous image in
  the normal training folder, and normal test images that look like a training
  image (the test then measures memory);
* defect images without a ground-truth mask, when ground truth exists.

Nothing here writes to the dataset.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from visionforge.core.cancellation import CancellationToken, is_cancelled
from visionforge.core.image_quality import QualityTally
from visionforge.core.near_duplicates import MAX_DISTANCE, difference_hash

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
DEFAULT_LIMIT = 20_000
MAX_LIMIT = 200_000
MAX_EXAMPLES = 50
# A defect kind with fewer test images than this gives an AUROC that one image
# moves by several points.
FEW_IMAGES = 10
_GROUND_TRUTH = "ground_truth"
# A flat image hashes to 0; it matches every other flat image, not a defect.
_FLAT = 0


@dataclass(frozen=True)
class AnomalyImage:
    group: str  # "train", "test_normal" or "defect:<kind>"
    path: Path


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def list_images(
    root: Path, train_dir: str, test_dir: str, normal_dir: str
) -> tuple[list[AnomalyImage], dict[str, Any]]:
    """Every image by group, and the folder layout around them."""
    train, test = root / train_dir, root / test_dir
    items = [AnomalyImage("train", p) for p in _images(train / normal_dir)]
    items += [AnomalyImage("test_normal", p) for p in _images(test / normal_dir)]
    kinds = (
        sorted(d.name for d in test.iterdir() if d.is_dir() and d.name != normal_dir)
        if test.is_dir()
        else []
    )
    for kind in kinds:
        items += [AnomalyImage(f"defect:{kind}", p) for p in _images(test / kind)]
    extra_train = (
        sorted(d.name for d in train.iterdir() if d.is_dir() and d.name != normal_dir)
        if train.is_dir()
        else []
    )
    layout = {
        "train_present": (train / normal_dir).is_dir(),
        "test_present": test.is_dir(),
        "defect_kinds": kinds,
        "extra_train_folders": extra_train,
        "ground_truth": (root / _GROUND_TRUTH).is_dir(),
    }
    return items, layout


def images_signature(items: list[AnomalyImage]) -> str:
    """A hash of every image's group, path, size and modification time."""
    digest = hashlib.sha256()
    for item in items:
        try:
            stat = os.stat(item.path)
            sized = f"{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            sized = "-"
        digest.update(f"{item.group}\t{item.path.as_posix()}\t{sized}\n".encode())
    return digest.hexdigest()


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def matches(
    left: list[tuple[str, int]], right: list[tuple[str, int]], max_distance: int
) -> list[dict[str, Any]]:
    """Pairs (left, right) whose dHashes differ in at most ``max_distance`` bits.

    Flat images (hash 0) are left out: they match each other, not content.
    """
    left = [(p, h) for p, h in left if h != _FLAT]
    right = [(p, h) for p, h in right if h != _FLAT]
    if not left or not right:
        return []
    b = np.array([h for _, h in right], dtype=np.uint64)
    found: list[dict[str, Any]] = []
    for path, value in left:
        xor = np.bitwise_xor(b, np.uint64(value))
        bits = np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        k = int(bits.argmin())
        if int(bits[k]) <= max_distance:
            found.append({"path": path, "like": right[k][0], "distance": int(bits[k])})
    return found


def analyze_anomaly(
    root: Path,
    *,
    train_dir: str = "train",
    test_dir: str = "test",
    normal_dir: str = "good",
    max_images: int = DEFAULT_LIMIT,
    token: CancellationToken | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Open and hash every image (or a sample) and describe the dataset."""
    root = Path(root).resolve()
    items, layout = list_images(root, train_dir, test_dir, normal_dir)
    counts = Counter(item.group for item in items)
    max_images = max(1, min(max_images, MAX_LIMIT))
    step = max(1, math.ceil(len(items) / max_images)) if items else 1
    chosen = items[::step]
    sizes: dict[str, Counter[str]] = {}
    modes: dict[str, Counter[str]] = {}
    hashes: dict[str, list[tuple[str, int]]] = {}
    unreadable: list[dict[str, str]] = []
    unreadable_count = 0
    quality = QualityTally()
    cancelled = False
    for done, item in enumerate(chosen, start=1):
        if is_cancelled(token):
            cancelled = True
            break
        rel = _rel(root, item.path)
        try:
            with Image.open(item.path) as im:
                size, mode = f"{im.size[0]}×{im.size[1]}", im.mode
            value = difference_hash(item.path)
            # Brightness, contrast and sharpness of an image that hashed (ADR-240).
            measured = quality.measure(item.group, rel, item.path)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
            measured = False
        if not measured:
            unreadable_count += 1
            if len(unreadable) < MAX_EXAMPLES:
                unreadable.append({"group": item.group, "path": rel})
        else:
            sizes.setdefault(item.group, Counter())[size] += 1
            modes.setdefault(item.group, Counter())[mode] += 1
            hashes.setdefault(item.group, []).append((rel, value))
        if on_progress is not None:
            on_progress(done, len(chosen))
    defects = [h for g, hs in hashes.items() if g.startswith("defect:") for h in hs]
    suspicious = matches(hashes.get("train", []), defects, MAX_DISTANCE)
    memorised = matches(
        hashes.get("test_normal", []), hashes.get("train", []), MAX_DISTANCE
    )
    return {
        "task": "anomaly",
        "total_images": len(items),
        "scanned": sum(sum(c.values()) for c in sizes.values()) + unreadable_count,
        "limit": max_images,
        "sampled": step > 1,
        "cancelled": cancelled,
        "max_distance": MAX_DISTANCE,
        "few_images": FEW_IMAGES,
        **layout,
        "counts": dict(sorted(counts.items())),
        "groups": _groups(sizes, modes),
        "size_mismatch": _mismatch(sizes, "train"),
        "mode_mismatch": _mismatch(modes, "train"),
        "train_like_defect_count": len(suspicious),
        "train_like_defect": suspicious[:MAX_EXAMPLES],
        "test_like_train_count": len(memorised),
        "test_like_train": memorised[:MAX_EXAMPLES],
        "missing_masks": _missing_masks(root, items)
        if layout["ground_truth"]
        else None,
        "unreadable_count": unreadable_count,
        "unreadable": unreadable,
        "quality": quality.report(),
    }


def _groups(
    sizes: dict[str, Counter[str]], modes: dict[str, Counter[str]]
) -> list[dict[str, Any]]:
    return [
        {
            "group": group,
            "sizes": [{"size": s, "count": n} for s, n in sizes[group].most_common(5)],
            "distinct_sizes": len(sizes[group]),
            "modes": dict(modes.get(group, Counter())),
        }
        for group in sorted(sizes)
    ]


def _mismatch(by_group: dict[str, Counter[str]], reference: str) -> list[str]:
    """Groups whose most common value differs from the reference group's."""
    ref = by_group.get(reference)
    if not ref:
        return []
    usual = ref.most_common(1)[0][0]
    return [
        g
        for g, c in sorted(by_group.items())
        if g != reference and c.most_common(1)[0][0] != usual
    ]


def _missing_masks(root: Path, items: list[AnomalyImage]) -> dict[str, Any]:
    """Defect images with no ``ground_truth/<kind>/<stem>_mask.*`` (MVTec)."""
    missing: list[str] = []
    for item in items:
        if not item.group.startswith("defect:"):
            continue
        kind = item.group.split(":", 1)[1]
        folder = root / _GROUND_TRUTH / kind
        stems = (f"{item.path.stem}_mask", item.path.stem)
        if not any((folder / f"{s}{e}").is_file() for s in stems for e in IMAGE_EXTS):
            missing.append(_rel(root, item.path))
    return {"count": len(missing), "paths": missing[:MAX_EXAMPLES]}
