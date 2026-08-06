"""A filtered copy of a dataset, on disk, for backends that own their pipeline.

Ultralytics loads and augments images itself — `model.train(data=data.yaml)`
hands it the whole job — so a preprocessing filter cannot be injected into its
DataLoader without subclassing its internal dataset and pinning the project to
one version of it. Instead the filters run once, the result is written to a
temporary folder, and the `data.yaml` points there.

That is cheaper than it sounds. The on-the-fly path filters **per image, per
epoch**; this filters once. Over 30 epochs with an expensive filter (CLAHE,
bilateral) it is roughly 30x less CPU.

Three things make it safe rather than a trap, and each has a test:

* **The copy is keyed by content**, not by run — a 20-trial sweep over the same
  dataset and the same pipeline materializes once.
* **It is reference-counted and removed when the last user leaves**, including
  when that user leaves by raising. Runs die for real (ADR-081).
* **Labels and every other non-image file are copied byte for byte.** Filtering
  an image without carrying its label produces a silently wrong training run.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image

from visionforge.core.dataset_fingerprint import fingerprint_dataset
from visionforge.core.preprocessing import apply_pipeline

# Written into every materialized folder so a sweep can tell a live copy from
# one a killed process left behind.
_SENTINEL = ".in-use"

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# How many callers are currently inside a `with` for a given folder. Process
# local by design: a folder left behind by a *different* process is exactly what
# `sweep_orphans` is for, and a refcount shared across processes would need a
# lock file whose own staleness is the same problem again.
_USERS: dict[Path, int] = {}


@dataclass(frozen=True)
class MaterializedDataset:
    """Where the training data lives for the duration of a run."""

    path: Path
    #: True when filters were applied; False when `path` is the original.
    filtered: bool

    @staticmethod
    def estimate_bytes(source: Path) -> int:
        """Total size of the source tree, for warning about disk before copying.

        PNG of photographic data commonly runs 5-10x the JPEG it replaces, so the
        caller should present this as a floor rather than a prediction.
        """
        return sum(f.stat().st_size for f in source.rglob("*") if f.is_file())


def cache_key(source: Path, steps: list[dict[str, Any]], image_format: str) -> str:
    """Identity of a filtered copy: the data it came from and what was done to it.

    Uses the dataset fingerprint rather than the path, so re-exporting a dataset
    to the same location produces a different key instead of silently reusing a
    copy of the old contents.
    """
    digest = fingerprint_dataset(source).digest
    pipeline = json.dumps(steps, sort_keys=True, ensure_ascii=False)
    return sha256(f"{digest}|{pipeline}|{image_format}".encode()).hexdigest()[:16]


def _write_filtered(
    src_file: Path, dest_file: Path, steps: list[dict[str, Any]], image_format: str
) -> None:
    """Filter one image into `dest_file`, or copy it verbatim if it is not one.

    The extension changes with the format; the stem does not. YOLO matches a
    label to its image by stem, so `img0.txt` still finds `img0.png`.
    """
    if src_file.suffix.lower() not in _IMAGE_SUFFIXES:
        shutil.copy2(src_file, dest_file)
        return
    with Image.open(src_file) as img:
        final, _ = apply_pipeline(img.convert("RGB"), steps)
        final.save(dest_file.with_suffix(f".{image_format}"))


def _build(
    source: Path, target: Path, steps: list[dict[str, Any]], image_format: str
) -> None:
    """Mirror `source` into `target`, filtering images and copying the rest."""
    for src_file in source.rglob("*"):
        if not src_file.is_file():
            continue
        dest_file = target / src_file.relative_to(source)
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        _write_filtered(src_file, dest_file, steps, image_format)


@contextmanager
def materialize_dataset(
    source: Path,
    steps: list[dict[str, Any]],
    *,
    cache_root: Path,
    image_format: str = "png",
) -> Iterator[MaterializedDataset]:
    """Yield a filtered copy of `source`, removed when the last user leaves.

    An empty pipeline yields the original untouched — there is nothing to apply,
    and copying a dataset to change nothing would be pure cost.

    PNG by default: re-encoding a JPEG dataset as JPEG would stack compression
    loss on top of the filter, and the copy is transient anyway.
    """
    if not steps:
        yield MaterializedDataset(path=source, filtered=False)
        return

    target = cache_root / cache_key(source, steps, image_format)
    if target.exists():
        logger.info("Reusing filtered dataset at {}", target)
    else:
        logger.info(
            "Filtering {} into {} ({:.1f} MB to read)",
            source,
            target,
            MaterializedDataset.estimate_bytes(source) / 1_048_576,
        )
        target.mkdir(parents=True)
        try:
            _build(source, target, steps, image_format)
        except Exception:
            # A half-written copy would be reused on the next run as if complete.
            shutil.rmtree(target, ignore_errors=True)
            raise
        (target / _SENTINEL).write_text("", encoding="utf-8")

    _USERS[target] = _USERS.get(target, 0) + 1
    try:
        yield MaterializedDataset(path=target, filtered=True)
    finally:
        _USERS[target] -= 1
        if _USERS[target] <= 0:
            del _USERS[target]
            shutil.rmtree(target, ignore_errors=True)
            logger.info("Removed filtered dataset at {}", target)


def sweep_orphans(cache_root: Path) -> int:
    """Delete filtered copies no live caller is holding. Returns how many.

    Covers the gap no `finally` can: a process killed outright leaves its copy
    behind, and nothing in that process ever runs again to clean it up. Called
    at GUI startup, where every folder present is by definition from a process
    that is already gone.
    """
    if not cache_root.is_dir():
        return 0
    removed = 0
    for folder in cache_root.iterdir():
        if not folder.is_dir() or folder in _USERS:
            continue
        shutil.rmtree(folder, ignore_errors=True)
        removed += 1
    if removed:
        logger.info("Removed {} orphaned filtered dataset(s)", removed)
    return removed


__all__ = ["MaterializedDataset", "cache_key", "materialize_dataset", "sweep_orphans"]
