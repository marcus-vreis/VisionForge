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

import errno
import json
import os
import shutil
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger
from PIL import Image

from visionforge.core.dataset_fingerprint import fingerprint_dataset
from visionforge.core.preprocessing import apply_pipeline

# Published only after every image and label has been written successfully.
_SENTINEL = ".in-use"
_LEASE_PREFIX = ".lease-"
_RELEASE_GRACE_S = 2.0

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _try_lock(fd: int) -> bool:
    """Acquire one OS lock without waiting; process death releases ownership."""
    try:
        # sys.platform, not os.name: mypy narrows on it, so each branch type-checks
        # only on its own system.
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            return False
        raise
    return True


@contextmanager
def _file_lock(
    path: Path, *, blocking: bool, create: bool = True
) -> Iterator[int | None]:
    """Hold a file lock, yielding None immediately when a try-lock is busy."""
    flags = os.O_RDWR | (os.O_CREAT if create else 0)
    fd = os.open(path, flags, 0o600)
    try:
        # No byte is written: a second opener writing under the first one's
        # byte-range lock would fail on Windows, and locking past EOF is legal.
        while not _try_lock(fd):
            if not blocking:
                yield None
                return
            time.sleep(0.05)
        yield fd
    finally:
        os.close(fd)


def _gate_path(cache_root: Path) -> Path:
    # Never unlink this gate: waiters must all lock the same inode, even after
    # the last dataset is removed. It lives outside the transient cache tree.
    return cache_root.parent / f".{cache_root.name}.materialized.lock"


def _lease_is_free(lease: Path, grace: float) -> bool:
    """Try-lock a lease, polling up to `grace` seconds; never waits forever."""
    deadline = time.monotonic() + grace
    while True:
        with _file_lock(lease, blocking=False, create=False) as locked:
            if locked is not None:
                return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _remove_if_unused(target: Path, grace: float = 0.0) -> bool:
    """Remove a dataset only after every owner's OS lease has been released.

    `grace` covers Windows, which frees a killed process's byte-range locks a
    moment after the process is gone; only the startup sweep pays for it.
    """
    if not target.is_dir() or target.is_symlink():
        return False
    try:
        for lease in target.glob(f"{_LEASE_PREFIX}*"):
            if not _lease_is_free(lease, grace):
                return False
            lease.unlink()
        shutil.rmtree(target)
    except OSError as exc:
        logger.warning("Could not remove filtered dataset at {}: {}", target, exc)
        return False
    return True


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

    cache_root = cache_root.resolve()
    target = cache_root / cache_key(source, steps, image_format)
    cache_root.mkdir(parents=True, exist_ok=True)
    gate = _gate_path(cache_root)
    try:
        with ExitStack() as ownership:
            with _file_lock(gate, blocking=True):
                if (target / _SENTINEL).is_file():
                    logger.info("Reusing filtered dataset at {}", target)
                else:
                    logger.info(
                        "Filtering {} into {} ({:.1f} MB to read)",
                        source,
                        target,
                        MaterializedDataset.estimate_bytes(source) / 1_048_576,
                    )
                    if target.exists():
                        shutil.rmtree(target)
                    target.mkdir()
                    try:
                        _build(source, target, steps, image_format)
                        (target / _SENTINEL).write_text("", encoding="utf-8")
                    except BaseException:
                        shutil.rmtree(target, ignore_errors=True)
                        raise
                lease = target / f"{_LEASE_PREFIX}{uuid4().hex}"
                ownership.enter_context(_file_lock(lease, blocking=True))
            yield MaterializedDataset(path=target, filtered=True)
    finally:
        # A concurrent build may hold the gate for a long time. Leave an orphan
        # for the next sweep instead of delaying teardown behind that build.
        with _file_lock(gate, blocking=False) as locked:
            if locked is not None and _remove_if_unused(target):
                logger.info("Removed filtered dataset at {}", target)


def sweep_orphans(cache_root: Path) -> int:
    """Delete filtered copies no live caller is holding. Returns how many.

    OS leases protect callers in other processes and threads. A busy build or
    cleanup is skipped immediately; killed owners release their locks in the OS.
    """
    if not cache_root.is_dir():
        return 0
    cache_root = cache_root.resolve()
    removed = 0
    try:
        with _file_lock(_gate_path(cache_root), blocking=False) as locked:
            if locked is None:
                return 0
            for folder in cache_root.iterdir():
                removed += _remove_if_unused(folder, grace=_RELEASE_GRACE_S)
    except OSError as exc:
        logger.warning("Could not sweep filtered datasets at {}: {}", cache_root, exc)
    if removed:
        logger.info("Removed {} orphaned filtered dataset(s)", removed)
    return removed


__all__ = ["MaterializedDataset", "cache_key", "materialize_dataset", "sweep_orphans"]
