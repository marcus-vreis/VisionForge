"""The synthetic sample dataset the "Primeiro treino" guide trains on (ADR-115).

It reuses the self-test builders (``utils/selftest_data.py``): offline,
deterministic, a few seconds of CPU per run. The folder lands in
``datasets/exemplo-<task>/`` under the GUI's working folder, so the researcher
can open it in the file manager and see what a classification dataset looks like
on disk.

The builder runs in a scratch folder next to the target and is renamed into
place, so a crash (disk full, process killed) never leaves a half-written
``exemplo-classificacao/`` that the next call would take for a finished one.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from visionforge.utils.selftest_data import build_classification_dataset

SAMPLE_DIR_NAMES = {"classification": "exemplo-classificacao"}

# Tasks the GUI knows about but whose sample is not built yet (ADR-115 ships
# classification only). Kept apart from "unknown" so the message can tell a
# researcher "not yet" from "never heard of it".
KNOWN_TASKS = {"classification", "detection", "regression", "segmentation", "anomaly"}

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
_SPLITS = ("train", "val", "test")


class SampleTaskError(ValueError):
    """The task has no sample dataset (yet, or at all)."""


class SampleDatasetExistsError(FileExistsError):
    """The target folder is in use; ``path`` is where it is, so it can be reused."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"{path} already exists and is not empty; use it as the dataset "
            "or remove it to generate a new one."
        )
        self.path = path


@dataclass(frozen=True)
class SampleDataset:
    """What was built: the absolute path, the class names and images per split."""

    path: Path
    classes: list[str]
    counts: dict[str, int]


def _describe(path: Path) -> SampleDataset:
    classes = sorted(p.name for p in (path / "train").iterdir() if p.is_dir())
    counts = {
        split: sum(
            1
            for f in (path / split).rglob("*")
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        )
        for split in _SPLITS
    }
    return SampleDataset(path=path, classes=classes, counts=counts)


def create_sample_dataset(datasets_dir: Path, task: str) -> SampleDataset:
    """Build ``<datasets_dir>/exemplo-<task>/`` and describe it.

    Raises:
        SampleTaskError: the task is unknown, or its sample is not built yet.
        SampleDatasetExistsError: the folder exists and is not empty (nothing is
            touched: the researcher may have put their own files there).
    """
    if task not in KNOWN_TASKS:
        raise SampleTaskError(f"Unknown task '{task}'.")
    name = SAMPLE_DIR_NAMES.get(task)
    if name is None:
        raise SampleTaskError(
            f"Only classification has a sample dataset so far (asked for '{task}')."
        )

    target = (datasets_dir / name).resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise SampleDatasetExistsError(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.parent / f".{name}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        build_classification_dataset(scratch)
        if target.exists():  # empty folder left by hand: take its place
            target.rmdir()
        scratch.rename(target)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return _describe(target)
