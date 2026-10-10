"""The synthetic sample datasets the "Primeiro treino" guides train on (ADR-115, ADR-117).

They reuse the self-test builders (``utils/selftest_data.py``): offline,
deterministic, a few seconds of CPU per run. Each folder lands in
``datasets/exemplo-<tarefa>/`` under the GUI's working folder, so the researcher
can open it in the file manager and see what that task's dataset looks like on
disk: class folders for classification, ``images/`` + ``labels/`` for detection,
a CSV manifest for regression, image and mask pairs for segmentation, normal
images only in ``train/`` for anomaly detection.

The builder runs in a scratch folder next to the target and is renamed into
place, so a crash (disk full, process killed) never leaves a half-written
``exemplo-<tarefa>/`` that the next call would take for a finished one.
"""

from __future__ import annotations

import csv
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from visionforge.utils.messages import tr
from visionforge.utils.selftest_data import (
    build_anomaly_dataset,
    build_classification_dataset,
    build_detection_dataset,
    build_regression_dataset,
    build_segmentation_dataset,
)

# The folder of each task's sample. Portuguese, like the rest of the pages the
# researcher meets, so a task is never named twice in two languages on disk.
SAMPLE_DIR_NAMES = {
    "classification": "exemplo-classificacao",
    "detection": "exemplo-deteccao",
    "regression": "exemplo-regressao",
    "segmentation": "exemplo-segmentacao",
    "anomaly": "exemplo-anomalia",
}

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

# The one class of the detection sample. The self-test builder writes label
# lines with class id 0 and no names; a ``classes.txt`` beside them is the plain
# way a YOLO dataset says what id 0 is, and the panels read it.
_DETECTION_CLASS = "object"


class SampleTaskError(ValueError):
    """The task has no sample dataset."""


class SampleDatasetExistsError(FileExistsError):
    """The target folder is in use; ``path`` is where it is, so it can be reused."""

    def __init__(self, path: Path) -> None:
        super().__init__(tr("sample.exists", path=path))
        self.path = path


@dataclass(frozen=True)
class SampleDataset:
    """What was built: where, and what is in it.

    ``counts`` is the number of items per split, in the unit that task reads:
    images for classification and detection, CSV rows for regression, image and
    mask pairs for segmentation, normal training images and labelled test images
    for anomaly detection. ``classes`` is what the layout declares: class folders
    (classification), the names in ``classes.txt`` (detection), the pixel values
    found in the masks (segmentation, ``"0"`` is background) or the subfolders of
    ``test/`` (anomaly); regression has none. ``targets`` is the CSV's target
    columns, for regression only.
    """

    task: str
    path: Path
    classes: list[str]
    counts: dict[str, int]
    targets: list[str] = field(default_factory=list)


def _count_images(folder: Path, *, recursive: bool = False) -> int:
    if not folder.is_dir():
        return 0
    files = folder.rglob("*") if recursive else folder.iterdir()
    return sum(1 for f in files if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)


def _describe_classification(path: Path) -> SampleDataset:
    classes = sorted(p.name for p in (path / "train").iterdir() if p.is_dir())
    counts = {
        split: _count_images(path / split, recursive=True)
        for split in ("train", "val", "test")
    }
    return SampleDataset("classification", path, classes, counts)


def _describe_detection(path: Path) -> SampleDataset:
    names = (path / "classes.txt").read_text(encoding="utf-8").splitlines()
    counts = {
        split: _count_images(path / "images" / split) for split in ("train", "val")
    }
    return SampleDataset(
        "detection", path, [n.strip() for n in names if n.strip()], counts
    )


def _describe_regression(path: Path) -> SampleDataset:
    counts: dict[str, int] = {}
    targets: list[str] = []
    for split in ("train", "val", "test"):
        with (path / f"{split}.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        counts[split] = max(len(rows) - 1, 0)
        if split == "train" and rows:
            targets = [c for c in rows[0] if c != "image"]
    return SampleDataset("regression", path, [], counts, targets)


def _describe_segmentation(path: Path) -> SampleDataset:
    counts = {
        split: _count_images(path / split / "images")
        for split in ("train", "val", "test")
    }
    values: set[int] = set()
    for mask in sorted((path / "train" / "masks").glob("*.png")):
        with Image.open(mask) as img:
            values.update(int(v) for v in np.unique(np.asarray(img)))
    return SampleDataset("segmentation", path, [str(v) for v in sorted(values)], counts)


def _describe_anomaly(path: Path) -> SampleDataset:
    counts = {
        "train": _count_images(path / "train", recursive=True),
        "test": _count_images(path / "test", recursive=True),
    }
    classes = sorted(p.name for p in (path / "test").iterdir() if p.is_dir())
    return SampleDataset("anomaly", path, classes, counts)


_DESCRIBERS: dict[str, Callable[[Path], SampleDataset]] = {
    "classification": _describe_classification,
    "detection": _describe_detection,
    "regression": _describe_regression,
    "segmentation": _describe_segmentation,
    "anomaly": _describe_anomaly,
}


def _build(task: str, scratch: Path) -> None:
    """Write ``task``'s layout into ``scratch``.

    The builders are looked up here, at call time, rather than stored in a table:
    a test that replaces one of them must see its replacement.
    """
    if task == "classification":
        build_classification_dataset(scratch)
    elif task == "detection":
        build_detection_dataset(scratch)
        (scratch / "classes.txt").write_text(f"{_DETECTION_CLASS}\n", encoding="utf-8")
    elif task == "regression":
        build_regression_dataset(scratch)
    elif task == "segmentation":
        build_segmentation_dataset(scratch)
    else:
        build_anomaly_dataset(scratch)


def create_sample_dataset(datasets_dir: Path, task: str) -> SampleDataset:
    """Build ``<datasets_dir>/exemplo-<tarefa>/`` and describe it.

    Raises:
        SampleTaskError: the task is not one of the five built-in tasks.
        SampleDatasetExistsError: the folder exists and is not empty (nothing is
            touched: the researcher may have put their own files there).
    """
    name = SAMPLE_DIR_NAMES.get(task)
    if name is None:
        raise SampleTaskError(tr("sample.unknown_task", task=task))

    target = (datasets_dir / name).resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise SampleDatasetExistsError(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.parent / f".{name}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        _build(task, scratch)
        if target.exists():  # empty folder left by hand: take its place
            target.rmdir()
        scratch.rename(target)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return _DESCRIBERS[task](target)
