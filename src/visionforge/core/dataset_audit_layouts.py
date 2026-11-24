"""Where each task keeps the images of its splits, for the image audit (ADR-230).

The audit walks one folder per split, which is how classification, anomaly and
segmentation (``<split>/<images_subdir>``) lay out their data; the browser
names those folders itself. Two layouts need the dataset read first: a YOLO
root keeps its images in ``images/<split>`` or ``<split>/images``, and a
regression dataset lists its images in one CSV per split.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from visionforge.core.dataset_audit import audit_dataset
from visionforge.core.detection_data import resolve_yolo_split

AuditLayout = Literal["folders", "yolo", "csv"]


def yolo_split_folders(root: Path, splits: Mapping[str, str]) -> dict[str, str]:
    """Each split's image folder under a YOLO root, relative to it.

    A split neither layout has maps to ``images/<split>``, which does not
    exist, so the audit reports it missing.
    """
    folders: dict[str, str] = {}
    for name, split in splits.items():
        found = resolve_yolo_split(root, split)
        images = found[0] if found is not None else root / "images" / split
        folders[name] = images.relative_to(root).as_posix()
    return folders


def csv_split_images(
    root: Path, csvs: Mapping[str, str], images_dir: str, image_column: str
) -> dict[str, list[Path] | None]:
    """The images each split's CSV lists; None for a CSV that does not exist.

    Raises:
        ValueError: a CSV is outside the root or lacks ``image_column``.
    """
    images_root = root / images_dir
    listed: dict[str, list[Path] | None] = {}
    for name, filename in csvs.items():
        path = root / filename
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"CSV outside the dataset root: {filename!r}")
        if not path.is_file():
            listed[name] = None
            continue
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if image_column not in (reader.fieldnames or []):
                raise ValueError(
                    f"{filename} has no column {image_column!r}; "
                    f"columns: {reader.fieldnames}"
                )
            # normpath keeps the path lexical: one that climbs out of the root
            # stays outside it, and the audit reports it instead of reading it.
            listed[name] = [
                Path(os.path.normpath(images_root / cell))
                for row in reader
                if (cell := (row.get(image_column) or "").strip())
            ]
    return listed


def audit_task_dataset(
    root: Path,
    splits: Mapping[str, str] | None,
    layout: AuditLayout,
    *,
    images_dir: str = "images",
    image_column: str = "image",
    **options: Any,
) -> dict[str, Any]:
    """Audit a dataset in the layout of its task (see ``audit_dataset``).

    For ``"csv"`` each split maps to its CSV file; for ``"yolo"`` to the split
    name; for ``"folders"`` to the split's image folder. ``options`` go to
    ``audit_dataset``, ``optional_splits`` among them.

    Raises:
        ValueError: as ``audit_dataset``, or a CSV lacks ``image_column``.
    """
    root = Path(root).resolve(strict=True)
    names = (
        dict(splits)
        if splits is not None
        else {"train": "train", "val": "val", "test": "test"}
    )
    if layout == "csv":
        listed = csv_split_images(root, names, images_dir, image_column)
        return audit_dataset(root, split_images=listed, **options)
    if layout == "yolo":
        return audit_dataset(root, yolo_split_folders(root, names), **options)
    return audit_dataset(root, names, **options)
