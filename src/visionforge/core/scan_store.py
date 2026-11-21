"""The last integrity scan of each dataset, kept on disk (ADR-231).

A scan opens every image and can take minutes, so the Laboratory keeps the last
finished report of each dataset and shows it when the dataset is opened again.
Next to the report goes a signature of the dataset as the scan saw it: the path,
labels, size and modification time of every image the index lists. When the
signature differs the report is still shown, marked as stale, because what it
found may no longer be there.

The store is a folder of JSON files named by a hash of the dataset's location;
nothing is written inside the dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visionforge.core.dataset_explorer import DatasetIndex

_FORMAT = 1


def scan_key(index: DatasetIndex, split_dirs: Mapping[str, str]) -> str:
    """Which dataset a report belongs to: its root, task and split folders."""
    parts = [str(index.root.resolve()), index.task]
    parts += [f"{split}={folder}" for split, folder in sorted(split_dirs.items())]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def scan_signature(index: DatasetIndex) -> str:
    """A hash of every listed image's path, labels, size and modification time.

    An image that cannot be read is hashed with no size: the scan reports it
    as corrupted, and it changing is a change.
    """
    digest = hashlib.sha256(index.task.encode("utf-8"))
    root = index.root
    for image in sorted(index.images, key=lambda im: (im.split, im.rel)):
        try:
            stat = os.stat(root / image.rel)
            sized = f"{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            sized = "-"
        line = "\t".join((image.split, image.rel, ",".join(image.classes), sized))
        digest.update(line.encode("utf-8") + b"\n")
    return digest.hexdigest()


def save_scan(
    directory: Path, key: str, signature: str, report: Mapping[str, Any]
) -> None:
    """Keep ``report`` as the last scan of ``key``, replacing the earlier one."""
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "format": _FORMAT,
        "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "signature": signature,
        "report": dict(report),
    }
    target = directory / f"{key}.json"
    # Write then rename, so a reader never sees half a report.
    partial = target.with_suffix(".json.tmp")
    partial.write_text(json.dumps(record), encoding="utf-8")
    os.replace(partial, target)


def load_scan(directory: Path, key: str, signature: str) -> dict[str, Any] | None:
    """The kept report of ``key`` with when it was saved and whether it is stale.

    None when there is none, or when the file is unreadable or from another
    format: a cache that cannot be read is simply empty.
    """
    try:
        record = json.loads((directory / f"{key}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("format") != _FORMAT:
        return None
    return {
        "report": record["report"],
        "saved_at": record["saved_at"],
        "stale": record["signature"] != signature,
    }
