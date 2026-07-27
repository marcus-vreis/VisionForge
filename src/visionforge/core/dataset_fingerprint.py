"""Dataset fingerprint for run.json (ADR-061).

A comparison is only meaningful if both runs saw the same data. `run.json`
already records the config, the seed and the environment — but a config points
at a *path*, and paths lie: files get added, a split gets re-shuffled, someone
re-exports the dataset between runs. The fingerprint turns "same base_dir" into
a checkable claim.

Two methods, and the run.json says which one produced the digest, because they
guarantee different things:

* ``manifest`` (default) — sha256 over the sorted ``(relative path, size)``
  list. Fast enough to run before every training (it only stats files), and it
  catches added, removed, renamed or resized files. It does **not** catch an
  edit that preserves the byte count.
* ``content`` — sha256 over the file bytes as well. Complete, but reads the
  whole dataset; opt-in for when a claim has to be airtight.

Claiming more than the method delivers would be worse than not fingerprinting
at all, so ``note`` spells out the limitation in the artifact itself.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from loguru import logger

Method = Literal["manifest", "content"]

# Above this the walk is abandoned rather than stalling a training run.
_MAX_FILES = 200_000
_CHUNK = 1024 * 1024

_MANIFEST_NOTE = (
    "paths+sizes only: a modification that preserves file size is not detected"
)
_CONTENT_NOTE = "full content hash"


@dataclass(frozen=True)
class DatasetFingerprint:
    """Digest of a dataset directory, plus what the digest actually covers."""

    digest: str
    method: str  # manifest | content | unavailable
    n_files: int
    total_bytes: int
    root: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form for run.json."""
        return asdict(self)


def _unavailable(root: str, note: str) -> DatasetFingerprint:
    return DatasetFingerprint(
        digest="",
        method="unavailable",
        n_files=0,
        total_bytes=0,
        root=root,
        note=note,
    )


def fingerprint_dataset(
    root: Path | str,
    *,
    method: Method = "manifest",
    max_files: int = _MAX_FILES,
) -> DatasetFingerprint:
    """Digest every file under ``root``, deepest-deterministic and sorted.

    Never raises: a missing directory or an unreadable file yields an
    ``unavailable`` fingerprint with the reason, because a provenance nicety
    must never be the thing that fails a training run.
    """
    root_path = Path(root)
    root_str = str(root_path)
    if not root_path.is_dir():
        return _unavailable(root_str, "base_dir is not a directory")

    try:
        files = sorted(p for p in root_path.rglob("*") if p.is_file())
    except OSError as exc:  # pragma: no cover - permission edge cases
        return _unavailable(root_str, f"could not walk the dataset: {exc}")

    if len(files) > max_files:
        return _unavailable(
            root_str, f"more than {max_files} files: fingerprint skipped for speed"
        )

    hasher = hashlib.sha256()
    total = 0
    try:
        for path in files:
            relative = path.relative_to(root_path).as_posix()
            size = path.stat().st_size
            total += size
            hasher.update(relative.encode("utf-8"))
            hasher.update(str(size).encode("utf-8"))
            if method == "content":
                with path.open("rb") as handle:
                    while chunk := handle.read(_CHUNK):
                        hasher.update(chunk)
    except OSError as exc:
        return _unavailable(root_str, f"could not read the dataset: {exc}")

    return DatasetFingerprint(
        digest=hasher.hexdigest(),
        method=method,
        n_files=len(files),
        total_bytes=total,
        root=root_str,
        note=_MANIFEST_NOTE if method == "manifest" else _CONTENT_NOTE,
    )


def fingerprint_from_config(
    config: Any, *, method: Method = "manifest"
) -> dict[str, Any]:
    """Fingerprint the dataset a task config points at, for run.json.

    Every task config exposes ``data.base_dir``; anything else yields an
    ``unavailable`` entry rather than an exception, so adding this to a
    trainer can never break training.

    It describes *whatever lives under ``data.base_dir``* — nothing more. A
    task that synthesizes its own data (the counting example) treats that path
    as a marker, so its digest covers the working directory rather than a
    dataset: compare digests between runs of the same task, not across tasks.
    """
    base_dir = getattr(getattr(config, "data", None), "base_dir", None)
    if base_dir is None:
        return _unavailable("", "config has no data.base_dir").to_dict()
    try:
        return fingerprint_dataset(base_dir, method=method).to_dict()
    except Exception as exc:  # noqa: BLE001 - provenance must never break a run
        logger.warning("Dataset fingerprint failed: {}", exc)
        return _unavailable(str(base_dir), f"unexpected error: {exc}").to_dict()


def same_dataset(a: dict[str, Any], b: dict[str, Any]) -> bool | None:
    """Whether two run.json fingerprints describe the same data.

    ``None`` when the question cannot be answered — either digest missing, or
    the two used different methods (a manifest digest and a content digest of
    the same data do not match, and pretending otherwise would be a lie).
    """
    if not a.get("digest") or not b.get("digest"):
        return None
    if a.get("method") != b.get("method"):
        return None
    return bool(a["digest"] == b["digest"])


__all__ = [
    "DatasetFingerprint",
    "Method",
    "fingerprint_dataset",
    "fingerprint_from_config",
    "same_dataset",
]
