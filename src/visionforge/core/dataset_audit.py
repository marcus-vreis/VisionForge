"""Read-only checks for exact image duplicates and subject leakage across splits."""

from __future__ import annotations

import csv
import hashlib
import os
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any

from PIL import Image

from visionforge.utils.messages import tr

_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
}


def _relative_path(value: str) -> Path:
    path = Path(value)
    if (
        not value
        or value.startswith("/")
        or path.is_absolute()
        or PureWindowsPath(value).drive
        or ".." in path.parts
        or "\\" in value
    ):
        raise ValueError(f"Expected a safe relative POSIX path: {value!r}")
    return path


def load_group_manifest(
    root: Path,
    manifest: Path,
    image_paths: Sequence[Path],
    *,
    allow_other_images: bool = False,
) -> list[str]:
    """Read an exact path,group CSV mapping in the supplied image order.

    Raises:
        ValueError: if the manifest or paths escape root, or CSV coverage is not exact.
    """
    root = root.absolute()
    resolved_root = root.resolve(strict=True)
    expected = _expected_manifest_images(root, resolved_root, image_paths)
    manifest_path = (root / manifest).resolve()  # absolute manifests replace root
    if not manifest_path.is_relative_to(resolved_root):
        raise ValueError(tr("audit.manifest_outside_root", path=manifest, root=root))
    groups: dict[str, str] = {}
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["path", "group"]:
            raise ValueError("Group manifest must have exactly the columns path,group")
        for number, row in enumerate(reader, start=2):
            raw_path = row.get("path") or ""
            relative = _relative_path(raw_path).as_posix()
            group = (row.get("group") or "").strip()
            if None in row or not group:
                raise ValueError(f"Invalid group manifest row {number}: group required")
            if relative in groups:
                raise ValueError(f"Duplicate group manifest path: {relative}")
            if relative not in expected and not _allowed_manifest_image(
                root, resolved_root, relative, allow_other_images
            ):
                raise ValueError(f"Unknown group manifest path: {relative}")
            groups[relative] = group
    missing = sorted(expected.keys() - groups.keys())
    if missing:
        raise ValueError(f"Missing group manifest entries: {missing[:5]}")
    return [
        groups[path.absolute().relative_to(root).as_posix()] for path in image_paths
    ]


def _expected_manifest_images(
    root: Path, resolved_root: Path, image_paths: Sequence[Path]
) -> dict[str, Path]:
    """Index expected image paths after checking dataset containment."""
    expected: dict[str, Path] = {}
    for path in image_paths:
        if not path.resolve(strict=True).is_relative_to(resolved_root):
            raise ValueError(f"Image path escapes dataset root: {path}")
        relative = path.absolute().relative_to(root).as_posix()
        expected[relative] = path
    return expected


def _allowed_manifest_image(
    root: Path, resolved_root: Path, relative: str, allow_other_images: bool
) -> bool:
    """Check whether an extra manifest entry names a contained image."""
    candidate = root / relative
    return (
        allow_other_images
        and candidate.resolve().is_relative_to(resolved_root)
        and candidate.is_file()
        and candidate.suffix.lower() in _IMAGE_SUFFIXES
    )


def audit_dataset(
    root: Path,
    splits: Mapping[str, str] | None = None,
    *,
    group_manifest: Path | None = None,
    strict_groups: bool = False,
    max_issues: int = 100,
) -> dict[str, Any]:
    """Return a deterministic JSON report without changing the dataset.

    Raises:
        ValueError: if root, split paths, or a supplied group manifest are invalid.
    """
    root = Path(root).resolve(strict=True)
    _validate_audit_options(root, group_manifest, strict_groups, max_issues)
    split_paths = (
        dict(splits)
        if splits is not None
        else {"train": "train", "val": "val", "test": "test"}
    )
    if not split_paths:
        raise ValueError("At least one split is required")
    counts = dict.fromkeys(split_paths, 0)
    missing: list[str] = []
    present: list[str] = []
    issues: list[dict[str, str]] = []
    unreadable_count = 0
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    images: list[Path] = []
    memberships: list[str] = []

    def record_issue(path: Path, reason: str) -> None:
        """Count every unreadable image while bounding the reported details."""
        nonlocal unreadable_count
        unreadable_count += 1
        if len(issues) < max_issues:
            issues.append({"path": path.relative_to(root).as_posix(), "reason": reason})

    for split, directory in split_paths.items():
        split_root = root / _relative_path(directory)
        if not split_root.resolve().is_relative_to(root):
            record_issue(split_root, "path escapes dataset root")
            continue
        if not split_root.is_dir():
            missing.append(split)
            continue
        present.append(split)

        for path, digest in _walk_split_images(root, split_root, record_issue):
            relative = path.relative_to(root).as_posix()
            counts[split] += 1
            hashes[digest].append((split, relative))
            images.append(path)
            memberships.append(split)
    duplicates = _cross_split_groups(hashes, "sha256")
    overlaps: list[dict[str, Any]] = []
    if group_manifest is not None:
        overlaps = _group_overlaps(root, Path(group_manifest), images, memberships)
    empty = [split for split in present if counts[split] == 0]
    return {
        "ok": not (
            duplicates
            or unreadable_count
            or missing
            or empty
            or (strict_groups and overlaps)
        ),
        "root": str(root),
        "hash_algorithm": "sha256",
        "visual_similarity_checked": False,
        "image_counts": counts,
        "duplicate_groups": duplicates,
        "missing_splits": missing,
        "empty_splits": empty,
        "unreadable_count": unreadable_count,
        "unreadable": issues,
        "group_manifest": str(group_manifest) if group_manifest is not None else None,
        "strict_groups": strict_groups,
        "group_overlaps": overlaps,
    }


def _validate_audit_options(
    root: Path, group_manifest: Path | None, strict_groups: bool, max_issues: int
) -> None:
    """Reject invalid audit inputs before traversing the dataset."""
    if not root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {root}")
    if max_issues < 0:
        raise ValueError("max_issues must be non-negative")
    if strict_groups and group_manifest is None:
        raise ValueError("strict_groups requires a group manifest")


def _prune_linked_directories(
    root: Path,
    current: str,
    directories: list[str],
    record_issue: Callable[[Path, str], None],
) -> None:
    """Keep traversal inside the root without following directory links."""
    # Snapshot the names because rejected directories are removed during traversal.
    for name in tuple(directories):
        path = Path(current) / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            directories.remove(name)
            record_issue(path, "directory link is not followed")


def _verified_image_digest(
    root: Path, path: Path, record_issue: Callable[[Path, str], None]
) -> str | None:
    """Hash a readable contained image and record any verification failure."""
    try:
        if not path.resolve(strict=True).is_relative_to(root):
            record_issue(path, "path escapes dataset root")
            return None
        with path.open("rb") as stream, Image.open(stream) as image:
            image.verify()
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except (
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
    ) as error:
        record_issue(path, type(error).__name__)
        return None


def _walk_split_images(
    root: Path, split_root: Path, record_issue: Callable[[Path, str], None]
) -> Iterator[tuple[Path, str]]:
    """Yield verified image paths and hashes in deterministic traversal order."""

    def walk_error(error: OSError, directory: Path = split_root) -> None:
        """Report traversal failures against their filename or split directory."""
        record_issue(Path(error.filename or directory), type(error).__name__)

    for current, directories, files in os.walk(
        split_root, followlinks=False, onerror=walk_error
    ):
        directories.sort()
        _prune_linked_directories(root, current, directories, record_issue)
        for name in sorted(files):
            path = Path(current) / name
            if path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            digest = _verified_image_digest(root, path, record_issue)
            if digest is not None:
                yield path, digest


def _cross_split_groups(
    grouped: Mapping[str, list[tuple[str, str]]], key: str
) -> list[dict[str, Any]]:
    """Describe groups whose entries belong to more than one split."""
    return [
        {
            key: identifier,
            "splits": sorted({split for split, _ in entries}),
            "paths": sorted(path for _, path in entries),
        }
        for identifier, entries in sorted(grouped.items())
        if len({split for split, _ in entries}) > 1
    ]


def _group_overlaps(
    root: Path, manifest: Path, images: list[Path], memberships: list[str]
) -> list[dict[str, Any]]:
    """Load image groups and describe their overlap across dataset splits."""
    groups = load_group_manifest(root, manifest, images)
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path, split, group in zip(images, memberships, groups, strict=True):
        grouped[group].append((split, path.relative_to(root).as_posix()))
    return _cross_split_groups(grouped, "group")
