"""Building derived datasets: what incorporate (ADR-166) and merge (ADR-167) share.

A derived dataset is never an edit in place. The fingerprint of the original
(ADR-129) is what old ``run.json`` files remember, so changing the folder would
silently change what those runs trained on. The result is a sibling folder built
under a temporary name in the same parent and published with a single rename, so
a cancellation or an error leaves nothing behind and a destination that appears
meanwhile is never replaced or merged into.

This module holds the mechanics (linking files, free file names, the temporary
folder and the atomic publish, the progress counter, the registry, the class-list
files of a YOLO dataset). What goes *into* the folder is each feature's business.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from loguru import logger

from visionforge.core.dataset_fingerprint import fingerprint_dataset
from visionforge.utils.paths import validate_path_component

REGISTRY_NAME = "visionforge_dataset.json"
REGISTRY_SCHEMA = 1
CLASSES_TXT = "classes.txt"
NAMES_TXT = "names.txt"
YAML_NAMES = ("data.yaml", "data.yml")
CLASS_FILES = (CLASSES_TXT, NAMES_TXT, *YAML_NAMES)

_LINK_FALLBACK_ERRNOS = (errno.EXDEV, errno.EPERM, errno.EMLINK, errno.ENOTSUP)
_PROGRESS_STEP = 25
_UTF8 = "utf-8"
_NAMES = "names"


class DerivedError(Exception):
    """A refusal with a code the API turns into a translated message."""

    def __init__(self, code: str, **params: object) -> None:
        super().__init__(code)
        self.code = code
        self.params = params


# ── files ─────────────────────────────────────────────────────────────────────


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def link_or_copy(
    src: Path,
    dst: Path,
    *,
    link: Callable[[Path, Path], None] = os.link,
) -> str:
    """Hardlink ``src`` to ``dst``, or copy it when linking is not possible.

    A hardlink needs the same volume and a file system that supports it; any
    ``OSError`` falls back to a copy. Symbolic links are copied (their target's
    bytes), never linked, so the result does not depend on the original tree.

    Returns:
        ``"link"`` or ``"copy"``.
    """
    if not src.is_symlink():
        try:
            link(src, dst)
            return "link"
        except OSError as exc:
            if exc.errno not in _LINK_FALLBACK_ERRNOS:
                logger.debug("hardlink failed ({}), copying {}", exc, src.name)
    shutil.copy2(src, dst)
    return "copy"


def free_name(taken: set[str], name: str, sha: str, *, with_label: bool) -> str:
    """``name`` if unused (case-insensitively), else ``stem-<sha8>`` with a counter.

    ``with_label``: the stem's ``.txt`` must be free too (YOLO pairs by stem).
    """

    def free(candidate: str) -> bool:
        if candidate.lower() in taken:
            return False
        return not (with_label and f"{Path(candidate).stem.lower()}.txt" in taken)

    if free(name):
        return name
    stem, ext = os.path.splitext(name)
    candidate = f"{stem}-{sha[:8]}{ext}"
    counter = 2
    while not free(candidate):
        candidate = f"{stem}-{sha[:8]}-{counter}{ext}"
        counter += 1
    return candidate


def claim_name(taken: set[str], name: str, sha: str, *, with_label: bool) -> str:
    """:func:`free_name`, and mark the name (and its label's) as used."""
    chosen = free_name(taken, name, sha, with_label=with_label)
    taken.add(chosen.lower())
    if with_label:
        taken.add(f"{Path(chosen).stem.lower()}.txt")
    return chosen


def list_files(root: Path) -> list[str]:
    """Every file under ``root`` as a sorted POSIX path relative to it."""
    found: list[str] = []
    for current, dirs, names in os.walk(root, followlinks=False):
        dirs.sort()
        found.extend(
            (Path(current) / n).relative_to(root).as_posix() for n in sorted(names)
        )
    return found


def remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


class ProgressCounter:
    """Counts finished steps and reports every few of them."""

    def __init__(
        self, total: int, on_progress: Callable[[int, int], None] | None
    ) -> None:
        self.total = total
        self.done = 0
        self._on_progress = on_progress
        if on_progress:
            on_progress(0, total)

    def tick(self) -> None:
        self.done += 1
        if self._on_progress and (
            self.done % _PROGRESS_STEP == 0 or self.done == self.total
        ):
            self._on_progress(self.done, self.total)


# ── the destination ───────────────────────────────────────────────────────────


def prepare_destination(
    parent: Path,
    dest_name: str,
    *,
    error: type[DerivedError],
    bad_name_code: str,
    exists_code: str,
) -> Path:
    """The folder the result will have, once ``dest_name`` is known to be usable.

    Raises:
        DerivedError: (of type ``error``) on an invalid name or an existing folder.
    """
    try:
        validate_path_component(dest_name)
    except ValueError as exc:
        raise error(bad_name_code) from exc
    dest = parent / dest_name
    if dest.exists():
        raise error(exists_code, name=dest_name)
    return dest


def ensure_unchanged(
    root: Path,
    recorded: Mapping[str, Any],
    *,
    error: Callable[[], DerivedError],
) -> None:
    """Raise ``error()`` if ``root`` no longer has the fingerprint ``recorded``."""
    if fingerprint_dataset(root).to_dict().get("digest") != recorded.get("digest"):
        raise error()


def publish[T](
    parent: Path,
    dest: Path,
    write: Callable[[Path], T | None],
    *,
    error: type[DerivedError],
    exists_code: str,
) -> T | None:
    """Run ``write(tmp)`` on a temporary folder, then rename it to ``dest``.

    ``write`` returns ``None`` when it was cancelled. On cancellation or any error
    the temporary folder is removed and nothing is left at the destination.
    """
    tmp = parent / f".{dest.name}.tmp-{uuid.uuid4().hex[:8]}"
    try:
        result = write(tmp)
    except BaseException:
        remove_tree(tmp)
        raise
    if result is None:
        remove_tree(tmp)
        return None
    try:
        # Re-checked: a folder created meanwhile must not be replaced or merged.
        if dest.exists():
            raise error(exists_code, name=dest.name)
        os.rename(tmp, dest)
    except BaseException:
        remove_tree(tmp)
        raise
    return result


# ── registry and class files ──────────────────────────────────────────────────


def write_registry(tmp: Path, registry: Mapping[str, Any]) -> None:
    (tmp / REGISTRY_NAME).write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding=_UTF8,
        newline="\n",
    )


def write_class_files(
    base: Path,
    tmp: Path,
    dest: Path,
    classes: Sequence[str],
    *,
    extra_splits: Mapping[str, str] | None = None,
) -> None:
    """Write the class list wherever ``base`` kept one (``classes.txt`` if nowhere).

    Always written fresh: a hardlink would share the bytes with the original.
    """
    wrote = False
    for name in (CLASSES_TXT, NAMES_TXT):
        if (base / name).is_file():
            (tmp / name).write_text(
                "\n".join(classes) + "\n", encoding=_UTF8, newline="\n"
            )
            wrote = True
    for name in YAML_NAMES:
        if (base / name).is_file():
            write_yaml(
                base / name, tmp / name, classes, base, dest, extra_splits=extra_splits
            )
            wrote = True
    if not wrote:
        (tmp / CLASSES_TXT).write_text(
            "\n".join(classes) + "\n", encoding=_UTF8, newline="\n"
        )


def write_yaml(
    src: Path,
    out: Path,
    classes: Sequence[str],
    base: Path,
    dest: Path,
    *,
    extra_splits: Mapping[str, str] | None = None,
) -> None:
    """``src`` with ``names`` and ``nc`` updated and an absolute ``path`` re-pointed.

    ``extra_splits``: ``{split: images folder}`` for splits the result has and the
    file did not mention; a key the file already has is left alone.
    """
    import yaml

    data = yaml.safe_load(src.read_text(encoding=_UTF8))
    if not isinstance(data, dict):
        data = {}
    if isinstance(data.get(_NAMES), dict):
        data[_NAMES] = dict(enumerate(classes))
    else:
        data[_NAMES] = list(classes)
    if "nc" in data:
        data["nc"] = len(classes)
    for split, folder in (extra_splits or {}).items():
        data.setdefault(split, folder)
    path_value = data.get("path")
    if isinstance(path_value, str) and Path(path_value).is_absolute():
        try:
            if Path(path_value).resolve() == base:
                data["path"] = str(dest)
        except OSError:
            pass
    out.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding=_UTF8,
        newline="\n",
    )


__all__ = [
    "CLASSES_TXT",
    "CLASS_FILES",
    "NAMES_TXT",
    "REGISTRY_NAME",
    "REGISTRY_SCHEMA",
    "YAML_NAMES",
    "DerivedError",
    "ProgressCounter",
    "claim_name",
    "ensure_unchanged",
    "file_sha256",
    "free_name",
    "link_or_copy",
    "list_files",
    "prepare_destination",
    "publish",
    "remove_tree",
    "write_class_files",
    "write_registry",
    "write_yaml",
]
