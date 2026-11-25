"""Server mode: a dataset sent as a .zip, checked before it is unpacked (ADR-246).

A person on a shared server sends their own dataset as one archive. An
archive is a list of names and sizes written by whoever made it, so nothing in
it is believed: before a byte is written the index is read and refused when a
name would leave the target folder (an absolute path, ``..``, a drive), when
an entry is a link or a device, when there are too many entries, when the
unpacked total is too large, or when one entry expands far beyond what a real
image or label file does (a zip bomb). While unpacking, each entry is counted
again, since the sizes in the index are only what the archive claims.

The index itself is measured before :mod:`zipfile` reads it: ``zipfile`` loads
the whole central directory into memory, and an archive that claims millions
of entries would take the server's memory before any check ran.
"""

from __future__ import annotations

import os
import stat
import struct
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from visionforge.gui.disk_quota import size_text
from visionforge.utils.messages import current_lang

# More files than this is not a dataset a browser sends in one archive.
MAX_ENTRIES = 200_000
# The central directory of MAX_ENTRIES ordinary names fits well inside this.
MAX_INDEX_BYTES = 64 * 1024 * 1024
# A real image, mask or label file never expands this many times; a bomb does.
MAX_RATIO = 100
# An entry this small is harmless whatever its ratio (an empty CSV compresses
# to almost nothing); the total cap still counts it.
RATIO_FLOOR = 1024 * 1024
# A folder macOS adds to every archive it makes; it is not part of the dataset.
_MAC_METADATA = "__MACOSX"
_CHUNK = 1024 * 1024
_NAME_SHOWN = 120

_EOCD = struct.Struct("<4s4H2LH")
_EOCD_SIG = b"PK\x05\x06"
_LOCATOR = struct.Struct("<4sLQL")
_LOCATOR_SIG = b"PK\x06\x07"
_EOCD64 = struct.Struct("<4sQ2H2L4Q")
_EOCD64_SIG = b"PK\x06\x06"
_CENTRAL_HEAD = 46
_CENTRAL_SIG = b"PK\x01\x02"
_NAME_LENGTHS = struct.Struct("<3H")
_NAME_LENGTHS_AT = 28
_ZIP64_MARK = 0xFFFFFFFF


class ArchiveError(ValueError):
    """The archive is refused; ``key`` and ``params`` word the reason (``tr``)."""

    def __init__(self, key: str, **params: object) -> None:
        super().__init__(key)
        self.key = key
        self.params = params


@dataclass(frozen=True)
class Member:
    """One entry that will be unpacked, at ``parts`` under the target folder."""

    info: zipfile.ZipInfo
    parts: tuple[str, ...]
    is_dir: bool


@dataclass(frozen=True)
class ArchivePlan:
    """What an archive holds once every check passed."""

    members: tuple[Member, ...] = field(repr=False)
    files: int
    unpacked_bytes: int


def _shown(name: str) -> str:
    return name[:_NAME_SHOWN]


def _index_location(handle: BinaryIO, size: int) -> tuple[int, int]:
    """``(start, length)`` of the archive's central directory, from its end record."""
    tail_len = min(size, _EOCD.size + 0xFFFF)
    handle.seek(size - tail_len)
    tail = handle.read(tail_len)
    at = tail.rfind(_EOCD_SIG)
    if at < 0 or at + _EOCD.size > len(tail):
        raise ArchiveError("upload.not_zip")
    record = _EOCD.unpack_from(tail, at)
    entries, length, offset = record[4], record[5], record[6]
    record_at = size - tail_len + at
    if entries == 0xFFFF or length == _ZIP64_MARK or offset == _ZIP64_MARK:
        locator_at = record_at - _LOCATOR.size
        if locator_at < 0:
            raise ArchiveError("upload.not_zip")
        handle.seek(locator_at)
        locator = _LOCATOR.unpack(handle.read(_LOCATOR.size))
        if locator[0] != _LOCATOR_SIG or locator[2] + _EOCD64.size > size:
            raise ArchiveError("upload.not_zip")
        handle.seek(locator[2])
        record64 = _EOCD64.unpack(handle.read(_EOCD64.size))
        if record64[0] != _EOCD64_SIG:
            raise ArchiveError("upload.not_zip")
        length, record_at = record64[8], locator[2]
    # Like ``zipfile``, the directory ends where the end record starts, which
    # also covers bytes prepended to the archive.
    start = record_at - length
    if start < 0:
        raise ArchiveError("upload.not_zip")
    return start, length


def count_entries(path: Path) -> int:
    """How many entries the archive's index holds, walked without loading it.

    Raises:
        ArchiveError: not a zip, an index larger than ``MAX_INDEX_BYTES`` or
            with more than ``MAX_ENTRIES`` entries.
    """
    size = path.stat().st_size
    with path.open("rb") as handle:
        start, length = _index_location(handle, size)
        if length > MAX_INDEX_BYTES:
            raise ArchiveError("upload.too_many_entries", limit=MAX_ENTRIES)
        handle.seek(start)
        count = 0
        walked = 0
        while walked < length:
            head = handle.read(_CENTRAL_HEAD)
            if len(head) < _CENTRAL_HEAD or head[:4] != _CENTRAL_SIG:
                raise ArchiveError("upload.not_zip")
            skip = sum(_NAME_LENGTHS.unpack_from(head, _NAME_LENGTHS_AT))
            handle.seek(skip, os.SEEK_CUR)
            walked += _CENTRAL_HEAD + skip
            count += 1
            if count > MAX_ENTRIES:
                raise ArchiveError("upload.too_many_entries", limit=MAX_ENTRIES)
    return count


def _parts(name: str) -> tuple[str, ...] | None:
    """The folders and file name of an entry; None for macOS metadata.

    Raises:
        ArchiveError: a name that is absolute, climbs with ``..``, names a
            drive or a stream (``:``), uses ``\\`` or holds an empty segment.
    """
    unsafe = ArchiveError("upload.unsafe_path", name=_shown(name))
    if not name or "\x00" in name or "\\" in name or name.startswith("/"):
        raise unsafe
    parts = tuple(name.rstrip("/").split("/"))
    if any(part in ("", ".", "..") or ":" in part for part in parts):
        raise unsafe
    return None if parts[0] == _MAC_METADATA else parts


def _kind(info: zipfile.ZipInfo) -> bool:
    """True for a folder entry, False for a regular file.

    Raises:
        ArchiveError: a symbolic link, a device or a pipe, or an entry
            protected by a password.
    """
    mode = stat.S_IFMT(info.external_attr >> 16)
    if mode == stat.S_IFLNK:
        raise ArchiveError("upload.symlink", name=_shown(info.filename))
    if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise ArchiveError("upload.special_file", name=_shown(info.filename))
    if info.flag_bits & 0x1:
        raise ArchiveError("upload.encrypted", name=_shown(info.filename))
    return info.is_dir() or mode == stat.S_IFDIR


def _claim(seen: dict[tuple[str, ...], bool], member: Member) -> None:
    """Refuse a name that is both a file and a folder, or a file given twice."""
    for depth in range(1, len(member.parts)):
        prefix = member.parts[:depth]
        if seen.setdefault(prefix, True) is False:
            raise ArchiveError("upload.duplicate", name=_shown(member.info.filename))
    before = seen.get(member.parts)
    if before is not None and not (before and member.is_dir):
        raise ArchiveError("upload.duplicate", name=_shown(member.info.filename))
    seen[member.parts] = member.is_dir


def plan(archive: zipfile.ZipFile, max_unpacked: int) -> ArchivePlan:
    """Check every entry of an open archive; nothing is written.

    Raises:
        ArchiveError: an unsafe name, a link or special file, a password, a
            name given twice, an entry that expands more than ``MAX_RATIO``
            times, an unpacked total above ``max_unpacked``, or no file at all.
    """
    infos = archive.infolist()
    if len(infos) > MAX_ENTRIES:
        raise ArchiveError("upload.too_many_entries", limit=MAX_ENTRIES)
    members: list[Member] = []
    seen: dict[tuple[str, ...], bool] = {}
    total = 0
    for info in infos:
        parts = _parts(info.filename)
        is_dir = _kind(info)
        if parts is None:
            continue
        member = Member(info=info, parts=parts, is_dir=is_dir)
        _claim(seen, member)
        if not is_dir:
            size = info.file_size
            if size > RATIO_FLOOR and size > MAX_RATIO * max(info.compress_size, 1):
                raise ArchiveError(
                    "upload.ratio", name=_shown(info.filename), ratio=MAX_RATIO
                )
            total += size
            if total > max_unpacked:
                limit = size_text(max_unpacked, current_lang())
                raise ArchiveError("upload.too_big_unpacked", limit=limit)
        members.append(member)
    files = sum(1 for member in members if not member.is_dir)
    if files == 0:
        raise ArchiveError("upload.empty")
    return ArchivePlan(members=tuple(members), files=files, unpacked_bytes=total)


def open_archive(path: Path) -> zipfile.ZipFile:
    """``path`` as a zip, once its index was measured (:func:`count_entries`).

    Raises:
        ArchiveError: not a zip, or an index too large to read.
    """
    count_entries(path)
    try:
        return zipfile.ZipFile(path)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, ValueError) as exc:
        raise ArchiveError("upload.not_zip") from exc


def inspect(path: Path, max_unpacked: int) -> ArchivePlan:
    """The plan of the archive at ``path``; see :func:`plan`."""
    with open_archive(path) as archive:
        return plan(archive, max_unpacked)


def _inside(target: Path, root: Path) -> bool:
    resolved = target.resolve()
    return resolved != root and resolved.is_relative_to(root)


def _copy(archive: zipfile.ZipFile, member: Member, target: Path, budget: int) -> int:
    """Write one entry to ``target`` (which must not exist); its bytes written."""
    written = 0
    try:
        with archive.open(member.info) as source, target.open("xb") as out:
            while chunk := source.read(_CHUNK):
                written += len(chunk)
                if written > member.info.file_size or written > budget:
                    raise ArchiveError(
                        "upload.ratio",
                        name=_shown(member.info.filename),
                        ratio=MAX_RATIO,
                    )
                out.write(chunk)
    except FileExistsError as exc:
        # Two names the target's file system sees as one (``A.jpg``, ``a.jpg``).
        raise ArchiveError(
            "upload.duplicate", name=_shown(member.info.filename)
        ) from exc
    except (zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError) as exc:
        raise ArchiveError("upload.corrupt", reason=str(exc)[:_NAME_SHOWN]) from exc
    return written


def extract(path: Path, dest: Path, max_unpacked: int) -> Path:
    """Unpack the archive at ``path`` into ``dest``, a folder that must not exist.

    Returns the dataset's root: ``dest`` itself, or the one folder directly in
    it when the archive holds a single top folder (a zipped dataset folder).

    Raises:
        ArchiveError: any refusal of :func:`plan`, a name that resolves outside
            ``dest``, an entry that unpacks past its declared size, or a
            corrupt entry. ``dest`` is left for the caller to remove.
    """
    dest.mkdir(parents=True)
    root = dest.resolve()
    with open_archive(path) as archive:
        checked = plan(archive, max_unpacked)
        written = 0
        for member in checked.members:
            target = root.joinpath(*member.parts)
            if not _inside(target, root):
                raise ArchiveError(
                    "upload.unsafe_path", name=_shown(member.info.filename)
                )
            if member.is_dir:
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            written += _copy(archive, member, target, max_unpacked - written)
    top = list(root.iterdir())
    return top[0] if len(top) == 1 and top[0].is_dir() else root


__all__ = [
    "MAX_ENTRIES",
    "MAX_RATIO",
    "ArchiveError",
    "ArchivePlan",
    "count_entries",
    "extract",
    "inspect",
]
