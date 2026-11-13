"""Datasets that live in the user's Google Drive, for Colab runs (ADR-164).

The dataset sits in the local folder of Google Drive for desktop, so the
VisionForge on this machine reads it like any path. A Colab job does not upload
it: the path is translated to the VM's ``/content/drive/MyDrive/...``, the VM
mounts the Drive and copies the folder to its local disk before training.

Three pieces live here, none of them talks to Google:

- finding the "My Drive" folder of this machine and translating a path under it;
- reading the consent URL out of the CLI's ``drivemount`` output;
- ``DriveConsent``, where the job thread waits until the person says, through
  the GUI, that they granted the access.

Facts from the spikes (design doc, 2026-11-05): ``drivemount`` asks for a
browser consent on every new runtime (the credential is ephemeral, per VM). It
prints an ``accounts.google.com`` URL and waits for Enter on stdin. An Enter sent
before the consent makes the CLI print ``Error propagating: 400`` and hang.
"""

from __future__ import annotations

import os
import re
import string
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Literal

# The VM's view of the Drive once mounted at /content/drive.
REMOTE_MY_DRIVE = PurePosixPath("/content/drive/MyDrive")
# The name Drive for desktop gives the user's own folder, by interface language.
# Only English and Portuguese are known; other languages use the manual root.
MY_DRIVE_NAMES = ("My Drive", "Meu Drive")
_CONSENT_URL = re.compile(r"https://accounts\.google\.com/\S+")
_REGISTRY_KEYS = (
    r"Software\Policies\Google\DriveFS",
    r"Software\Google\DriveFS",
)

ConsentWait = Literal["granted", "timeout", "cancelled"]


def consent_url(line: str) -> str | None:
    """The consent page a ``drivemount`` line names, or None."""
    match = _CONSENT_URL.search(line)
    return match.group(0) if match else None


def _same(a: str, b: str) -> bool:
    return a.casefold() == b.casefold()


def _windows(path: str | PurePath) -> PureWindowsPath:
    r"""Drive for desktop paths are Windows paths (``G:\Meu Drive``) whatever
    the OS the server runs on, so they are read with Windows rules everywhere:
    on Linux a plain ``Path`` would take ``G:\`` for a folder name."""
    return PureWindowsPath(str(path))


def _starts_with(parts: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return len(parts) >= len(prefix) and all(
        _same(a, b) for a, b in zip(parts, prefix, strict=False)
    )


def remote_path(
    local: str | PurePath, roots: Iterable[str | PurePath]
) -> PurePosixPath | None:
    r"""Where ``local`` is on the VM, or None when it is not inside the Drive.

    ``roots`` are "My Drive" folders known for sure (the manual one, the one the
    registry names). Without a match, a path component named like "My Drive"
    right under a drive letter (``G:\Meu Drive``) or under the home folder
    (mirror mode) counts as well.
    """
    parts = _windows(local).parts
    for root in roots:
        root_parts = _windows(root).parts
        if _starts_with(parts, root_parts):
            return REMOTE_MY_DRIVE.joinpath(*parts[len(root_parts) :])
    home = _windows(Path.home()).parts
    for i, part in enumerate(parts):
        if not any(_same(part, name) for name in MY_DRIVE_NAMES):
            continue
        parent = parts[:i]
        under_letter = len(parent) == 1 and _windows(parent[0]).drive != ""
        under_home = len(parent) == len(home) and _starts_with(parent, home)
        if under_letter or under_home:
            return REMOTE_MY_DRIVE.joinpath(*parts[i + 1 :])
    return None


def registry_mount_point() -> str | None:
    """``DefaultMountPoint`` of Drive for desktop, when an admin or the user set it."""
    if sys.platform != "win32":
        return None
    import winreg

    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for key in _REGISTRY_KEYS:
            try:
                with winreg.OpenKey(hive, key) as handle:
                    value, _ = winreg.QueryValueEx(handle, "DefaultMountPoint")
            except OSError:
                continue
            if isinstance(value, str) and value.strip():
                return os.path.expandvars(value.strip())
    return None


def _mount_root(mount: str) -> PureWindowsPath:
    # "G" or "G:" is a drive letter; anything else is a folder.
    letter = mount.rstrip(":\\/")
    if len(letter) == 1 and letter.upper() in string.ascii_uppercase:
        return PureWindowsPath(f"{letter.upper()}:\\")
    return _windows(mount)


def my_drive_roots(
    manual: str | None,
    mount_point: Callable[[], str | None] = registry_mount_point,
) -> list[PureWindowsPath]:
    """The "My Drive" folders a path is checked against, the manual one first."""
    roots: list[PureWindowsPath] = []
    if manual and manual.strip():
        roots.append(_windows(manual.strip()))
    mount = mount_point()
    if mount:
        root = _mount_root(mount)
        roots.extend(root / name for name in MY_DRIVE_NAMES)
    return roots


def _system_drives() -> list[str]:
    r"""The drive letters in use, where Drive for desktop puts ``G:\My Drive``."""
    if sys.platform == "win32":
        return os.listdrives()
    return []


def _drive_root_candidates(
    manual: str | None,
    mount_point: Callable[[], str | None],
    home: Path,
    drives: Iterable[str],
) -> list[Path]:
    # The manual and registry roots first, then the defaults Drive for desktop
    # uses: mirror mode under the home folder, stream mode on its own letter.
    roots = [Path(str(root)) for root in my_drive_roots(manual, mount_point)]
    roots.extend(home / name for name in MY_DRIVE_NAMES)
    for drive in drives:
        roots.extend(Path(drive) / name for name in MY_DRIVE_NAMES)
    return roots


def find_my_drive_root(
    manual: str | None,
    mount_point: Callable[[], str | None] = registry_mount_point,
    home: Path | None = None,
    drives: Callable[[], Iterable[str]] = _system_drives,
) -> Path | None:
    """The first "My Drive" folder that exists on this machine, or None.

    The same folders ``remote_path`` trusts, so a folder picked under the one
    found here always translates to the VM.
    """
    candidates = _drive_root_candidates(
        manual, mount_point, Path.home() if home is None else home, drives()
    )
    for root in candidates:
        try:
            if root.is_dir():
                return root
        except OSError:
            continue
    return None


# How many sub-folders one listing returns; a folder with more says so.
MAX_LISTED_FOLDERS = 500
# Windows marks its own folders ($RECYCLE.BIN, System Volume Information) so.
_HIDDEN_ATTRIBUTES = 0x2 | 0x4  # FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM

DriveFolderProblem = Literal["outside", "missing", "unreadable"]


class DriveFolderError(ValueError):
    """A folder the Drive browser does not list, and why."""

    def __init__(self, reason: DriveFolderProblem, rel: str) -> None:
        super().__init__(f"{reason}: {rel!r}")
        self.reason: DriveFolderProblem = reason
        self.rel = rel


@dataclass(frozen=True)
class DriveFolders:
    """One folder under "My Drive": where it is here and on the VM, and its sub-folders."""

    root: Path
    parts: tuple[str, ...]
    path: Path
    remote: PurePosixPath
    folders: list[str]
    truncated: bool


def _rel_parts(rel: str) -> tuple[str, ...]:
    """The folder names of a path relative to "My Drive"; ``..`` and drives refused."""
    parts = tuple(p for p in re.split(r"[\\/]", rel) if p)
    if rel.startswith(("/", "\\")) or any(p in (".", "..") or ":" in p for p in parts):
        raise DriveFolderError("outside", rel)
    return parts


def _listed(entry: os.DirEntry[str]) -> bool:
    """A real, visible folder: no file, link, junction, dot or system folder."""
    if entry.name.startswith(".") or entry.is_symlink() or entry.is_junction():
        return False
    if not entry.is_dir(follow_symlinks=False):
        return False
    attributes = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
    return not attributes & _HIDDEN_ATTRIBUTES


def list_drive_folders(
    root: Path, rel: str, limit: int = MAX_LISTED_FOLDERS
) -> DriveFolders:
    """The sub-folders of ``root/rel``, by name, sorted, at most ``limit`` of them.

    Both paths are resolved (links, junctions, ``..``) before the containment
    check, so nothing here reads a folder outside the "My Drive" root.
    """
    parts = _rel_parts(rel)
    path = root.joinpath(*parts)
    try:
        inside = path.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        raise DriveFolderError("missing", rel) from None
    if not inside:
        raise DriveFolderError("outside", rel)
    remote = remote_path(path, [root])
    if remote is None:
        raise DriveFolderError("outside", rel)
    try:
        with os.scandir(path) as entries:
            names = [entry.name for entry in entries if _listed(entry)]
    except (FileNotFoundError, NotADirectoryError):
        raise DriveFolderError("missing", rel) from None
    except OSError:
        raise DriveFolderError("unreadable", rel) from None
    names.sort(key=lambda name: (name.casefold(), name))
    return DriveFolders(
        root=root,
        parts=parts,
        path=path,
        remote=remote,
        folders=names[:limit],
        truncated=len(names) > limit,
    )


def count_files(folder: Path) -> int:
    """Files under ``folder``, hidden ones left out, as the VM counts them."""
    total = 0
    for _dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        total += sum(1 for f in filenames if not f.startswith("."))
    return total


class DriveConsent:
    """Where a Colab job waits for the person to grant the Drive access.

    The job thread calls ``wait`` once it showed the consent URL; the GUI calls
    ``grant`` when the person comes back to the VisionForge window or presses
    "I already allowed". A grant that arrives while nobody waits is refused, so
    a stale one cannot answer the next attempt's URL.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._granted = threading.Event()
        self._waiting = False

    @property
    def waiting(self) -> bool:
        with self._lock:
            return self._waiting

    def grant(self) -> bool:
        with self._lock:
            if not self._waiting:
                return False
            self._granted.set()
            return True

    def wait(
        self, timeout: float, cancelled: Callable[[], bool], poll: float = 0.5
    ) -> ConsentWait:
        with self._lock:
            self._granted.clear()
            self._waiting = True
        deadline = time.monotonic() + timeout
        try:
            while True:
                if self._granted.wait(min(poll, max(0.0, deadline - time.monotonic()))):
                    return "granted"
                if cancelled():
                    return "cancelled"
                if time.monotonic() >= deadline:
                    return "timeout"
        finally:
            with self._lock:
                self._waiting = False
                self._granted.clear()


__all__ = [
    "MAX_LISTED_FOLDERS",
    "MY_DRIVE_NAMES",
    "REMOTE_MY_DRIVE",
    "DriveConsent",
    "DriveFolderError",
    "DriveFolders",
    "consent_url",
    "count_files",
    "find_my_drive_root",
    "list_drive_folders",
    "my_drive_roots",
    "registry_mount_point",
    "remote_path",
]
