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
from pathlib import Path, PurePosixPath
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


def remote_path(local: Path, roots: Iterable[Path]) -> PurePosixPath | None:
    """Where ``local`` is on the VM, or None when it is not inside the Drive.

    ``roots`` are "My Drive" folders known for sure (the manual one, the one the
    registry names). Without a match, a path component named like "My Drive"
    right under a drive root (``G:\\Meu Drive``) or under the home folder
    (mirror mode) counts as well.
    """
    parts = Path(os.path.abspath(local)).parts
    for root in roots:
        root_parts = Path(os.path.abspath(root)).parts
        if len(parts) >= len(root_parts) and all(
            _same(a, b) for a, b in zip(parts, root_parts, strict=False)
        ):
            return REMOTE_MY_DRIVE.joinpath(*parts[len(root_parts) :])
    home = Path(os.path.abspath(Path.home())).parts
    for i, part in enumerate(parts):
        if not any(_same(part, name) for name in MY_DRIVE_NAMES):
            continue
        parent = parts[:i]
        under_volume = len(parent) == 1
        under_home = len(parent) == len(home) and all(
            _same(a, b) for a, b in zip(parent, home, strict=False)
        )
        if under_volume or under_home:
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


def _mount_root(mount: str) -> Path:
    # "G" or "G:" is a drive letter; anything else is a folder.
    letter = mount.rstrip(":\\/")
    if len(letter) == 1 and letter.upper() in string.ascii_uppercase:
        return Path(f"{letter.upper()}:\\")
    return Path(mount)


def my_drive_roots(
    manual: str | None,
    mount_point: Callable[[], str | None] = registry_mount_point,
) -> list[Path]:
    """The "My Drive" folders a path is checked against, the manual one first."""
    roots: list[Path] = []
    if manual and manual.strip():
        roots.append(Path(manual.strip()))
    mount = mount_point()
    if mount:
        root = _mount_root(mount)
        roots.extend(root / name for name in MY_DRIVE_NAMES)
    return roots


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
    "MY_DRIVE_NAMES",
    "REMOTE_MY_DRIVE",
    "DriveConsent",
    "consent_url",
    "count_files",
    "my_drive_roots",
    "registry_mount_point",
    "remote_path",
]
