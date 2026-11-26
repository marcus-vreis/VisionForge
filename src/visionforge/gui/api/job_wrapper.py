"""The operator's command in front of a server job's child process (ADR-254).

A job's child (ADR-248) is a process of the server's own system user: it can
open every file that user can. ``VISIONFORGE_JOB_WRAPPER`` names a command the
server puts in front of the child's (``bwrap ...``, ``sudo -n -u vfjob ...``, a
script of the operator's), and that command is what denies the rest of the disk.
The server says which folders the job is for in three variables the wrapper
reads; it does not sandbox anything itself, since how to do so differs by host.
"""

from __future__ import annotations

import os
import shlex
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

WRAPPER_ENV = "VISIONFORGE_JOB_WRAPPER"
# What the wrapper reads: the job's temporary working folder, and the folders
# it writes into and reads from, joined with ``os.pathsep``.
WORKDIR_ENV = "VISIONFORGE_JOB_WORKDIR"
WRITE_ENV = "VISIONFORGE_JOB_WRITE"
READ_ENV = "VISIONFORGE_JOB_READ"


class JobWrapperError(ValueError):
    """``VISIONFORGE_JOB_WRAPPER`` cannot be used as given."""


@dataclass(frozen=True)
class JobFolders:
    """The folders one job is for: where its runs go and where its datasets live."""

    write: tuple[Path, ...] = ()
    read: tuple[Path, ...] = ()


_FOLDERS: ContextVar[JobFolders | None] = ContextVar(
    "visionforge_job_folders", default=None
)


@contextmanager
def job_folders(folders: JobFolders) -> Iterator[None]:
    """Bind ``folders`` for the job started inside the block."""
    token = _FOLDERS.set(folders)
    try:
        yield
    finally:
        _FOLDERS.reset(token)


def wrapper_command(environ: Mapping[str, str] | None = None) -> list[str]:
    """The words to put in front of the child's command; empty when none is set.

    Raises:
        JobWrapperError: the variable cannot be split into words, or its first
            word is not a program this machine has.
    """
    text = (os.environ if environ is None else environ).get(WRAPPER_ENV, "").strip()
    if not text:
        return []
    try:
        words = shlex.split(text, posix=os.name != "nt")
    except ValueError as exc:
        raise JobWrapperError(f"{WRAPPER_ENV} is not a command line: {exc}.") from None
    if os.name == "nt":
        # Without POSIX rules the quotes around a path with spaces stay.
        words = [word.strip('"') for word in words]
    if shutil.which(words[0]) is None:
        raise JobWrapperError(
            f"{WRAPPER_ENV} names {words[0]!r}, which is not a program on this machine."
        )
    return words


def wrapper_env(workdir: Path) -> dict[str, str]:
    """What the wrapper is told about the job being started in this context."""
    folders = _FOLDERS.get() or JobFolders()
    return {
        WORKDIR_ENV: str(workdir),
        WRITE_ENV: os.pathsep.join(str(path) for path in folders.write),
        READ_ENV: os.pathsep.join(str(path) for path in folders.read),
    }


__all__ = [
    "READ_ENV",
    "WORKDIR_ENV",
    "WRAPPER_ENV",
    "WRITE_ENV",
    "JobFolders",
    "JobWrapperError",
    "job_folders",
    "wrapper_command",
    "wrapper_env",
]
