"""Which folders a request may read datasets from (server mode, ADR-214).

On a local install a request may name any folder: the person at the browser
owns the machine. In server mode the gate (``gui/server_mode.py``) binds a
:class:`DatasetScope` for the request: the signed-in person's own dataset
folder and the folders the operator shares. The request models of the dataset
routes type their path fields with :data:`DatasetDir` and :data:`DatasetPart`,
so a path outside the scope is refused while the body is validated, before any
handler touches the disk.

With no scope bound (the local tool) both types accept what ``str`` accepts.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated

from pydantic import AfterValidator

from visionforge.utils.messages import tr


@dataclass(frozen=True)
class DatasetScope:
    """The resolved folders one request may read datasets from, own folder first.

    ``owner`` is the profile of the signed-in person: the routes that list or
    stop jobs show and touch only that profile's (ADR-215). ``own`` is that
    person's dataset folder, the only place a request may write a dataset
    (ADR-219); it is one of ``readable``.
    """

    readable: tuple[Path, ...]
    owner: str = ""
    own: Path | None = None

    def holds(self, path: Path) -> bool:
        """True when the resolved ``path`` is one of the folders or inside one."""
        return any(path == root or path.is_relative_to(root) for root in self.readable)


_SCOPE: ContextVar[DatasetScope | None] = ContextVar(
    "visionforge_dataset_scope", default=None
)


def current_scope() -> DatasetScope | None:
    """The scope of the request being served; ``None`` on a local install."""
    return _SCOPE.get()


@contextmanager
def dataset_scope(scope: DatasetScope | None) -> Iterator[None]:
    """Bind ``scope`` for the code inside the block."""
    token = _SCOPE.set(scope)
    try:
        yield
    finally:
        _SCOPE.reset(token)


def _confined_dir(raw: str) -> str:
    """``raw`` resolved, when a scope is bound and holds it; else refused."""
    scope = current_scope()
    if scope is None:
        return raw
    text = raw.strip()
    # ``~`` is the home of the account the server runs as, nobody's dataset.
    if not text or text.startswith("~"):
        raise ValueError(tr("server.dataset_outside"))
    candidate = Path(text)
    if not candidate.is_absolute():
        raise ValueError(tr("server.dataset_outside"))
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(tr("server.dataset_outside")) from exc
    if not scope.holds(resolved):
        raise ValueError(tr("server.dataset_outside"))
    return str(resolved)


def _relative_part(raw: str) -> str:
    """``raw`` as is, when a scope is bound and it only goes down; else refused."""
    if current_scope() is None:
        return raw
    windows, posix = PureWindowsPath(raw), PurePosixPath(raw)
    climbs = ".." in windows.parts or ".." in posix.parts
    rooted = bool(windows.drive or windows.root or posix.root)
    if climbs or rooted or raw.strip().startswith("~"):
        raise ValueError(tr("server.dataset_part", value=raw[:80]))
    return raw


DatasetDir = Annotated[str, AfterValidator(_confined_dir)]
"""A dataset folder. In server mode it must be absolute and sit in the scope."""


def _own_new_dir(raw: str) -> str:
    """``raw`` resolved, when it is a folder inside the person's own; else refused.

    The own folder itself is refused too: a download writes split folders, and
    a second one into the same place would mix two datasets.
    """
    scope = current_scope()
    if scope is None:
        return raw
    text = raw.strip()
    candidate = Path(text)
    if scope.own is None or not text or text.startswith("~"):
        raise ValueError(tr("server.dataset_write_outside"))
    if not candidate.is_absolute():
        raise ValueError(tr("server.dataset_write_outside"))
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(tr("server.dataset_write_outside")) from exc
    if resolved == scope.own or not resolved.is_relative_to(scope.own):
        raise ValueError(tr("server.dataset_write_outside"))
    return str(resolved)


def _run_folder_only(raw: str | None) -> str | None:
    """``raw`` as is on a local install; on a server only an empty one passes."""
    if current_scope() is not None and raw is not None and raw.strip():
        raise ValueError(tr("server.output_fixed"))
    return raw


OwnDatasetDir = Annotated[str, AfterValidator(_own_new_dir)]
"""A folder a dataset is written into. In server mode it must be absolute and
sit inside the signed-in person's own dataset folder (ADR-219)."""

RunFolderFile = Annotated[str | None, AfterValidator(_run_folder_only)]
"""Where an action on a run writes its file. In server mode it stays empty, and
the file goes into the run's own folder (ADR-218)."""

DatasetPart = Annotated[str, AfterValidator(_relative_part)]
"""A name inside a dataset folder (``train``, ``images``, ``train.csv``). In
server mode it may only go down: no ``..``, no root, no drive, no ``~``."""


__all__ = [
    "DatasetDir",
    "DatasetPart",
    "DatasetScope",
    "current_scope",
    "dataset_scope",
]
