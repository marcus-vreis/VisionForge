"""Background jobs shared by the endpoints that write a derived dataset.

Incorporating images (ADR-166) and merging datasets (ADR-167) both read and hash a
folder (the preview) and then write a sibling folder (the apply). Both take as long
as the dataset is big, so they run in a worker thread with a progress counter and
a cancellation token, like the explorer's scan. One job of a kind runs at a time
per person (on a local install that is one at a time): both are disk-bound. A job
belongs to whoever started it, and nobody else can read, plan, apply or cancel it
(server mode, ADR-220). The preview keeps its analysis in memory so that changing a
choice re-plans in milliseconds instead of re-hashing, and so that the apply writes
exactly what the user saw.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from loguru import logger

from visionforge.core.cancellation import CancellationToken
from visionforge.core.dataset_derived import DerivedError
from visionforge.gui.api.dataset_scope import current_scope
from visionforge.utils.messages import tr

_KEEP = 4


class JobBusyError(RuntimeError):
    """Another job of the same kind is still running."""


def job_owner() -> str:
    """Who the request being served belongs to; ``""`` on a local install."""
    scope = current_scope()
    return scope.owner if scope is not None else ""


def copy_only(src: Path, dst: Path) -> None:
    """A ``link`` that never links, so ``link_or_copy`` copies the file."""
    raise OSError("hard links are not used across a shared and a personal folder")


def placement(source_root: Path) -> dict[str, Any]:
    """Where the dataset derived from ``source_root`` is written (ADR-220).

    Next to its source, which is what the build functions do with no keyword:
    always on a local install, and on a server when the source is inside the
    person's own dataset folder. A source in a shared folder is the operator's:
    its result goes into the person's own folder instead, copied and never
    hard-linked, so nothing of theirs shares a file with what everyone reads.

    Returns:
        The keywords for the build function: empty, or ``parent`` and ``link``.

    Raises:
        HTTPException: 403 on a server with no folder of the person's to write in.
    """
    scope = current_scope()
    if scope is None:
        return {}
    if scope.own is None:
        raise HTTPException(403, tr("server.dataset_write_outside"))
    beside = source_root.parent.resolve()
    if beside == scope.own or beside.is_relative_to(scope.own):
        return {}
    return {"parent": scope.own, "link": copy_only}


def destination_parent(source_root: Path) -> Path:
    """The folder the result of ``source_root`` will be created in."""
    parent: Path = placement(source_root).get("parent", source_root.parent)
    return parent


@dataclass
class DerivedJob:
    job_id: str
    kind: str  # preview | apply
    owner: str = ""
    state: str = "running"  # running | done | cancelled | failed
    done: int = 0
    total: int = 0
    result: dict[str, Any] | None = None
    error_code: str | None = None
    error_params: dict[str, object] = field(default_factory=dict)
    error_text: str | None = None
    analysis: Any = None
    token: CancellationToken = field(default_factory=CancellationToken)

    def snapshot(self) -> dict[str, Any]:
        """JSON form; the error is translated in the language of the asking request."""
        error = self.error_text
        if self.error_code is not None:
            error = tr(self.error_code, **self.error_params)
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "state": self.state,
            "done": self.done,
            "total": self.total,
            "result": self.result,
            "error": error,
        }

    def progress(self, done: int, total: int) -> None:
        self.done, self.total = done, total


class DerivedJobs:
    """Registry of the latest preview and apply jobs of one feature."""

    def __init__(self, label: str) -> None:
        self._label = label
        self._jobs: OrderedDict[str, DerivedJob] = OrderedDict()
        self._lock = threading.Lock()

    def register(self, kind: str) -> DerivedJob:
        """Add a job.

        Raises:
            JobBusyError: if another job is still running.
        """
        owner = job_owner()
        with self._lock:
            mine = [j for j in self._jobs.values() if j.owner == owner]
            if any(j.state == "running" for j in mine):
                raise JobBusyError(f"a {self._label} job is already running")
            job = DerivedJob(job_id=uuid.uuid4().hex, kind=kind, owner=owner)
            self._jobs[job.job_id] = job
            # The oldest of this person's go first: someone else's jobs never
            # push out a preview this person is about to apply.
            for old in mine[: max(0, len(mine) + 1 - _KEEP)]:
                del self._jobs[old.job_id]
        return job

    def spawn(self, job: DerivedJob, run: Callable[[], None]) -> None:
        """Run ``run`` in a daemon thread; refusals and errors end up on the job."""

        def guarded() -> None:
            try:
                run()
            except DerivedError as exc:
                job.error_code, job.error_params = exc.code, dict(exc.params)
                job.state = "failed"
            # Reported to the client, not raised.
            except Exception as exc:  # noqa: BLE001
                logger.error("{} job failed: {}", self._label, type(exc).__name__)
                job.error_text = f"{type(exc).__name__}: {exc}"
                job.state = "failed"

        threading.Thread(target=guarded, daemon=True).start()

    def get(self, job_id: str) -> DerivedJob | None:
        """The job, for whoever started it; ``None`` for anybody else."""
        with self._lock:
            job = self._jobs.get(job_id)
        return job if job is not None and job.owner == job_owner() else None

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        job.token.cancel()
        return True
