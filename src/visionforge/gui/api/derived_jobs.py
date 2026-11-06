"""Background jobs shared by the endpoints that write a derived dataset.

Incorporating images (ADR-166) and merging datasets (ADR-167) both read and hash a
folder (the preview) and then write a sibling folder (the apply). Both take as long
as the dataset is big, so they run in a worker thread with a progress counter and
a cancellation token, like the explorer's scan. One job of a kind runs at a time:
both are disk-bound. The preview keeps its analysis in memory so that changing a
choice re-plans in milliseconds instead of re-hashing, and so that the apply writes
exactly what the user saw.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from visionforge.core.cancellation import CancellationToken
from visionforge.core.dataset_derived import DerivedError
from visionforge.utils.messages import tr

_KEEP = 4


class JobBusyError(RuntimeError):
    """Another job of the same kind is still running."""


@dataclass
class DerivedJob:
    job_id: str
    kind: str  # preview | apply
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
        with self._lock:
            if any(j.state == "running" for j in self._jobs.values()):
                raise JobBusyError(f"a {self._label} job is already running")
            job = DerivedJob(job_id=uuid.uuid4().hex, kind=kind)
            self._jobs[job.job_id] = job
            while len(self._jobs) > _KEEP:
                self._jobs.popitem(last=False)
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
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        job.token.cancel()
        return True
