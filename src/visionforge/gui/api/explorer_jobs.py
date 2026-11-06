"""Background integrity scans for the dataset explorer (ADR-165).

A scan opens and hashes every image, which on a big dataset takes minutes, so it
runs in a worker thread with a progress counter and a cancellation token instead
of inside the request. One scan runs at a time: they are disk-bound and a second
one would only slow the first.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from visionforge.core.cancellation import CancellationToken
from visionforge.core.dataset_explorer import DatasetIndex, scan_integrity

_KEEP = 8


class ScanBusyError(RuntimeError):
    """A scan is already running."""


@dataclass
class ScanJob:
    job_id: str
    state: str = "running"  # running | done | cancelled | failed
    done: int = 0
    total: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
    token: CancellationToken = field(default_factory=CancellationToken)

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "done": self.done,
            "total": self.total,
            "result": self.result,
            "error": self.error,
        }


class ScanJobs:
    """Registry of the latest scans, safe to call from request threads."""

    def __init__(self) -> None:
        self._jobs: OrderedDict[str, ScanJob] = OrderedDict()
        self._lock = threading.Lock()

    def start(self, index: DatasetIndex, max_images: int) -> ScanJob:
        """Launch a scan.

        Raises:
            ScanBusyError: if another scan is still running.
        """
        with self._lock:
            if any(j.state == "running" for j in self._jobs.values()):
                raise ScanBusyError("a scan is already running")
            job = ScanJob(job_id=uuid.uuid4().hex)
            self._jobs[job.job_id] = job
            while len(self._jobs) > _KEEP:
                self._jobs.popitem(last=False)
        threading.Thread(
            target=self._run, args=(job, index, max_images), daemon=True
        ).start()
        return job

    def _run(self, job: ScanJob, index: DatasetIndex, max_images: int) -> None:
        def progress(done: int, total: int) -> None:
            job.done, job.total = done, total

        try:
            result = scan_integrity(
                index, max_images=max_images, token=job.token, on_progress=progress
            )
        # Reported to the client, not raised.
        except Exception as exc:  # noqa: BLE001
            logger.error("Explorer scan failed: {}", type(exc).__name__)
            job.error = f"{type(exc).__name__}: {exc}"
            job.state = "failed"
            return
        job.result = result
        job.state = "cancelled" if result["cancelled"] else "done"

    def get(self, job_id: str) -> ScanJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        job.token.cancel()
        return True
