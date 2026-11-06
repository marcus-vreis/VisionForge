"""Background jobs that incorporate new images into a dataset (ADR-166).

Reading and hashing the new folder (the preview) and writing the result (the
apply) both take as long as the dataset is big, so they run in a worker thread
with a progress counter and a cancellation token, like the explorer's scan. One
job runs at a time: both are disk-bound. The preview keeps its :class:`Analysis`
in memory so that changing a checkbox, the seed or the ratios re-plans in
milliseconds instead of re-hashing the dataset, and so that the apply writes
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

from loguru import logger

from visionforge.core.cancellation import CancellationToken
from visionforge.core.dataset_explorer import ExplorerTask
from visionforge.core.dataset_incorporate import (
    Analysis,
    IncorporateError,
    Plan,
    analyze_incorporation,
    build_incorporated_dataset,
)
from visionforge.utils.messages import tr

_KEEP = 4


class JobBusyError(RuntimeError):
    """Another incorporate job is still running."""


@dataclass
class IncorporateJob:
    job_id: str
    kind: str  # preview | apply
    state: str = "running"  # running | done | cancelled | failed
    done: int = 0
    total: int = 0
    result: dict[str, Any] | None = None
    error_code: str | None = None
    error_params: dict[str, object] = field(default_factory=dict)
    error_text: str | None = None
    analysis: Analysis | None = None
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


class IncorporateJobs:
    """Registry of the latest preview and apply jobs."""

    def __init__(self) -> None:
        self._jobs: OrderedDict[str, IncorporateJob] = OrderedDict()
        self._lock = threading.Lock()

    def _register(self, kind: str) -> IncorporateJob:
        with self._lock:
            if any(j.state == "running" for j in self._jobs.values()):
                raise JobBusyError("an incorporate job is already running")
            job = IncorporateJob(job_id=uuid.uuid4().hex, kind=kind)
            self._jobs[job.job_id] = job
            while len(self._jobs) > _KEEP:
                self._jobs.popitem(last=False)
        return job

    def start_preview(
        self,
        base: Path,
        task: ExplorerTask,
        source: Path,
        split_dirs: dict[str, str],
        suggested_name: str,
    ) -> IncorporateJob:
        """Launch the read-and-compare pass.

        Raises:
            JobBusyError: if another job is still running.
        """
        job = self._register("preview")

        def run() -> None:
            analysis = analyze_incorporation(
                base,
                task,
                source,
                split_dirs,
                token=job.token,
                on_progress=lambda done, total: _progress(job, done, total),
            )
            if analysis is None:
                job.state = "cancelled"
                return
            job.analysis = analysis
            job.result = {
                "suggested_name": suggested_name,
                "task": analysis.task,
                "base_dir": str(analysis.base_root),
                "source_dir": str(analysis.source_root),
                "base_classes": analysis.base_classes,
                "base_counts": analysis.base_counts,
                "present": analysis.present,
                "source_images": analysis.source_images,
                "new_classes": [
                    {"name": c.name, "images": c.images, "source_id": c.source_id}
                    for c in analysis.new_classes
                ],
            }
            job.state = "done"

        self._spawn(job, run)
        return job

    def start_apply(
        self, analysis: Analysis, plan: Plan, dest_name: str
    ) -> IncorporateJob:
        """Launch the write.

        Raises:
            JobBusyError: if another job is still running.
        """
        job = self._register("apply")

        def run() -> None:
            result = build_incorporated_dataset(
                analysis,
                plan,
                dest_name,
                token=job.token,
                on_progress=lambda done, total: _progress(job, done, total),
            )
            if result is None:
                job.state = "cancelled"
                return
            job.result = {
                "path": result["path"],
                "added": result["added"],
                "files_linked": result["registry"]["files_linked"],
                "files_copied": result["registry"]["files_copied"],
                "classes": result["registry"]["classes"],
            }
            job.state = "done"

        self._spawn(job, run)
        return job

    @staticmethod
    def _spawn(job: IncorporateJob, run: Callable[[], None]) -> None:
        def guarded() -> None:
            try:
                run()
            except IncorporateError as exc:
                job.error_code, job.error_params = exc.code, dict(exc.params)
                job.state = "failed"
            # Reported to the client, not raised.
            except Exception as exc:  # noqa: BLE001
                logger.error("Incorporate job failed: {}", type(exc).__name__)
                job.error_text = f"{type(exc).__name__}: {exc}"
                job.state = "failed"

        threading.Thread(target=guarded, daemon=True).start()

    def get(self, job_id: str) -> IncorporateJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        job.token.cancel()
        return True


def _progress(job: IncorporateJob, done: int, total: int) -> None:
    job.done, job.total = done, total
