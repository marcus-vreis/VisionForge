"""FIFO queue for training submissions on one GPU (ADR-075).

The GUI has always accepted one run at a time: a second submit got a 409 and the
researcher had to sit at the machine and resubmit when the first finished. That
makes the obvious overnight workflow — line up the evening's experiments and walk
away — impossible for anything the sweep/replicates orchestrators do not already
parametrize, which is exactly the heterogeneous case (a detection run, then a
segmentation run, then the same classifier on a different dataset).

This module holds the pending list and drains it one job at a time, so the
machine stays single-run while submission stops being blocking.

Two things had to change beyond "keep a list", and they are the reason this is a
module rather than a few lines in the route layer:

- **A finished run's result has to survive the next one starting.** The route
  layer kept exactly one ``_current_run`` dict, which ``/experiment/result`` read
  from; the second job would overwrite the first job's report before the browser
  fetched it. Terminal snapshots are therefore recorded per ``run_id`` here.
- **The SSE queue must be created when a job starts, not when it is submitted.**
  It used to be built in the request handler, which was safe only because a
  second submit was refused: with queueing, submitting job 2 would replace the
  live queue of job 1 and its progress stream would go dead.

The queue is deliberately in-memory and not persisted. It describes what this
server process is about to do; a restart losing the pending list is correct
behavior, not a gap — the durable record of a run is its ``run.json`` on disk.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

# Terminal snapshots are kept so a result can be fetched after later runs start.
# Bounded because a long session would otherwise grow without limit, and the
# durable record is run.json — this cache only serves the "just finished" fetch.
_MAX_RECORDS = 100


@dataclass
class QueuedJob:
    """One submission: what to show in the queue, and the coroutine to run."""

    run_id: str
    label: str  # experiment name, for the queue panel
    task: str  # classification | detection | ... | custom:<key>
    strategy: str  # simple | sweep | replicates | cv | comparison | ...
    start: Callable[[], Awaitable[None]]
    submitted_at: datetime = field(default_factory=datetime.now)

    def describe(self) -> dict[str, Any]:
        """JSON-ready form for the queue endpoint (never includes the callable)."""
        return {
            "run_id": self.run_id,
            "label": self.label,
            "task": self.task,
            "strategy": self.strategy,
            "submitted_at": self.submitted_at.isoformat(),
        }


class RunQueue:
    """Accepts submissions at any time and executes them one at a time.

    The two callbacks are how this stays independent of the route module's
    globals: ``on_start`` prepares per-run state (the fresh SSE queue and the
    "running" marker) and ``on_finish`` hands back the terminal snapshot to
    record under the job's ``run_id``.
    """

    def __init__(
        self,
        on_start: Callable[[QueuedJob], None],
        on_finish: Callable[[QueuedJob], dict[str, Any] | None],
    ) -> None:
        self._on_start = on_start
        self._on_finish = on_finish
        self._pending: deque[QueuedJob] = deque()
        self._active: QueuedJob | None = None
        self._records: dict[str, dict[str, Any]] = {}
        self._worker: asyncio.Task[None] | None = None

    # ── submission ────────────────────────────────────────────────────────────

    def submit(self, job: QueuedJob) -> str:
        """Enqueue a job; returns ``"running"`` if it starts now, else ``"queued"``."""
        self._pending.append(job)
        starts_now = self._active is None
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._drain())
        if not starts_now:
            logger.info(
                "Queue: {} queued behind {} ({} waiting).",
                job.run_id,
                self._active.run_id if self._active else "?",
                len(self._pending),
            )
        return "running" if starts_now else "queued"

    def cancel(self, run_id: str) -> bool:
        """Drop a *pending* job. Returns False if it is unknown or already running.

        A running job is deliberately not cancellable here: the trainers own
        their loop and have no cooperative stop point, so "cancel" would either
        lie or leave a half-written run directory.
        """
        for job in list(self._pending):
            if job.run_id == run_id:
                self._pending.remove(job)
                logger.info("Queue: {} cancelled before it started.", run_id)
                return True
        return False

    # ── inspection ────────────────────────────────────────────────────────────

    @property
    def active_run_id(self) -> str | None:
        """The run currently executing, if any."""
        return self._active.run_id if self._active else None

    def is_busy(self) -> bool:
        """True while a job is executing."""
        return self._active is not None

    def pending_count(self) -> int:
        """How many jobs are waiting to start."""
        return len(self._pending)

    def snapshot(self) -> dict[str, Any]:
        """Active job + ordered pending list, for ``GET /api/queue``."""
        return {
            "active": self._active.describe() if self._active else None,
            "pending": [job.describe() for job in self._pending],
        }

    def record(self, run_id: str) -> dict[str, Any] | None:
        """The terminal snapshot of a finished run, or None if not recorded."""
        return self._records.get(run_id)

    def reset(self) -> None:
        """Drop all state. For tests — a live server has no reason to call it."""
        self._pending.clear()
        self._active = None
        self._records.clear()
        if self._worker is not None and not self._worker.done():
            self._worker.cancel()
        self._worker = None

    # ── worker ────────────────────────────────────────────────────────────────

    async def _drain(self) -> None:
        """Run pending jobs in submission order until none are left."""
        while self._pending:
            job = self._pending.popleft()
            self._active = job
            try:
                self._on_start(job)
                await job.start()
            except Exception:
                # The executors already convert failures into a "failed" status;
                # reaching here means something outside them broke. Log it and
                # keep draining — one bad job must not strand the rest of the
                # queue, which is the whole point of submitting a batch.
                logger.exception("Queue: {} raised outside its executor.", job.run_id)
            finally:
                snapshot = self._on_finish(job)
                if snapshot is not None:
                    self._remember(job.run_id, snapshot)
                self._active = None

    def _remember(self, run_id: str, snapshot: dict[str, Any]) -> None:
        self._records[run_id] = snapshot
        while len(self._records) > _MAX_RECORDS:
            self._records.pop(next(iter(self._records)))


__all__ = ["QueuedJob", "RunQueue"]
