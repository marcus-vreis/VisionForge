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
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from visionforge.core.cancellation import CancellationToken

# Terminal snapshots are kept so a result can be fetched after later runs start.
# Bounded because a long session would otherwise grow without limit, and the
# durable record is run.json — this cache only serves the "just finished" fetch.
_MAX_RECORDS = 100

# Where a running job stops when asked (ADR-111): the boundary after which it
# starts nothing new. Inside a multi-unit job the training in flight is still cut
# at its own epoch where it has one; "phase" is PatchCore, which has no epochs.
# "vm" is a Colab job (ADR-163): stopping releases the VM at once.
StopPoint = Literal["epoch", "trial", "fold", "model", "replicate", "phase", "vm"]


class NotStoppableError(Exception):
    """The running job never reads a stop request, so none was sent."""


@dataclass
class QueuedJob:
    """One submission: what to show in the queue, and the coroutine to run."""

    run_id: str
    label: str  # experiment name, for the queue panel
    task: str  # classification | detection | ... | custom:<key>
    strategy: str  # simple | sweep | replicates | cv | comparison | ...
    start: Callable[[], Awaitable[None]]
    submitted_at: datetime = field(default_factory=datetime.now)
    # Handed to the trainer so a running job can be asked to stop at its next
    # epoch boundary. Pending jobs never read it -- they are simply dropped.
    cancel_token: CancellationToken = field(default_factory=CancellationToken)
    # None for a job whose executor never reads the token (a custom task that
    # owns its loop): cancelling it would report a stop that does not happen.
    stop_at: StopPoint | None = "epoch"
    # Whose folders the job writes into (ADR-114): the profile slug, and the
    # display name the queue panel prints. The queue itself stays shared.
    profile: str = "default"
    profile_name: str = "default"
    resume_dir: Path | None = None
    # Which queue runs it (ADR-163, phase 4): LOCAL_LANE, or the lane of one
    # Google account. Set by RunLanes.submit.
    lane: str = "local"

    def describe(self) -> dict[str, Any]:
        """JSON-ready form for the queue endpoint (never includes the callable)."""
        return {
            "run_id": self.run_id,
            "label": self.label,
            "task": self.task,
            "strategy": self.strategy,
            "submitted_at": self.submitted_at.isoformat(),
            "stop_at": self.stop_at,
            "profile": self.profile,
            "profile_name": self.profile_name,
            "lane": self.lane,
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
        """Stop a job, whether it is waiting or already training.

        A pending job is dropped outright. A running one is *asked* to stop:
        ADR-075 refused this because the trainers had no safe place to stop, and
        ADR-088 gave them one — the epoch boundary, where the checkpoint is
        already written. So this returns True meaning "the request was
        delivered", not "training has ended"; the run stops at its ``stop_at``
        boundary and keeps the best checkpoint it has earned.

        Raises:
            NotStoppableError: the job is running and its ``stop_at`` is None.
                The token is left untouched, so nothing claims a stop.
        """
        for job in self._pending:
            if job.run_id == run_id:
                self._pending.remove(job)
                logger.info("Queue: {} cancelled before it started.", run_id)
                return True
        if self._active is not None and self._active.run_id == run_id:
            if self._active.stop_at is None:
                raise NotStoppableError(run_id)
            self._active.cancel_token.cancel()
            logger.info(
                "Queue: {} asked to stop at the next {} boundary.",
                run_id,
                self._active.stop_at,
            )
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

    def is_writing(self, run_id: str, run_dir: Path, profile: str) -> bool:
        """Whether an active or queued job can change the selected run."""
        jobs = list(self._pending)
        if self._active is not None:
            jobs.append(self._active)
        return any(
            job.profile == profile
            and (
                job.run_id in {run_id, f"{run_dir.parent.name}_{run_dir.name}"}
                or (
                    job.resume_dir is not None
                    and job.resume_dir.resolve() == run_dir.resolve()
                )
            )
            for job in jobs
        )

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


LOCAL_LANE = "local"
COLAB_LANE_PREFIX = "colab:"


def colab_lane(account: str) -> str:
    """The lane of one Google account: profiles on the same account share it."""
    return COLAB_LANE_PREFIX + account


class RunLanes:
    """The queues that may run at the same time (ADR-163, phase 4).

    One lane per resource that runs a job: this machine (GPU or CPU, never both
    at once, as ADR-075 made it) and each Google account, whose session quota
    and compute units are the account's. Inside a lane jobs still run one at a
    time; across lanes they run side by side. A run id lives in one lane, so the
    lookups below search them all.
    """

    def __init__(self, local: RunQueue, make_lane: Callable[[], RunQueue]) -> None:
        self._lanes: dict[str, RunQueue] = {LOCAL_LANE: local}
        self._make_lane = make_lane

    def lane(self, key: str) -> RunQueue:
        """The queue of ``key``, created the first time a job goes to it."""
        if key not in self._lanes:
            self._lanes[key] = self._make_lane()
        return self._lanes[key]

    def submit(self, job: QueuedJob, lane: str = LOCAL_LANE) -> str:
        """Enqueue in ``lane``; ``"running"`` if it starts now, else ``"queued"``."""
        job.lane = lane
        return self.lane(lane).submit(job)

    def cancel(self, run_id: str) -> bool:
        """Stop the job with that id in whichever lane holds it (see RunQueue.cancel)."""
        return any(queue.cancel(run_id) for queue in list(self._lanes.values()))

    def is_busy(self) -> bool:
        """True while any lane runs a job or has one waiting."""
        return any(
            queue.is_busy() or queue.pending_count() for queue in self._lanes.values()
        )

    def pending_count(self) -> int:
        """Jobs waiting to start, over all lanes."""
        return sum(queue.pending_count() for queue in self._lanes.values())

    def is_writing(self, run_id: str, run_dir: Path, profile: str) -> bool:
        return any(
            queue.is_writing(run_id, run_dir, profile) for queue in self._lanes.values()
        )

    def record(self, run_id: str) -> dict[str, Any] | None:
        """The terminal snapshot of a finished run, from whichever lane ran it."""
        for queue in self._lanes.values():
            snapshot = queue.record(run_id)
            if snapshot is not None:
                return snapshot
        return None

    def snapshots(self) -> list[dict[str, Any]]:
        """Each lane's active job and pending list, the local lane first."""
        return [{"lane": key, **queue.snapshot()} for key, queue in self._lanes.items()]

    def reset(self) -> None:
        """Drop all state and every lane but the local one. For tests."""
        for queue in self._lanes.values():
            queue.reset()
        self._lanes = {LOCAL_LANE: self._lanes[LOCAL_LANE]}


__all__ = [
    "COLAB_LANE_PREFIX",
    "LOCAL_LANE",
    "NotStoppableError",
    "QueuedJob",
    "RunLanes",
    "RunQueue",
    "StopPoint",
    "colab_lane",
]
