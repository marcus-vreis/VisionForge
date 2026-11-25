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

On a server for several people (ADR-245) a lane is *fair*: the next job is the
oldest one of the person whose job started least recently, so one person's
batch does not hold everybody else behind it. On a local install it stays FIFO.
"""

from __future__ import annotations

import asyncio
import time
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

# How many jobs one person may have waiting or running on a server (ADR-245),
# when the operator does not say (``--max-queued``).
DEFAULT_MAX_QUEUED = 3

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
    # On a server, the folders of the person's own dataset folder the job reads,
    # and where its runs are written (ADR-249): a dataset sent to be deleted
    # after the run is kept while a job still names it.
    datasets: tuple[Path, ...] = ()
    models_dir: Path | None = None
    # When the queue started it (``time.time()``); None while it waits.
    started_at: float | None = None

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

    The callbacks are how this stays independent of the route module's
    globals: ``on_start`` prepares per-run state (the fresh SSE queue and the
    "running" marker) and ``on_finish`` hands back the terminal snapshot to
    record under the job's ``run_id``. ``on_end``, when given, is awaited once
    the job ended (completed, failed or stopped) and before that snapshot is
    taken, while the job is still the lane's active one (ADR-249).
    """

    def __init__(
        self,
        on_start: Callable[[QueuedJob], None],
        on_finish: Callable[[QueuedJob], dict[str, Any] | None],
        on_end: Callable[[QueuedJob], Awaitable[None]] | None = None,
    ) -> None:
        self._on_start = on_start
        self._on_finish = on_finish
        self._on_end = on_end
        self._pending: deque[QueuedJob] = deque()
        self._active: QueuedJob | None = None
        self._records: dict[str, dict[str, Any]] = {}
        self._worker: asyncio.Task[None] | None = None
        # Fair order (ADR-245): turned on by ``make_fair`` and kept, since a
        # process is a server for several people for its whole life or never.
        self._fair = False
        # When each profile's job last started, as a count of starts.
        self._starts = 0
        self._last_start: dict[str, int] = {}

    # ── submission ────────────────────────────────────────────────────────────

    def make_fair(self) -> None:
        """Pick the next job across profiles from now on (see :meth:`run_order`).

        A server for several people calls it before each submission; nothing
        turns it off but :meth:`reset`.
        """
        self._fair = True

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

    def count_of(self, profile: str) -> int:
        """How many jobs of ``profile`` wait or run here."""
        jobs = [self._active, *self._pending]
        return sum(1 for job in jobs if job is not None and job.profile == profile)

    def jobs(self) -> list[QueuedJob]:
        """The running job, if any, then the waiting ones."""
        return [job for job in (self._active, *self._pending) if job is not None]

    def run_order(self) -> list[QueuedJob]:
        """The waiting jobs in the order they will start, if nobody submits more.

        FIFO on a local install. In a fair lane (ADR-245) each next job is the
        oldest one of the profile whose job started least recently; a profile
        that never ran here comes first, and between those the oldest job wins.
        """
        if not self._fair:
            return list(self._pending)
        last = dict(self._last_start)
        starts = self._starts
        left = list(self._pending)
        order: list[QueuedJob] = []
        while left:
            index = min(
                range(len(left)), key=lambda i: (last.get(left[i].profile, -1), i)
            )
            job = left.pop(index)
            starts += 1
            last[job.profile] = starts
            order.append(job)
        return order

    def snapshot(self) -> dict[str, Any]:
        """Active job + pending list in run order, for ``GET /api/queue``."""
        return {
            "active": self._active.describe() if self._active else None,
            "pending": [job.describe() for job in self.run_order()],
        }

    def record(self, run_id: str) -> dict[str, Any] | None:
        """The terminal snapshot of a finished run, or None if not recorded."""
        return self._records.get(run_id)

    def profile_of(self, run_id: str) -> str | None:
        """Whose the running or waiting job ``run_id`` is; None if not here."""
        jobs = [self._active, *self._pending]
        return next((j.profile for j in jobs if j and j.run_id == run_id), None)

    def reset(self) -> None:
        """Drop all state. For tests — a live server has no reason to call it."""
        self._pending.clear()
        self._active = None
        self._records.clear()
        if self._worker is not None and not self._worker.done():
            self._worker.cancel()
        self._worker = None
        self._fair = False
        self._starts = 0
        self._last_start.clear()

    # ── worker ────────────────────────────────────────────────────────────────

    def _next(self) -> QueuedJob:
        """Take the job that starts now: the first of :meth:`run_order`."""
        job = self.run_order()[0] if self._fair else self._pending[0]
        self._pending.remove(job)
        self._starts += 1
        self._last_start[job.profile] = self._starts
        return job

    async def _drain(self) -> None:
        """Run pending jobs in run order until none are left."""
        while self._pending:
            job = self._next()
            self._active = job
            job.started_at = time.time()
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
                await self._ended(job)
                snapshot = self._on_finish(job)
                if snapshot is not None:
                    self._remember(job.run_id, snapshot)
                self._active = None

    async def _ended(self, job: QueuedJob) -> None:
        """Await ``on_end``; what it raises is logged, never the next job's."""
        if self._on_end is None:
            return
        try:
            await self._on_end(job)
        except Exception:
            logger.exception("Queue: the end step of {} failed.", job.run_id)

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

    def submit(
        self, job: QueuedJob, lane: str = LOCAL_LANE, *, fair: bool = False
    ) -> str:
        """Enqueue in ``lane``; ``"running"`` if it starts now, else ``"queued"``.

        ``fair`` turns the lane's fair order on (:meth:`RunQueue.make_fair`).
        """
        job.lane = lane
        queue = self.lane(lane)
        if fair:
            queue.make_fair()
        return queue.submit(job)

    def cancel(self, run_id: str) -> bool:
        """Stop the job with that id in whichever lane holds it (see RunQueue.cancel)."""
        return any(queue.cancel(run_id) for queue in self._lanes.values())

    def is_busy(self) -> bool:
        """True while any lane runs a job or has one waiting."""
        return any(
            queue.is_busy() or queue.pending_count() for queue in self._lanes.values()
        )

    def pending_count(self) -> int:
        """Jobs waiting to start, over all lanes."""
        return sum(queue.pending_count() for queue in self._lanes.values())

    def count_of(self, profile: str) -> int:
        """How many jobs of ``profile`` wait or run, over all lanes."""
        return sum(queue.count_of(profile) for queue in self._lanes.values())

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

    def profile_of(self, run_id: str) -> str | None:
        """Whose the running or waiting job ``run_id`` is, in whichever lane."""
        for queue in self._lanes.values():
            profile = queue.profile_of(run_id)
            if profile is not None:
                return profile
        return None

    def jobs(self) -> list[QueuedJob]:
        """Every running and waiting job, over all lanes."""
        return [job for queue in self._lanes.values() for job in queue.jobs()]

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
    "DEFAULT_MAX_QUEUED",
    "LOCAL_LANE",
    "NotStoppableError",
    "QueuedJob",
    "RunLanes",
    "RunQueue",
    "StopPoint",
    "colab_lane",
]
