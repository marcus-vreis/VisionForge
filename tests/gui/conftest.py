"""Shared helpers for the route tests.

The run queue (ADR-075) replaced the old "409 while busy" contract, so the tests
that used to fake a busy server by assigning ``routes._current_run`` now need to
make the *queue* busy instead — otherwise the submission is accepted, the drain
worker starts, and the test trains a real model.
"""

from __future__ import annotations

from typing import Any

from visionforge.gui.api.run_queue import QueuedJob


class _StuckWorker:
    """Stands in for the drain task so an occupied queue does not execute.

    ``RunQueue.submit`` only spawns a worker when none is running; presenting one
    that never finishes is what keeps a faked-busy queue from draining the job
    the test just submitted.
    """

    def done(self) -> bool:
        return False

    def cancel(self) -> None:  # called by RunQueue.reset
        return None


async def _never_runs() -> None:  # pragma: no cover - never awaited
    raise AssertionError("The placeholder job must not execute.")


def occupy_queue(routes_mod: Any, run_id: str = "busy_run") -> None:
    """Make the queue look like it is mid-run, without running anything."""
    queue = routes_mod._RUN_QUEUE
    queue._active = QueuedJob(
        run_id=run_id,
        label=run_id,
        task="classification",
        strategy="simple",
        start=_never_runs,
    )
    queue._worker = _StuckWorker()


def release_queue(routes_mod: Any) -> None:
    """Clear queue state between tests."""
    routes_mod._RUN_QUEUE.reset()
    routes_mod._current_run = None
