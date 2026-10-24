"""Shared helpers for the route tests.

The run queue (ADR-075) replaced the old "409 while busy" contract, so the tests
that used to fake a busy server by assigning ``routes._current_run`` now need to
make the *queue* busy instead — otherwise the submission is accepted, the drain
worker starts, and the test trains a real model.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

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


@pytest.fixture
def live_client() -> Iterator[TestClient]:
    """Client whose event loop outlives each request, for tests that await a job.

    Outside ``with``, ``TestClient`` runs every request on a loop of its own and
    closes it once the response is back. The queue's drain task lives on the
    submitting request's loop, so closing it cancels the drain, and a
    ``to_thread(block.run)`` that no pool thread has picked up yet is cancelled
    with it: ``setup()`` ran, ``run()`` never does (#77).
    """
    from visionforge.gui.server import app

    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


@pytest.fixture(autouse=True)
def _drop_queue_leftovers() -> Iterator[None]:
    """Clear the shared run queue after every test.

    A per-request loop that closes before its drain task first runs leaves the
    job in ``_pending``; the next test's submit would then run it first, with
    the real executor its fakes no longer cover.
    """
    yield
    routes_mod = sys.modules.get("visionforge.gui.api.routes")
    if routes_mod is not None:
        release_queue(routes_mod)
