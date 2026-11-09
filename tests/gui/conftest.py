"""Shared helpers for the route tests.

The run queue (ADR-075) replaced the old "409 while busy" contract, so the tests
that used to fake a busy server by assigning the route module's run state now need to
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


def occupy_queue(
    routes_mod: Any, run_id: str = "busy_run", lane: str = "local"
) -> None:
    """Make a queue (the local one by default) look mid-run, running nothing."""
    queue = routes_mod._LANES.lane(lane)
    queue._active = QueuedJob(
        run_id=run_id,
        label=run_id,
        task="classification",
        strategy="simple",
        start=_never_runs,
    )
    queue._worker = _StuckWorker()


def release_queue(routes_mod: Any) -> None:
    """Clear every queue and every job's state between tests (ADR-163, phase 4)."""
    routes_mod._LANES.reset()
    routes_mod._JOB_STATES.clear()
    routes_mod._latest_run_id = None


def start_job(
    routes_mod: Any,
    run_id: str,
    log: Any = None,
    lane: str = "local",
    cancel_token: Any = None,
) -> Any:
    """Register a job's state as the queue would, for a test that calls an executor.

    ``log`` replaces the fresh RunEventLog (a test's own subclass); returns the
    JobState, whose ``run`` the executor fills in.
    """
    job = QueuedJob(
        run_id=run_id,
        label=run_id,
        task="classification",
        strategy="simple",
        start=_never_runs,
    )
    job.lane = lane
    if cancel_token is not None:
        job.cancel_token = cancel_token
    routes_mod._begin_job(job)
    state = routes_mod._JOB_STATES[run_id]
    if log is not None:
        state.log = log
    return state


def fake_run(routes_mod: Any, run: dict[str, Any]) -> Any:
    """A started job whose state is ``run``, as if its executor had written it."""
    state = start_job(routes_mod, run["run_id"])
    state.run = dict(run)
    return state


def job_run(routes_mod: Any, run_id: str | None = None) -> Any:
    """The run dict of a started job: the one named, or the latest."""
    return routes_mod._run_state(run_id)


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
def _testclient_host_is_this_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let ``TestClient``'s default ``Host: testserver`` through the Host guard.

    The guard (``gui/security.py``) answers only to loopback names unless
    ``start_server`` installs another policy; ``testserver`` is what every client
    here sends. Only the fallback policy changes: ``test_host_origin_guard`` sets
    its own on the app and checks the real loopback-only default.
    """
    from visionforge.gui import security

    monkeypatch.setattr(
        security, "DEFAULT_POLICY", security.HostPolicy(extra=frozenset({"testserver"}))
    )


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
