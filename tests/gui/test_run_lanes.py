"""Parallel queues (ADR-163, phase 4): one local, one per Google account."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from visionforge.gui.api import colab_jobs as cj
from visionforge.gui.api.run_queue import (
    LOCAL_LANE,
    NotStoppableError,
    QueuedJob,
    RunLanes,
    RunQueue,
    colab_lane,
)

from .conftest import release_queue

COLAB = colab_lane("ana@example.com")


def _job(run_id: str, start: Any, stop_at: Any = "epoch") -> QueuedJob:
    return QueuedJob(
        run_id=run_id,
        label=run_id,
        task="classification",
        strategy="simple",
        start=start,
        stop_at=stop_at,
    )


def _lanes(started: list[str]) -> RunLanes:
    def make() -> RunQueue:
        return RunQueue(
            on_start=lambda job: started.append(job.run_id),
            on_finish=lambda job: {"run_id": job.run_id, "status": "completed"},
        )

    return RunLanes(make(), make)


# ── the lanes on their own ───────────────────────────────────────────────────


def test_two_lanes_run_side_by_side_and_one_lane_runs_one_at_a_time() -> None:
    async def scenario() -> tuple[list[str], list[dict[str, Any]], list[str]]:
        started: list[str] = []
        lanes = _lanes(started)
        gate = asyncio.Event()

        async def hold() -> None:
            await gate.wait()

        assert lanes.submit(_job("local-1", hold)) == "running"
        assert lanes.submit(_job("colab-1", hold), COLAB) == "running"
        await asyncio.sleep(0.01)
        assert lanes.submit(_job("local-2", hold)) == "queued"
        await asyncio.sleep(0.01)
        during = lanes.snapshots()
        both = list(started)
        gate.set()
        await asyncio.sleep(0.02)
        return both, during, started

    both, during, started = asyncio.run(scenario())

    assert both == ["local-1", "colab-1"]
    assert [(s["lane"], s["active"]["run_id"]) for s in during] == [
        (LOCAL_LANE, "local-1"),
        (COLAB, "colab-1"),
    ]
    assert [j["run_id"] for j in during[0]["pending"]] == ["local-2"]
    assert during[1]["pending"] == []
    assert during[1]["active"]["lane"] == COLAB
    # local-2 waited for local-1 only, never for the Colab job's lane.
    assert started == ["local-1", "colab-1", "local-2"]


def test_cancel_reaches_only_the_job_it_names() -> None:
    async def scenario() -> tuple[QueuedJob, QueuedJob, RunLanes]:
        lanes = _lanes([])
        gate = asyncio.Event()

        async def hold() -> None:
            await gate.wait()

        local, colab = _job("local-1", hold), _job("colab-1", hold, stop_at="vm")
        lanes.submit(local)
        lanes.submit(colab, COLAB)
        await asyncio.sleep(0.01)
        assert lanes.cancel("colab-1") is True
        assert lanes.cancel("nobody") is False
        gate.set()
        await asyncio.sleep(0.02)
        return local, colab, lanes

    local, colab, lanes = asyncio.run(scenario())

    assert colab.cancel_token.cancelled is True
    assert local.cancel_token.cancelled is False
    # Each lane keeps the record of what it ran.
    assert lanes.record("colab-1") == {"run_id": "colab-1", "status": "completed"}
    assert lanes.record("local-1") == {"run_id": "local-1", "status": "completed"}


def test_a_job_that_cannot_stop_still_says_so_from_its_lane() -> None:
    async def scenario() -> None:
        lanes = _lanes([])
        gate = asyncio.Event()

        async def hold() -> None:
            await gate.wait()

        lanes.submit(_job("custom", hold, stop_at=None), COLAB)
        await asyncio.sleep(0.01)
        try:
            with pytest.raises(NotStoppableError):
                lanes.cancel("custom")
        finally:
            gate.set()
            await asyncio.sleep(0.01)

    asyncio.run(scenario())


def test_busy_and_pending_count_cover_every_lane() -> None:
    async def scenario() -> tuple[bool, int, bool]:
        lanes = _lanes([])
        gate = asyncio.Event()

        async def hold() -> None:
            await gate.wait()

        lanes.submit(_job("c1", hold), COLAB)
        lanes.submit(_job("c2", hold), COLAB)
        await asyncio.sleep(0.01)
        busy, pending = lanes.is_busy(), lanes.pending_count()
        gate.set()
        await asyncio.sleep(0.02)
        return busy, pending, lanes.is_busy()

    assert asyncio.run(scenario()) == (True, 1, False)


def test_reset_keeps_only_the_local_lane() -> None:
    lanes = _lanes([])
    lanes.lane(COLAB)
    lanes.reset()
    assert [s["lane"] for s in lanes.snapshots()] == [LOCAL_LANE]


# ── a local job and a Colab job at once, through the routes ──────────────────


def test_a_local_job_and_a_colab_job_run_at_once_each_with_its_own_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The phase-4 promise end to end: two jobs, two lanes, two streams, two states."""
    import visionforge.gui.api.routes as routes_mod

    colab_started = threading.Event()
    colab_may_end = threading.Event()

    class Jobs:
        def run_job(self, slug, accelerator, payload, on_event, cancelled):  # type: ignore[no-untyped-def]
            on_event({"event": "remote", "stage": "train"})
            colab_started.set()
            assert colab_may_end.wait(5)
            on_event({"event": "epoch_end", "epoch": 1, "where": "colab"})
            return cj.RemoteResult("completed", report={"where": "colab"})

    monkeypatch.setattr(routes_mod, "_COLAB_JOBS", Jobs())

    async def local_executor(run_id: str, gate: asyncio.Event) -> None:
        # The same shape as the real executors: its own state, found by run_id.
        state = routes_mod._job_state(run_id)
        await state.log.put({"event": "epoch_end", "epoch": 1, "where": "local"})
        await gate.wait()
        state.run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": {"where": "local"},
            "run_dir": None,
        }
        await state.log.put(None)

    async def read(run_id: str) -> list[dict[str, Any]]:
        lines = [line async for line in routes_mod._sse_generator(run_id)]
        return [json.loads(line[len("data: ") :]) for line in lines]

    async def scenario() -> dict[str, Any]:
        local_gate = asyncio.Event()
        local = routes_mod._submit_job(
            "local-run",
            "local",
            "classification",
            "simple",
            lambda: local_executor("local-run", local_gate),
        )
        colab = routes_mod._submit_job(
            "colab-run",
            "colab",
            "classification",
            "colab",
            lambda: routes_mod._execute_colab("colab-run", "ana", "T4", {}, tmp_path),
            stop_at="vm",
            lane=COLAB,
        )
        await asyncio.to_thread(colab_started.wait, 5)
        await asyncio.sleep(0.01)
        during = {
            "submitted": (local.status, colab.status),
            "lanes": [
                (s["lane"], (s["active"] or {}).get("run_id"))
                for s in routes_mod._LANES.snapshots()
            ],
            "statuses": (
                (await routes_mod.get_status("local-run")).status,
                (await routes_mod.get_status("colab-run")).status,
            ),
        }
        # Stopping the Colab job must leave the local one alone.
        assert routes_mod._LANES.cancel("colab-run") is True
        during["tokens"] = (
            routes_mod._job_state("local-run").token.cancelled,
            routes_mod._job_state("colab-run").token.cancelled,
        )
        colab_may_end.set()
        local_gate.set()
        streams = await asyncio.gather(read("local-run"), read("colab-run"))
        await asyncio.sleep(0.02)
        during["streams"] = streams
        during["results"] = (
            (await routes_mod.get_result("local-run")).report,
            (await routes_mod.get_result("colab-run")).report,
        )
        return during

    try:
        seen = asyncio.run(scenario())
    finally:
        colab_may_end.set()
        release_queue(routes_mod)

    assert seen["submitted"] == ("running", "running")
    assert seen["lanes"] == [(LOCAL_LANE, "local-run"), (COLAB, "colab-run")]
    assert seen["statuses"] == ("running", "running")
    assert seen["tokens"] == (False, True)
    local_events, colab_events = seen["streams"]
    assert local_events == [{"event": "epoch_end", "epoch": 1, "where": "local"}]
    assert colab_events == [
        {"event": "remote", "stage": "train"},
        {"event": "epoch_end", "epoch": 1, "where": "colab"},
    ]
    assert seen["results"] == ({"where": "local"}, {"where": "colab"})


def test_a_client_that_names_no_run_follows_the_job_started_last() -> None:
    import visionforge.gui.api.routes as routes_mod

    from .conftest import start_job

    async def scenario() -> tuple[str | None, list[str]]:
        start_job(routes_mod, "first")
        second = start_job(routes_mod, "second", lane=COLAB)
        await second.log.put({"event": "start"})
        await second.log.put(None)
        status = await routes_mod.get_status()
        lines = [line async for line in routes_mod._sse_generator()]
        return status.run_id, lines

    try:
        run_id, lines = asyncio.run(scenario())
    finally:
        release_queue(routes_mod)

    assert run_id == "second"
    assert lines == ['data: {"event": "start"}\n\n']


def test_finished_states_are_bounded_but_running_ones_are_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import visionforge.gui.api.routes as routes_mod

    from .conftest import start_job

    monkeypatch.setattr(routes_mod, "_MAX_FINISHED_STATES", 2)
    try:
        start_job(routes_mod, "running-long")
        for i in range(4):
            start_job(routes_mod, f"done-{i}").run["status"] = "completed"
        start_job(routes_mod, "trigger")
        assert list(routes_mod._JOB_STATES) == [
            "running-long",
            "done-2",
            "done-3",
            "trigger",
        ]
    finally:
        release_queue(routes_mod)
