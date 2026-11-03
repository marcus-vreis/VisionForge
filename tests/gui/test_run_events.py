"""The run's event log serves every reader from the start (ADR-159)."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import visionforge.gui.api.routes as routes_mod
from visionforge.gui.api.run_events import RunEventLog
from visionforge.gui.server import app


def _closed_log(run_id: str, count: int) -> RunEventLog:
    log = RunEventLog(run_id)

    async def fill() -> None:
        for epoch in range(count):
            await log.put({"event": "epoch_end", "epoch": epoch})
        await log.put(None)

    asyncio.run(fill())
    return log


async def _collect(log: RunEventLog, after: int = 0) -> list[int]:
    return [item["epoch"] async for item in log.read(after)]


def test_two_readers_each_get_every_event() -> None:
    log = _closed_log("r1", 3)
    assert asyncio.run(_collect(log)) == [0, 1, 2]
    assert asyncio.run(_collect(log)) == [0, 1, 2]


def test_after_skips_what_the_reader_already_has() -> None:
    log = _closed_log("r1", 3)
    assert asyncio.run(_collect(log, after=2)) == [2]
    assert asyncio.run(_collect(log, after=99)) == []


def test_a_waiting_reader_gets_events_put_later() -> None:
    async def scenario() -> list[int]:
        log = RunEventLog("r1")
        reader = asyncio.create_task(_collect(log))
        await asyncio.sleep(0)
        await log.put({"event": "epoch_end", "epoch": 7})
        await asyncio.sleep(0)
        await log.put(None)
        return await asyncio.wait_for(reader, timeout=2)

    assert asyncio.run(scenario()) == [7]


def _stream(client: TestClient, query: str) -> list[dict[str, object]]:
    with client.stream("GET", f"/api/experiment/events{query}") as resp:
        return [
            json.loads(line[len("data: ") :])
            for line in resp.iter_lines()
            if line.startswith("data: ")
        ]


def test_route_replays_the_run_and_refuses_another_runs_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes_mod, "_event_queue", _closed_log("r1", 2))
    client = TestClient(app)
    assert [e["epoch"] for e in _stream(client, "")] == [0, 1]
    assert [e["epoch"] for e in _stream(client, "?run_id=r1&after=1")] == [1]
    assert _stream(client, "?run_id=outro") == [{}]


def test_route_rejects_a_negative_after() -> None:
    assert TestClient(app).get("/api/experiment/events?after=-1").status_code == 422
