"""Server mode: a bound on the heavy actions that run outside the queue (ADR-226).

Evaluating a run, downloading or auditing a dataset and publishing do their
work inside the request. ``--max-actions`` bounds how many run at once across
everybody; the next one is told the server is busy and nothing waits in line.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from visionforge.gui import server_mode
from visionforge.gui.action_limit import (
    DEFAULT_MAX_ACTIONS,
    HEAVY_ROUTES,
    ActionSlots,
    heavy,
)
from visionforge.gui.api import evidence_routes, routes
from visionforge.gui.api.sample_dataset import create_sample_dataset
from visionforge.gui.server_mode import (
    ROUTE_CLASSES,
    ServerMode,
    ServerModeError,
    answers,
)

PROXY = "10.0.0.5"
ANA = {"X-Forwarded-User": "ana@lab.org"}
BIA = {"X-Forwarded-User": "bia@lab.org"}


def test_the_option_needs_server_mode_and_one_or_more() -> None:
    with pytest.raises(ServerModeError, match="--max-actions"):
        ServerMode.from_options(enabled=False, max_actions=3)
    for bad in (0, -2):
        with pytest.raises(ServerModeError, match="1 or more"):
            ServerMode.from_options(
                enabled=True, trusted_proxies=[PROXY], max_actions=bad
            )
    chosen = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY], max_actions=5
    )
    assert chosen.max_actions == 5
    default = ServerMode.from_options(enabled=True, trusted_proxies=[PROXY])
    assert default.max_actions == DEFAULT_MAX_ACTIONS == 2


def test_every_heavy_route_is_one_that_answers_on_a_server() -> None:
    for key in HEAVY_ROUTES:
        assert key in ROUTE_CLASSES, key
        assert answers(key), key
    assert heavy("POST /api/runs/{run_id}/gradcam")
    assert heavy("POST /api/dataset/download")
    # A training waits in the queue, and reading is light: neither counts.
    for key in (
        "POST /api/experiment/run",
        "GET /api/runs",
        "POST /api/dataset/stats",
        "DELETE /api/runs/{run_id}",
        "POST /api/explorer/resplit/{job_id}/apply",
    ):
        assert not heavy(key)


def test_slots_count_what_runs_and_never_go_below_zero() -> None:
    slots = ActionSlots()
    assert slots.take(2)
    assert slots.take(2)
    assert not slots.take(2)
    assert slots.busy == 2
    slots.give()
    assert slots.take(2)
    for _ in range(5):
        slots.give()
    assert slots.busy == 0


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    shared = (tmp_path / "shared").resolve()
    return {
        "shared": shared,
        "beans": create_sample_dataset(shared, "classification").path,
    }


@pytest.fixture
def client(
    world: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """The real app in server mode, one heavy action at a time."""
    from visionforge.gui.server import app

    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(server_mode, "SLOTS", ActionSlots())
    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    app.state.server_mode = ServerMode.from_options(
        enabled=True,
        trusted_proxies=[PROXY],
        datasets_dirs=[world["shared"]],
        max_actions=1,
    )
    yield TestClient(app, base_url="http://localhost", client=(PROXY, 50000))
    if had:
        app.state.server_mode = saved
    else:
        del app.state.server_mode


def test_a_second_heavy_action_is_told_the_server_is_busy(
    client: TestClient, world: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    release = threading.Event()
    real_audit = evidence_routes.audit_dataset

    def slow_audit(*args: Any, **kwargs: Any) -> Any:
        started.set()
        assert release.wait(timeout=30)
        return real_audit(*args, **kwargs)

    monkeypatch.setattr(evidence_routes, "audit_dataset", slow_audit)
    body = {"base_dir": str(world["beans"])}
    first: dict[str, Any] = {}

    def ask() -> None:
        first["response"] = client.post("/api/dataset/audit", json=body, headers=ANA)

    thread = threading.Thread(target=ask)
    thread.start()
    try:
        assert started.wait(timeout=30)
        # Ana's audit holds the one slot: Bia's is refused, in her language.
        busy = client.post("/api/dataset/audit", json=body, headers=BIA)
        assert busy.status_code == 429, busy.text
        assert "ocupado" in busy.json()["detail"]
        english = client.post(
            "/api/dataset/audit", json=body, headers={**BIA, "X-VF-Lang": "en"}
        )
        assert english.status_code == 429
        assert "busy" in english.json()["detail"]
        # Another kind of heavy action shares the same bound.
        other = client.post("/api/runs/some_run/gradcam", json={}, headers=BIA)
        assert other.status_code == 429
        # What is light, or waits in the queue, is not held back.
        assert client.get("/api/runs", headers=BIA).status_code == 200
        stats = client.post("/api/dataset/stats", json=body, headers=BIA)
        assert stats.status_code == 200, stats.text
    finally:
        release.set()
        thread.join(timeout=30)
    assert first["response"].status_code == 200, first["response"].text

    # The slot is free again once the first action ends.
    after = client.post("/api/dataset/audit", json=body, headers=BIA)
    assert after.status_code == 200, after.text
    assert server_mode.SLOTS.busy == 0


def test_an_action_that_fails_gives_its_slot_back(client: TestClient) -> None:
    for _ in range(3):
        # No such run: the handler answers 404 after taking the slot.
        missing = client.post("/api/runs/no_such_run/gradcam", json={}, headers=ANA)
        assert missing.status_code in (404, 422), missing.text
    assert server_mode.SLOTS.busy == 0
