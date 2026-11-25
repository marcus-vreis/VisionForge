"""Server mode: each person's share of the queue, and turns between people (ADR-245).

A person may have at most ``--max-queued`` jobs waiting or running; the next
one is refused with 429 before anything is queued. When the GPU frees, the
next job is the oldest one of the person whose job started least recently. A
local install keeps the plain FIFO and no limit.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from visionforge.gui.api import execution, routes, submission, training_routes
from visionforge.gui.api.execution_routes import _lane_for
from visionforge.gui.api.profiles import ensure_profile
from visionforge.gui.api.run_queue import DEFAULT_MAX_QUEUED, QueuedJob, RunQueue
from visionforge.gui.server_mode import ServerMode, ServerModeError, user_slug
from visionforge.utils.messages import LANG_HEADER, message

PROXY = "10.0.0.5"
ANA = {"X-Forwarded-User": "ana@lab.org"}
BIA = {"X-Forwarded-User": "bia@lab.org"}
ANA_SLUG = user_slug("ana@lab.org")
BIA_SLUG = user_slug("bia@lab.org")
WAIT_S = 10.0


async def _nothing() -> None:
    """A job body that is never awaited: the unit tests place jobs by hand."""


def _job(run_id: str, profile: str) -> QueuedJob:
    return QueuedJob(
        run_id=run_id,
        label=run_id,
        task="classification",
        strategy="simple",
        start=_nothing,
        profile=profile,
        profile_name=profile,
    )


def _queue(fair: bool, *jobs: QueuedJob) -> RunQueue:
    queue = RunQueue(on_start=lambda job: None, on_finish=lambda job: None)
    queue._fair = fair
    queue._pending.extend(jobs)
    return queue


def _ids(jobs: list[QueuedJob]) -> list[str]:
    return [job.run_id for job in jobs]


# ── the order ────────────────────────────────────────────────────────────────


def test_a_local_lane_runs_in_submission_order() -> None:
    queue = _queue(False, _job("a2", "a"), _job("a3", "a"), _job("b1", "b"))
    queue._last_start["a"] = 1
    assert _ids(queue.run_order()) == ["a2", "a3", "b1"]
    assert _ids([queue._next() for _ in range(3)]) == ["a2", "a3", "b1"]


def test_a_fair_lane_takes_turns_between_people() -> None:
    queue = _queue(
        True,
        _job("a1", "a"),
        _job("a2", "a"),
        _job("a3", "a"),
        _job("b1", "b"),
        _job("c1", "c"),
        _job("b2", "b"),
    )
    expected = ["a1", "b1", "c1", "a2", "b2", "a3"]
    assert _ids(queue.run_order()) == expected
    assert _ids([queue._next() for _ in range(6)]) == expected
    assert queue.pending_count() == 0


def test_a_person_who_just_ran_waits_behind_who_did_not() -> None:
    queue = _queue(True, _job("a2", "a"), _job("b1", "b"))
    queue._starts = 7
    queue._last_start.update({"a": 7, "b": 3})
    assert _ids(queue.run_order()) == ["b1", "a2"]
    queue._last_start["b"] = 9
    assert _ids(queue.run_order()) == ["a2", "b1"]


def test_the_queue_shows_the_order_it_will_run_and_counts_who_is_ahead() -> None:
    queue = _queue(True, _job("a2", "a"), _job("a3", "a"), _job("b1", "b"))
    queue._active = _job("a1", "a")
    queue._starts = 1
    queue._last_start["a"] = 1
    snap = {"lane": "local", **queue.snapshot()}
    assert [job["run_id"] for job in snap["pending"]] == ["b1", "a2", "a3"]
    assert _lane_for("b", snap).others_ahead == 1
    assert _lane_for("c", snap).others_ahead == 4
    mine = _lane_for("a", snap)
    assert mine.active is not None
    assert (_ids_of(mine.pending), mine.others_ahead) == (["a2", "a3"], 0)
    # A submission keeps the deque in arrival order; only the start order changes.
    assert _ids(list(queue._pending)) == ["a2", "a3", "b1"]
    assert queue.count_of("a") == 3
    assert queue.count_of("b") == 1


def _ids_of(jobs: list[Any]) -> list[str]:
    return [job.run_id for job in jobs]


def test_reset_forgets_fairness_and_turns() -> None:
    queue = _queue(True, _job("a1", "a"))
    queue._next()
    queue.reset()
    assert (queue._fair, queue._starts, queue._last_start) == (False, 0, {})


# ── the option ───────────────────────────────────────────────────────────────


def test_the_option_needs_server_mode_and_one_or_more() -> None:
    with pytest.raises(ServerModeError, match="--max-queued"):
        ServerMode.from_options(enabled=False, max_queued=3)
    for bad in (0, -1):
        with pytest.raises(ServerModeError, match="1 or more"):
            ServerMode.from_options(
                enabled=True, trusted_proxies=[PROXY], max_queued=bad
            )
    chosen = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY], max_queued=5
    )
    assert chosen.max_queued == 5
    default = ServerMode.from_options(enabled=True, trusted_proxies=[PROXY])
    assert default.max_queued == DEFAULT_MAX_QUEUED == 3


def test_the_command_line_hands_the_option_over() -> None:
    from visionforge.__main__ import _build_parser

    args = _build_parser().parse_args(
        ["gui", "--server", "--trusted-proxy", PROXY, "--max-queued", "4"]
    )
    assert args.max_queued == 4
    assert _build_parser().parse_args(["gui"]).max_queued is None


# ── the real app ─────────────────────────────────────────────────────────────


class _Trainer:
    """Stands in for the classification executor: notes each start, then holds."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.release = threading.Event()

    async def __call__(self, config: Any, run_id: str, *_: Any, **__: Any) -> None:
        self.started.append(config.name)
        execution._job_state(run_id).run["status"] = "running"
        while not self.release.is_set():
            await asyncio.sleep(0.005)
        execution._job_state(run_id).run["status"] = "completed"

    def wait_for(self, count: int) -> None:
        deadline = time.monotonic() + WAIT_S
        while len(self.started) < count:
            assert time.monotonic() < deadline, f"started only {self.started}"
            time.sleep(0.01)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """A shared dataset, a fake trainer and an empty queue; the app is not served yet."""
    shared = tmp_path / "shared"
    (shared / "beans" / "train" / "healthy").mkdir(parents=True)
    trainer = _Trainer()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(execution, "_JOB_STATES", {})
    monkeypatch.setattr(execution, "_latest_run_id", None)
    monkeypatch.setattr(execution, "_execute_experiment", trainer)
    monkeypatch.setattr(submission, "cuda_context_broken", lambda: False)
    monkeypatch.setattr(submission, "_TRAINING_STACK", ())
    monkeypatch.setattr(
        training_routes, "_check_classification_classes", lambda config: None
    )
    execution._LANES.reset()
    yield {"shared": shared.resolve(), "trainer": trainer, "profiles": tmp_path}
    trainer.release.set()
    execution._LANES.reset()


@pytest.fixture
def serve(world: dict[str, Any]) -> Iterator[Callable[[ServerMode], TestClient]]:
    """Open the real app in ``mode`` with one loop for every request."""
    from visionforge.gui.server import app

    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    clients: list[TestClient] = []

    def open_with(mode: ServerMode) -> TestClient:
        app.state.server_mode = mode
        client = TestClient(app, base_url="http://localhost", client=(PROXY, 50000))
        clients.append(client)
        return client.__enter__()

    yield open_with
    world["trainer"].release.set()
    for client in clients:
        client.__exit__(None, None, None)
    if had:
        app.state.server_mode = saved
    else:
        del app.state.server_mode


def _server(world: dict[str, Any], **options: Any) -> ServerMode:
    return ServerMode.from_options(
        enabled=True,
        trusted_proxies=[PROXY],
        datasets_dirs=[world["shared"]],
        **options,
    )


def _submit(
    client: TestClient, world: dict[str, Any], name: str, headers: dict[str, str]
) -> Any:
    body = {
        "name": name,
        "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
        "data": {"base_dir": str(world["shared"] / "beans")},
    }
    return client.post("/api/experiment/run", json=body, headers=headers)


def _wait_idle(trainer: _Trainer, count: int) -> None:
    trainer.release.set()
    trainer.wait_for(count)
    deadline = time.monotonic() + WAIT_S
    while execution._LANES.is_busy():
        assert time.monotonic() < deadline, "the queue never drained"
        time.sleep(0.01)


def test_a_person_with_a_batch_does_not_hold_the_next_person_behind_it(
    serve: Callable[[ServerMode], TestClient], world: dict[str, Any]
) -> None:
    client = serve(_server(world))
    trainer: _Trainer = world["trainer"]
    assert _submit(client, world, "a1", ANA).json()["status"] == "running"
    trainer.wait_for(1)
    for name, who in (("a2", ANA), ("a3", ANA), ("b1", BIA)):
        response = _submit(client, world, name, who)
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "queued"

    bia = client.get("/api/queue", headers=BIA).json()
    assert [job["label"] for job in bia["pending"]] == ["b1"]
    assert bia["others_ahead"] == 1
    ana = client.get("/api/queue", headers=ANA).json()
    assert ana["active"]["label"] == "a1"
    assert [job["label"] for job in ana["pending"]] == ["a2", "a3"]
    assert ana["others_ahead"] == 0
    assert "b1" not in client.get("/api/queue", headers=ANA).text

    _wait_idle(trainer, 4)
    assert trainer.started == ["a1", "b1", "a2", "a3"]


def test_a_person_past_the_limit_is_refused_before_anything_is_queued(
    serve: Callable[[ServerMode], TestClient], world: dict[str, Any]
) -> None:
    client = serve(_server(world, max_queued=2))
    trainer: _Trainer = world["trainer"]
    assert _submit(client, world, "a1", ANA).status_code == 200
    trainer.wait_for(1)
    assert _submit(client, world, "a2", ANA).status_code == 200

    refused = _submit(client, world, "a3", {**ANA, LANG_HEADER: "en"})
    assert refused.status_code == 429
    assert refused.json()["detail"] == message("en", "server.queue_full", limit=2)
    in_pt = _submit(client, world, "a3", {**ANA, LANG_HEADER: "pt"})
    assert in_pt.json()["detail"] == message("pt", "server.queue_full", limit=2)
    assert execution._LANES.count_of(ANA_SLUG) == 2
    assert execution._LANES.pending_count() == 1

    # The limit is each person's: another one still gets in line.
    assert _submit(client, world, "b1", BIA).status_code == 200
    assert execution._LANES.count_of(BIA_SLUG) == 1
    assert client.get("/api/session", headers=ANA).json()["max_queued"] == 2

    # Once a job ends there is room again.
    _wait_idle(trainer, 3)
    assert _submit(client, world, "a4", ANA).status_code == 200


def test_a_local_install_keeps_fifo_and_no_limit(
    serve: Callable[[ServerMode], TestClient], world: dict[str, Any]
) -> None:
    client = serve(ServerMode())
    trainer: _Trainer = world["trainer"]
    ensure_profile("bia", "bia", world["profiles"] / "profiles")
    names = ["a1", "a2", "a3", "a4", "b1"]
    for index, name in enumerate(names):
        profile = {"X-VF-Profile": "bia"} if name.startswith("b") else {}
        response = _submit(client, world, name, profile)
        assert response.status_code == 200, response.text
        if index == 0:
            trainer.wait_for(1)
    assert client.get("/api/session").json()["max_queued"] is None
    _wait_idle(trainer, 5)
    assert trainer.started == names
