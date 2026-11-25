"""Server mode: the queue and a run's progress belong to who started it (ADR-215).

A person sees their own jobs in full and, of the others, only how many are
ahead; another person's run answers as a run that does not exist.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from visionforge.gui.api import execution, routes
from visionforge.gui.api.execution_routes import _lane_for
from visionforge.gui.api.run_queue import QueuedJob
from visionforge.gui.server_mode import ServerMode, user_slug

PROXY = "10.0.0.5"
ANA = {"X-Forwarded-User": "ana@lab.org"}
BIA = {"X-Forwarded-User": "bia@lab.org"}
ANA_SLUG = user_slug("ana@lab.org")
BIA_SLUG = user_slug("bia@lab.org")


async def _nothing() -> None:
    """A job body that is never awaited: these tests place jobs by hand."""


def _job(run_id: str, profile: str) -> QueuedJob:
    return QueuedJob(
        run_id=run_id,
        label=f"exp-{run_id}",
        task="classification",
        strategy="simple",
        start=_nothing,
        profile=profile,
        profile_name=profile,
    )


def _snap(active: QueuedJob | None, *pending: QueuedJob) -> dict[str, Any]:
    return {
        "lane": "local",
        "active": active.describe() if active else None,
        "pending": [job.describe() for job in pending],
    }


def test_a_local_install_sees_the_whole_queue() -> None:
    lane = _lane_for(None, _snap(_job("r1", "a"), _job("r2", "b"), _job("r3", "a")))
    assert lane.active is not None
    assert lane.active.run_id == "r1"
    assert [job.run_id for job in lane.pending] == ["r2", "r3"]
    assert lane.others_ahead == 0


def test_a_person_sees_their_jobs_and_a_count_of_those_ahead() -> None:
    snap = _snap(
        _job("r1", "bia"), _job("r2", "bia"), _job("r3", "ana"), _job("r4", "bia")
    )
    lane = _lane_for("ana", snap)
    assert lane.active is None
    assert [job.run_id for job in lane.pending] == ["r3"]
    assert lane.others_ahead == 2


def test_the_running_job_of_who_asks_has_nobody_ahead() -> None:
    lane = _lane_for(
        "ana", _snap(_job("r1", "ana"), _job("r2", "bia"), _job("r3", "ana"))
    )
    assert lane.active is not None
    assert lane.active.run_id == "r1"
    assert [job.run_id for job in lane.pending] == ["r3"]
    assert lane.others_ahead == 0


def test_with_no_job_of_their_own_a_person_sees_how_busy_the_queue_is() -> None:
    lane = _lane_for("ana", _snap(_job("r1", "bia"), _job("r2", "caio")))
    assert (lane.active, lane.pending, lane.others_ahead) == (None, [], 2)
    assert _lane_for("ana", _snap(None)).others_ahead == 0


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """The real app in server mode with an empty queue; a client at the proxy."""
    from visionforge.gui.server import app

    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(execution, "_JOB_STATES", {})
    monkeypatch.setattr(execution, "_latest_run_id", None)
    execution._LANES.reset()
    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    app.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY]
    )
    yield TestClient(app, base_url="http://localhost", client=(PROXY, 50000))
    execution._LANES.reset()
    if had:
        app.state.server_mode = saved
    else:
        del app.state.server_mode


def _started(run_id: str, profile: str) -> QueuedJob:
    job = _job(run_id, profile)
    execution._begin_job(job)
    return job


def test_whose_a_job_is_survives_its_state(served: TestClient) -> None:
    job = _started("r1", BIA_SLUG)
    assert execution.profile_of_run("r1") == BIA_SLUG
    record = execution._finish_job(job)
    assert record is not None
    assert record["profile"] == BIA_SLUG
    assert execution.profile_of_run("nope") is None
    assert execution.latest_run_of(BIA_SLUG) == "r1"
    assert execution.latest_run_of(ANA_SLUG) is None


def test_the_status_without_an_id_is_the_persons_own_latest_run(
    served: TestClient,
) -> None:
    _started("ana-1", ANA_SLUG)
    _started("bia-1", BIA_SLUG)
    assert served.get("/api/experiment/status", headers=ANA).json()["run_id"] == "ana-1"
    assert served.get("/api/experiment/status", headers=BIA).json()["run_id"] == "bia-1"
    caio = {"X-Forwarded-User": "caio@lab.org"}
    assert served.get("/api/experiment/status", headers=caio).json()["status"] == "idle"


def test_another_persons_run_answers_as_one_that_does_not_exist(
    served: TestClient,
) -> None:
    job = _started("bia-1", BIA_SLUG)
    execution._job_state("bia-1").run["status"] = "completed"
    execution._LANES.lane("local")._records["bia-1"] = execution._finish_job(job) or {}

    status = served.get("/api/experiment/status?run_id=bia-1", headers=ANA).json()
    assert (status["status"], status["run_id"]) == ("idle", None)
    assert served.get("/api/experiment/result/bia-1", headers=ANA).status_code == 404
    assert served.delete("/api/queue/bia-1", headers=ANA).status_code == 404
    events = served.get("/api/experiment/events?run_id=bia-1", headers=ANA)
    assert events.text.strip() == "data: {}"

    assert served.get("/api/experiment/result/bia-1", headers=BIA).status_code == 200
    mine = served.get("/api/experiment/status?run_id=bia-1", headers=BIA).json()
    assert mine["run_id"] == "bia-1"


def test_a_person_cannot_drop_a_job_that_waits_for_someone_else(
    served: TestClient,
) -> None:
    local = execution._LANES.lane("local")
    local._pending.extend([_job("bia-2", BIA_SLUG), _job("ana-2", ANA_SLUG)])

    assert served.delete("/api/queue/bia-2", headers=ANA).status_code == 404
    assert [job.run_id for job in local._pending] == ["bia-2", "ana-2"]

    queue = served.get("/api/queue", headers=ANA).json()
    assert [job["run_id"] for job in queue["pending"]] == ["ana-2"]
    assert queue["others_ahead"] == 1
    assert "bia" not in served.get("/api/queue", headers=ANA).text

    assert served.delete("/api/queue/ana-2", headers=ANA).status_code == 200
    assert [job.run_id for job in local._pending] == ["bia-2"]


def test_a_person_is_served_only_the_files_of_their_own_runs(
    served: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles = tmp_path / "outputs" / "profiles"
    monkeypatch.setattr(routes, "_PROFILES_DIR", profiles)
    monkeypatch.chdir(tmp_path)
    served.get("/api/session", headers=ANA)
    served.get("/api/session", headers=BIA)
    mine = profiles / ANA_SLUG / "graphics" / "curve.png"
    theirs = profiles / BIA_SLUG / "graphics" / "curve.png"
    shared = tmp_path / "outputs" / "preview_cache" / "x.png"
    for file in (mine, theirs, shared):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"png")

    def get(file: Path) -> int:
        relative = file.relative_to(tmp_path).as_posix()
        return int(served.get(f"/api/artifacts/{relative}", headers=ANA).status_code)

    assert (get(mine), get(theirs), get(shared)) == (200, 403, 403)
