"""Server mode: the dataset Laboratory, for several people (ADR-220).

It reads a dataset in the person's scope, its background jobs belong to whoever
started them, and a derived dataset is written into the person's own folder:
next to its source when the source is theirs, copied into their folder when
the source is a shared one.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from visionforge.gui.api import routes
from visionforge.gui.api.dataset_scope import DatasetScope, dataset_scope
from visionforge.gui.api.derived_jobs import (
    DerivedJobs,
    JobBusyError,
    copy_only,
    destination_parent,
    placement,
)
from visionforge.gui.api.sample_dataset import create_sample_dataset
from visionforge.gui.server_mode import READY, ServerMode, user_slug

PROXY = "10.0.0.5"
ANA = {"X-Forwarded-User": "ana@lab.org"}
BIA = {"X-Forwarded-User": "bia@lab.org"}


def _as(owner: str, own: Path | None = None) -> DatasetScope:
    return DatasetScope(readable=(own,) if own else (), owner=owner, own=own)


def test_a_job_is_seen_only_by_who_started_it() -> None:
    jobs = DerivedJobs("merge")
    with dataset_scope(_as("ana")):
        job = jobs.register("preview")
        assert jobs.get(job.job_id) is job
    with dataset_scope(_as("bia")):
        assert jobs.get(job.job_id) is None
        assert jobs.cancel(job.job_id) is False
    assert jobs.get(job.job_id) is None  # nor by a request with no account
    assert not job.token.cancelled
    with dataset_scope(_as("ana")):
        assert jobs.cancel(job.job_id) is True
    assert job.token.cancelled


def test_one_persons_running_job_does_not_block_another() -> None:
    jobs = DerivedJobs("merge")
    with dataset_scope(_as("ana")):
        jobs.register("preview")
        with pytest.raises(JobBusyError):
            jobs.register("preview")
    with dataset_scope(_as("bia")):
        jobs.register("preview")


def test_someone_elses_jobs_never_push_out_a_preview_about_to_be_applied() -> None:
    jobs = DerivedJobs("merge")
    with dataset_scope(_as("ana")):
        mine = jobs.register("preview")
        mine.state = "done"
    with dataset_scope(_as("bia")):
        for _ in range(12):
            jobs.register("preview").state = "done"
    with dataset_scope(_as("ana")):
        assert jobs.get(mine.job_id) is mine
        later = []
        for _ in range(6):
            job = jobs.register("preview")
            job.state = "done"
            later.append(job)
        assert jobs.get(mine.job_id) is None  # her own oldest goes first
        assert jobs.get(later[-1].job_id) is later[-1]


def test_on_a_local_install_jobs_have_no_owner() -> None:
    jobs = DerivedJobs("merge")
    job = jobs.register("preview")
    assert (job.owner, jobs.get(job.job_id)) == ("", job)
    with pytest.raises(JobBusyError):
        jobs.register("apply")


def test_where_a_derived_dataset_is_written(tmp_path: Path) -> None:
    own = (tmp_path / "ana" / "datasets").resolve()
    shared = (tmp_path / "shared").resolve()
    (own / "mine").mkdir(parents=True)
    (shared / "beans").mkdir(parents=True)

    # Local: next to the source, hard links allowed (the build's own default).
    assert placement(shared / "beans") == {}
    assert destination_parent(shared / "beans") == shared

    with dataset_scope(_as("ana", own)):
        # Her own dataset: next to it, which is inside her folder.
        assert placement(own / "mine") == {}
        assert destination_parent(own / "mine") == own
        # A shared one: into her folder, copied.
        assert placement(shared / "beans") == {"parent": own, "link": copy_only}
        assert destination_parent(shared / "beans") == own
    with pytest.raises(OSError):
        copy_only(shared / "a", own / "a")

    with dataset_scope(_as("ana")), pytest.raises(HTTPException) as refusal:
        placement(shared / "beans")  # a scope with no folder to write in
    assert refusal.value.status_code == 403


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    shared = (tmp_path / "shared").resolve()
    beans = create_sample_dataset(shared, "classification").path
    secret = (tmp_path / "secret").resolve()
    hidden = create_sample_dataset(secret, "classification").path
    own = (tmp_path / "profiles" / user_slug("ana@lab.org") / "datasets").resolve()
    return {"shared": shared, "beans": beans, "hidden": hidden, "own": own}


@pytest.fixture
def client(
    world: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """The real app in server mode, over real (tiny) datasets."""
    from visionforge.gui.server import app

    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    app.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY], datasets_dirs=[world["shared"]]
    )
    yield TestClient(app, base_url="http://localhost", client=(PROXY, 50000))
    if had:
        app.state.server_mode = saved
    else:
        del app.state.server_mode


def _finished(client: TestClient, url: str, who: dict[str, str]) -> dict[str, Any]:
    """Poll a job until it is no longer running."""
    for _ in range(200):
        body: dict[str, Any] = client.get(url, headers=who).json()
        if body.get("state") != "running":
            return body
        time.sleep(0.05)
    raise AssertionError(f"{url} never finished")


def test_a_shared_dataset_can_be_explored(
    client: TestClient, world: dict[str, Path]
) -> None:
    opened = client.post(
        "/api/explorer/summary", json={"base_dir": str(world["beans"])}, headers=ANA
    )
    assert opened.status_code == 200, opened.text
    page = client.post(
        "/api/explorer/images",
        json={"base_dir": str(world["beans"]), "limit": 1},
        headers=ANA,
    ).json()
    assert page["total"] > 0
    thumb = client.get(
        "/api/explorer/thumb",
        params={"root": str(world["beans"]), "path": page["items"][0]["path"]},
        headers=ANA,
    )
    assert (thumb.status_code, thumb.headers["content-type"]) == (200, "image/jpeg")


def test_a_dataset_outside_the_scope_cannot_be_explored(
    client: TestClient, world: dict[str, Path]
) -> None:
    outside = {"base_dir": str(world["hidden"])}
    for route in ("summary", "images", "scan", "resplit/preview"):
        refused = client.post(f"/api/explorer/{route}", json=outside, headers=ANA)
        assert refused.status_code == 422, route
    climbing = {"base_dir": str(world["beans"]), "train_dir": "../../secret"}
    assert (
        client.post("/api/explorer/summary", json=climbing, headers=ANA).status_code
        == 422
    )
    thumb = client.get(
        "/api/explorer/thumb",
        params={"root": str(world["hidden"]), "path": "train/x.png"},
        headers=ANA,
    )
    assert thumb.status_code == 403
    for body in (
        {"base_dir": str(world["beans"]), "other_dir": str(world["hidden"])},
        {"base_dir": str(world["beans"]), "other_dir": "../secret"},
    ):
        merge = client.post("/api/explorer/merge/preview", json=body, headers=ANA)
        assert merge.status_code == 422
    for route, body in (
        ("incorporate/preview", {"base_dir": str(world["beans"])}),
        ("coco/import/preview", {}),
    ):
        refused = client.post(
            f"/api/explorer/{route}",
            json={**body, "source_dir": str(world["hidden"])},
            headers=ANA,
        )
        assert refused.status_code == 422, route


def test_a_scan_is_its_owners(client: TestClient, world: dict[str, Path]) -> None:
    started = client.post(
        "/api/explorer/scan", json={"base_dir": str(world["beans"])}, headers=ANA
    )
    assert started.status_code == 200, started.text
    url = f"/api/explorer/scan/{started.json()['job_id']}"
    assert client.get(url, headers=BIA).status_code == 404
    assert client.post(f"{url}/cancel", headers=BIA).status_code == 404
    assert _finished(client, url, ANA)["state"] == "done"


def test_re_splitting_a_shared_dataset_writes_a_copy_into_the_own_folder(
    client: TestClient, world: dict[str, Path]
) -> None:
    before = sorted(p.name for p in world["shared"].iterdir())
    preview = client.post(
        "/api/explorer/resplit/preview",
        json={"base_dir": str(world["beans"])},
        headers=ANA,
    )
    assert preview.status_code == 200, preview.text
    job = f"/api/explorer/resplit/{preview.json()['job_id']}"
    assert _finished(client, job, ANA)["state"] == "done"

    # Nobody else can read, plan, apply or cancel her preview.
    assert client.get(job, headers=BIA).status_code == 404
    for action, body in (("plan", {}), ("apply", {"dest_name": "x"}), ("cancel", None)):
        theirs = client.post(f"{job}/{action}", json=body, headers=BIA)
        assert theirs.status_code == 404, action

    applied = client.post(f"{job}/apply", json={"dest_name": "beans-v2"}, headers=ANA)
    assert applied.status_code == 200, applied.text
    done = _finished(client, f"/api/explorer/resplit/{applied.json()['job_id']}", ANA)
    assert done["state"] == "done", done
    result = done["result"]
    assert Path(result["path"]) == world["own"] / "beans-v2"
    assert (world["own"] / "beans-v2" / "train").is_dir()
    # The operator's folder is as it was, and no file of hers is a link to it.
    assert sorted(p.name for p in world["shared"].iterdir()) == before
    assert (result["files_linked"], result["files_copied"] > 0) == (0, True)
    # The same name again is refused against her folder, not the shared one.
    again = client.post(f"{job}/apply", json={"dest_name": "beans-v2"}, headers=ANA)
    assert again.status_code == 409


def test_re_splitting_an_own_dataset_writes_next_to_it(
    client: TestClient, world: dict[str, Path]
) -> None:
    sample = client.post(
        "/api/sample-dataset", json={"task": "classification"}, headers=ANA
    ).json()
    preview = client.post(
        "/api/explorer/resplit/preview", json={"base_dir": sample["path"]}, headers=ANA
    )
    job = f"/api/explorer/resplit/{preview.json()['job_id']}"
    assert _finished(client, job, ANA)["state"] == "done"
    applied = client.post(f"{job}/apply", json={"dest_name": "mine-v2"}, headers=ANA)
    done = _finished(client, f"/api/explorer/resplit/{applied.json()['job_id']}", ANA)
    assert done["state"] == "done", done
    assert Path(done["result"]["path"]) == world["own"] / "mine-v2"


def test_every_laboratory_route_answers_on_a_server(client: TestClient) -> None:
    closed = client.get("/api/session", headers=ANA).json()["closed"]
    assert [key for key in closed if "/api/explorer/" in key] == []
    assert len([key for key in READY if "/api/explorer/" in key]) == 31
    assert "POST /api/dataset/preview_augment" in closed
