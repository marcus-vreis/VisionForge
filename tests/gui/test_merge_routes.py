"""Merge endpoints (ADR-167): preview job, plan, apply job, cancel, refusals."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from visionforge.gui.api import merge_routes
from visionforge.gui.api.merge_jobs import MergeJobs
from visionforge.gui.server import app

API = "/api/explorer/merge"


def _png(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (seed % 256, seed // 256, 3)).save(path)


def _build(root: Path, per_split: dict[str, dict[str, int]], start: int) -> Path:
    n = start
    for split, classes in per_split.items():
        for cls, count in classes.items():
            for i in range(count):
                n += 1
                _png(root / split / cls / f"{cls}{i}.png", n)
    return root


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def dataset(tmp_path: Path, client: TestClient) -> Path:
    base = _build(
        tmp_path / "ds",
        {"train": {"cat": 3, "dog": 3}, "val": {"cat": 1, "dog": 1}},
        0,
    )
    # The explorer must have opened it first.
    assert client.post("/api/explorer/summary", json={"base_dir": str(base)}).is_success
    return base


@pytest.fixture
def other(tmp_path: Path) -> Path:
    return _build(
        tmp_path / "other",
        {"train": {"cat": 2, "bird": 2}, "test": {"cat": 1, "bird": 1}},
        1000,
    )


def _wait(client: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(200):
        body: dict[str, Any] = client.get(f"{API}/{job_id}").json()
        if body["state"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def _preview(client: TestClient, base: Path, other_dir: Path) -> dict[str, Any]:
    res = client.post(
        f"{API}/preview", json={"base_dir": str(base), "other_dir": str(other_dir)}
    )
    assert res.status_code == 200, res.text
    return _wait(client, res.json()["job_id"])


def test_preview_lists_the_classes_of_both_and_a_suggested_name(
    client: TestClient, dataset: Path, other: Path
) -> None:
    job = _preview(client, dataset, other)
    assert job["state"] == "done"
    result = job["result"]
    assert result["suggested_name"] == "ds-other"
    assert result["classes"]["a"] == [
        {"name": "cat", "images": 4},
        {"name": "dog", "images": 4},
    ]
    assert result["classes"]["b"] == [
        {"name": "bird", "images": 3},
        {"name": "cat", "images": 3},
    ]
    assert result["found"] == {"a": 8, "b": 6}
    assert result["similar_names"] == []


def test_plan_follows_the_mapping_without_hashing_again(
    client: TestClient, dataset: Path, other: Path
) -> None:
    job_id = _preview(client, dataset, other)["job_id"]
    default = client.post(f"{API}/{job_id}/plan", json={}).json()
    assert [c["name"] for c in default["classes"]] == ["cat", "dog", "bird"]
    assert default["images_total"] == 14
    mapped = client.post(
        f"{API}/{job_id}/plan",
        json={"mapping": {"a": {"dog": "pet"}, "b": {"bird": None}}},
    ).json()
    assert [c["name"] for c in mapped["classes"]] == ["cat", "pet"]
    assert mapped["dropped_by_reason"] == {"discarded_class": 3}
    assert mapped["only_in"] == {"a": ["val"], "b": ["test"]}


def test_preview_plan_apply_happy_path(
    client: TestClient, dataset: Path, other: Path, tmp_path: Path
) -> None:
    before = sorted(p.name for p in dataset.rglob("*"))
    job_id = _preview(client, dataset, other)["job_id"]
    res = client.post(
        f"{API}/{job_id}/apply",
        json={"dest_name": "joined", "mapping": {"b": {"bird": None}}},
    )
    assert res.status_code == 200, res.text
    done = _wait(client, res.json()["job_id"])
    assert done["state"] == "done", done
    assert done["result"]["images"] == 11
    assert Path(done["result"]["path"]) == (tmp_path / "joined").resolve()
    assert (tmp_path / "joined" / "visionforge_dataset.json").is_file()
    assert sorted(p.name for p in dataset.rglob("*")) == before
    summary = client.post(
        "/api/explorer/summary", json={"base_dir": str(tmp_path / "joined")}
    ).json()
    assert summary["class_names"] == ["cat", "dog"]
    assert summary["total_images"] == 11


def test_apply_refuses_an_existing_or_unsafe_name(
    client: TestClient, dataset: Path, other: Path
) -> None:
    job_id = _preview(client, dataset, other)["job_id"]
    exists = client.post(f"{API}/{job_id}/apply", json={"dest_name": "other"})
    unsafe = client.post(f"{API}/{job_id}/apply", json={"dest_name": "a/b"})
    assert exists.status_code == 409
    assert unsafe.status_code == 400


def test_plan_and_apply_with_a_bad_mapping_are_400(
    client: TestClient, dataset: Path, other: Path
) -> None:
    job_id = _preview(client, dataset, other)["job_id"]
    unknown = client.post(f"{API}/{job_id}/plan", json={"mapping": {"a": {"x": "y"}}})
    empty = client.post(f"{API}/{job_id}/plan", json={"mapping": {"a": {"cat": " "}}})
    apply = client.post(
        f"{API}/{job_id}/apply",
        json={"dest_name": "joined", "mapping": {"a": {"cat": "a/b"}}},
    )
    assert (unknown.status_code, empty.status_code, apply.status_code) == (
        400,
        400,
        400,
    )


def test_apply_after_a_source_changed_is_409_and_writes_nothing(
    client: TestClient, dataset: Path, other: Path, tmp_path: Path
) -> None:
    job_id = _preview(client, dataset, other)["job_id"]
    _png(other / "train" / "cat" / "late.png", 7777)
    res = client.post(f"{API}/{job_id}/apply", json={"dest_name": "joined"})
    assert res.status_code == 409
    assert not (tmp_path / "joined").exists()
    _png(dataset / "train" / "cat" / "late.png", 8888)
    res = client.post(f"{API}/{job_id}/apply", json={"dest_name": "joined"})
    assert res.status_code == 409


def test_preview_of_a_dataset_the_explorer_never_opened_is_403(
    client: TestClient, tmp_path: Path, other: Path
) -> None:
    unopened = tmp_path / "unopened"
    _png(unopened / "train/cat/a.png", 1)
    res = client.post(
        f"{API}/preview", json={"base_dir": str(unopened), "other_dir": str(other)}
    )
    assert res.status_code == 403


def test_preview_with_missing_or_overlapping_folders_is_400(
    client: TestClient, dataset: Path, tmp_path: Path
) -> None:
    missing = client.post(
        f"{API}/preview",
        json={"base_dir": str(dataset), "other_dir": str(tmp_path / "nope")},
    )
    absent = client.post(
        f"{API}/preview",
        json={"base_dir": str(tmp_path / "nope"), "other_dir": str(dataset)},
    )
    same = client.post(
        f"{API}/preview", json={"base_dir": str(dataset), "other_dir": str(dataset)}
    )
    inside = client.post(
        f"{API}/preview",
        json={"base_dir": str(dataset), "other_dir": str(dataset / "train")},
    )
    assert [r.status_code for r in (missing, absent, same, inside)] == [400] * 4


def test_a_second_job_while_one_runs_is_409(
    client: TestClient,
    dataset: Path,
    other: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = MergeJobs()
    jobs.register("preview")  # never finishes
    monkeypatch.setattr(merge_routes, "_JOBS", jobs)
    res = client.post(
        f"{API}/preview", json={"base_dir": str(dataset), "other_dir": str(other)}
    )
    assert res.status_code == 409


def test_unknown_job_is_404_and_an_unfinished_preview_is_409(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert client.get(f"{API}/nope").status_code == 404
    assert client.post(f"{API}/nope/plan", json={}).status_code == 404
    assert client.post(f"{API}/nope/cancel").status_code == 404
    jobs = MergeJobs()
    running = jobs.register("preview")
    monkeypatch.setattr(merge_routes, "_JOBS", jobs)
    assert client.post(f"{API}/{running.job_id}/plan", json={}).status_code == 409


def test_cancelling_an_apply_leaves_no_folder(
    client: TestClient, dataset: Path, other: Path, tmp_path: Path
) -> None:
    job_id = _preview(client, dataset, other)["job_id"]
    started = client.post(f"{API}/{job_id}/apply", json={"dest_name": "joined"}).json()
    client.post(f"{API}/{started['job_id']}/cancel")
    done = _wait(client, started["job_id"])
    # Either the cancel won the race (nothing written) or the job had finished.
    if done["state"] == "cancelled":
        assert not (tmp_path / "joined").exists()
    else:
        assert done["state"] == "done"
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []
