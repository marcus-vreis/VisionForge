"""Incorporate endpoints (ADR-166): preview job, plan, apply job, cancel, refusals."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from visionforge.gui.server import app

API = "/api/explorer/incorporate"


def _png(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (seed % 256, seed // 256, 3)).save(path)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def dataset(tmp_path: Path, client: TestClient) -> Path:
    base = tmp_path / "ds"
    n = 0
    for split, count in (("train", 7), ("val", 2), ("test", 1)):
        for cls in ("cat", "dog"):
            for i in range(count):
                n += 1
                _png(base / split / cls / f"{cls}{i}.png", n)
    # The explorer must have opened it first.
    assert client.post("/api/explorer/summary", json={"base_dir": str(base)}).is_success
    return base


@pytest.fixture()
def new_images(tmp_path: Path) -> Path:
    source = tmp_path / "new"
    n = 5000
    for cls, count in (("cat", 10), ("bird", 4)):
        for i in range(count):
            n += 1
            _png(source / cls / f"n{i}.png", n)
    return source


def _wait(client: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(200):
        body: dict[str, Any] = client.get(f"{API}/{job_id}").json()
        if body["state"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def _preview(client: TestClient, base: Path, source: Path) -> dict[str, Any]:
    res = client.post(
        f"{API}/preview", json={"base_dir": str(base), "source_dir": str(source)}
    )
    assert res.status_code == 200, res.text
    return _wait(client, res.json()["job_id"])


def test_preview_lists_new_classes_and_a_suggested_name(
    client: TestClient, dataset: Path, new_images: Path
) -> None:
    job = _preview(client, dataset, new_images)
    assert job["state"] == "done"
    assert job["result"]["suggested_name"] == "ds-v2"
    assert job["result"]["new_classes"] == [
        {"name": "bird", "images": 4, "source_id": None}
    ]
    assert job["result"]["source_images"] == 14


def test_plan_follows_the_accepted_classes_and_the_seed(
    client: TestClient, dataset: Path, new_images: Path
) -> None:
    job_id = _preview(client, dataset, new_images)["job_id"]
    refused = client.post(f"{API}/{job_id}/plan", json={}).json()
    assert refused["added"] == 10
    assert refused["excluded_by_reason"] == {"unknown_class": 4}
    accepted = client.post(
        f"{API}/{job_id}/plan", json={"accept_classes": ["bird"], "seed": 5}
    ).json()
    assert accepted["added"] == 14
    assert accepted["seed"] == 5
    assert accepted["classes"] == ["cat", "dog", "bird"]


def test_apply_writes_the_new_dataset_and_leaves_the_old_one(
    client: TestClient, dataset: Path, new_images: Path, tmp_path: Path
) -> None:
    before = sorted(p.name for p in dataset.rglob("*"))
    job_id = _preview(client, dataset, new_images)["job_id"]
    res = client.post(
        f"{API}/{job_id}/apply", json={"dest_name": "ds-v2", "accept_classes": ["bird"]}
    )
    assert res.status_code == 200, res.text
    done = _wait(client, res.json()["job_id"])
    assert done["state"] == "done", done
    assert done["result"]["added"] == 14
    assert Path(done["result"]["path"]) == (tmp_path / "ds-v2").resolve()
    assert (tmp_path / "ds-v2" / "visionforge_dataset.json").is_file()
    assert sorted(p.name for p in dataset.rglob("*")) == before
    # The explorer opens the result as it is.
    summary = client.post(
        "/api/explorer/summary", json={"base_dir": str(tmp_path / "ds-v2")}
    ).json()
    assert summary["class_names"] == ["bird", "cat", "dog"]
    assert summary["total_images"] == 20 + 14


def test_apply_refuses_an_existing_or_unsafe_name(
    client: TestClient, dataset: Path, new_images: Path
) -> None:
    job_id = _preview(client, dataset, new_images)["job_id"]
    assert (
        client.post(f"{API}/{job_id}/apply", json={"dest_name": "ds"}).status_code
        == 409
    )
    assert (
        client.post(f"{API}/{job_id}/apply", json={"dest_name": "a/b"}).status_code
        == 400
    )


def test_plan_with_a_class_that_is_not_new_is_400(
    client: TestClient, dataset: Path, new_images: Path
) -> None:
    job_id = _preview(client, dataset, new_images)["job_id"]
    res = client.post(f"{API}/{job_id}/plan", json={"accept_classes": ["cat"]})
    assert res.status_code == 400


def test_preview_of_a_dataset_the_explorer_never_opened_is_403(
    client: TestClient, tmp_path: Path, new_images: Path
) -> None:
    other = tmp_path / "unopened"
    _png(other / "train/cat/a.png", 1)
    res = client.post(
        f"{API}/preview", json={"base_dir": str(other), "source_dir": str(new_images)}
    )
    assert res.status_code == 403


def test_preview_with_missing_folders_is_400(
    client: TestClient, dataset: Path, tmp_path: Path
) -> None:
    res = client.post(
        f"{API}/preview",
        json={"base_dir": str(dataset), "source_dir": str(tmp_path / "nope")},
    )
    assert res.status_code == 400


def test_source_inside_the_dataset_fails_the_job(
    client: TestClient, dataset: Path
) -> None:
    job = _preview(client, dataset, dataset / "train")
    assert job["state"] == "failed"
    assert job["error"]


def test_unknown_job_is_404(client: TestClient) -> None:
    assert client.get(f"{API}/nope").status_code == 404
    assert client.post(f"{API}/nope/plan", json={}).status_code == 404
    assert client.post(f"{API}/nope/cancel").status_code == 404


def test_cancelling_an_apply_leaves_no_folder(
    client: TestClient, dataset: Path, new_images: Path, tmp_path: Path
) -> None:
    job_id = _preview(client, dataset, new_images)["job_id"]
    started = client.post(f"{API}/{job_id}/apply", json={"dest_name": "ds-v2"}).json()
    client.post(f"{API}/{started['job_id']}/cancel")
    done = _wait(client, started["job_id"])
    # Either the cancel won the race (nothing written) or the job had finished.
    if done["state"] == "cancelled":
        assert not (tmp_path / "ds-v2").exists()
    else:
        assert done["state"] == "done"
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []
