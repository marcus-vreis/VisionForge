"""Re-split endpoints (ADR-168): preview job, plan, apply job, cancel, refusals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.helpers import wait_until
from visionforge.gui.server import app

API = "/api/explorer/resplit"


def _png(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (seed % 256, seed // 256, 3)).save(path)


def _build(root: Path, per_split: dict[str, dict[str, int]]) -> Path:
    n = 0
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
        {
            "train": {"cat": 6, "dog": 6},
            "val": {"cat": 2, "dog": 2},
            "test": {"cat": 1, "dog": 1},
        },
    )
    # The explorer must have opened it first.
    assert client.post("/api/explorer/summary", json={"base_dir": str(base)}).is_success
    return base


def _wait(client: TestClient, job_id: str) -> dict[str, Any]:
    body: dict[str, Any] = {}

    def _finished() -> bool:
        nonlocal body
        body = client.get(f"{API}/{job_id}").json()
        state: str = body["state"]
        return state != "running"

    wait_until(_finished, message="the job to finish")
    return body


def _preview(client: TestClient, base: Path, **extra: Any) -> dict[str, Any]:
    res = client.post(f"{API}/preview", json={"base_dir": str(base), **extra})
    assert res.status_code == 200, res.text
    return _wait(client, res.json()["job_id"])


def test_preview_describes_the_dataset_and_suggests_a_name(
    client: TestClient, dataset: Path
) -> None:
    job = _preview(client, dataset)
    assert job["state"] == "done"
    result = job["result"]
    assert result["suggested_name"] == "ds-resplit"
    assert result["images"] == 18
    assert result["per_split"] == {"train": 12, "val": 4, "test": 2}
    assert result["classes"] == ["cat", "dog"]
    assert result["group_csv"] is None


def test_plan_replans_without_hashing_again(client: TestClient, dataset: Path) -> None:
    job_id = _preview(client, dataset)["job_id"]
    default = client.post(f"{API}/{job_id}/plan", json={}).json()
    assert default["locked_splits"] == ["test"]
    assert default["after_totals"] == {"train": 12, "val": 4, "test": 2}
    changed = client.post(
        f"{API}/{job_id}/plan",
        json={"ratios": {"train": 1, "val": 1}, "seed": 3},
    ).json()
    assert changed["after_totals"] == {"train": 8, "val": 8, "test": 2}
    assert changed["seed"] == 3
    unlocked = client.post(
        f"{API}/{job_id}/plan",
        json={"ratios": {"train": 1, "val": 1, "test": 2}, "lock_test": False},
    ).json()
    assert unlocked["locked_splits"] == []
    assert unlocked["after_totals"]["test"] == 10


def test_preview_plan_apply_happy_path(
    client: TestClient, dataset: Path, tmp_path: Path
) -> None:
    before = sorted(p.name for p in dataset.rglob("*"))
    job_id = _preview(client, dataset)["job_id"]
    res = client.post(
        f"{API}/{job_id}/apply",
        json={"dest_name": "again", "ratios": {"train": 1, "val": 1}},
    )
    assert res.status_code == 200, res.text
    done = _wait(client, res.json()["job_id"])
    assert done["state"] == "done", done
    assert done["result"]["images"] == 18
    assert Path(done["result"]["path"]) == (tmp_path / "again").resolve()
    assert (tmp_path / "again" / "visionforge_dataset.json").is_file()
    assert sorted(p.name for p in dataset.rglob("*")) == before
    summary = client.post(
        "/api/explorer/summary", json={"base_dir": str(tmp_path / "again")}
    ).json()
    assert summary["splits"]["val"]["images"] == 8
    assert summary["splits"]["test"]["images"] == 2


def test_a_group_file_is_read_by_the_preview(
    client: TestClient, dataset: Path, tmp_path: Path
) -> None:
    groups = tmp_path / "groups.csv"
    groups.write_text(
        "path,group\ntrain/cat/cat0.png,a\ntrain/dog/dog0.png,a\n", encoding="utf-8"
    )
    job = _preview(client, dataset, group_csv=str(groups))
    assert job["result"]["group_count"] == 1
    assert job["result"]["ungrouped"] == 16
    plan = client.post(f"{API}/{job['job_id']}/plan", json={}).json()
    assert plan["by_group"] is True


def test_preview_refuses_a_dataset_the_explorer_did_not_open(
    client: TestClient, tmp_path: Path
) -> None:
    other = _build(tmp_path / "unopened", {"train": {"cat": 2}})
    res = client.post(f"{API}/preview", json={"base_dir": str(other)})
    assert res.status_code == 403


def test_preview_refuses_a_missing_folder_or_group_file(
    client: TestClient, dataset: Path, tmp_path: Path
) -> None:
    missing = client.post(f"{API}/preview", json={"base_dir": str(tmp_path / "nope")})
    assert missing.status_code == 400
    no_csv = client.post(
        f"{API}/preview",
        json={"base_dir": str(dataset), "group_csv": str(tmp_path / "none.csv")},
    )
    assert no_csv.status_code == 400


def test_a_bad_group_file_fails_the_job_with_a_message(
    client: TestClient, dataset: Path, tmp_path: Path
) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("file,team\nx,y\n", encoding="utf-8")
    job = _preview(client, dataset, group_csv=str(bad))
    assert job["state"] == "failed"
    assert "path,group" in job["error"]


def test_plan_refusals(client: TestClient, dataset: Path) -> None:
    job_id = _preview(client, dataset)["job_id"]
    bad_ratio = client.post(f"{API}/{job_id}/plan", json={"ratios": {"train": -1}})
    assert bad_ratio.status_code == 400
    test_locked = client.post(f"{API}/{job_id}/plan", json={"ratios": {"test": 1}})
    assert test_locked.status_code == 400
    no_groups = client.post(f"{API}/{job_id}/plan", json={"by_group": True})
    assert no_groups.status_code == 400
    bad_seed = client.post(f"{API}/{job_id}/plan", json={"seed": -1})
    assert bad_seed.status_code == 422
    unknown = client.post(f"{API}/{'0' * 32}/plan", json={})
    assert unknown.status_code == 404


def test_apply_refuses_bad_or_existing_names(client: TestClient, dataset: Path) -> None:
    job_id = _preview(client, dataset)["job_id"]
    exists = client.post(f"{API}/{job_id}/apply", json={"dest_name": "ds"})
    assert exists.status_code == 409
    unsafe = client.post(f"{API}/{job_id}/apply", json={"dest_name": "../x"})
    assert unsafe.status_code == 400


def test_apply_refuses_when_the_dataset_changed_since_the_preview(
    client: TestClient, dataset: Path, tmp_path: Path
) -> None:
    job_id = _preview(client, dataset)["job_id"]
    _png(dataset / "train" / "cat" / "late.png", 9000)
    res = client.post(f"{API}/{job_id}/apply", json={"dest_name": "late"})
    assert res.status_code == 409
    assert not (tmp_path / "late").exists()


def test_apply_refuses_when_the_group_file_changed(
    client: TestClient, dataset: Path, tmp_path: Path
) -> None:
    groups = tmp_path / "groups.csv"
    groups.write_text("path,group\ntrain/cat/cat0.png,a\n", encoding="utf-8")
    job_id = _preview(client, dataset, group_csv=str(groups))["job_id"]
    groups.write_text("path,group\ntrain/cat/cat0.png,b\n", encoding="utf-8")
    res = client.post(f"{API}/{job_id}/apply", json={"dest_name": "grouped"})
    assert res.status_code == 409


def test_status_and_cancel_of_an_unknown_job_are_404(client: TestClient) -> None:
    assert client.get(f"{API}/{'f' * 32}").status_code == 404
    assert client.post(f"{API}/{'f' * 32}/cancel").status_code == 404


def test_cancel_of_a_known_job_is_accepted(client: TestClient, dataset: Path) -> None:
    job_id = _preview(client, dataset)["job_id"]
    res = client.post(f"{API}/{job_id}/cancel")
    assert res.status_code == 200
    assert res.json() == {"cancelling": True}
