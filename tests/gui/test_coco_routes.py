"""COCO conversion endpoints (ADR-169): preview, plan, apply, cancel, refusals."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.helpers import wait_until
from visionforge.gui.api import coco_routes
from visionforge.gui.server import app

EXPORT = "/api/explorer/coco/export"
IMPORT = "/api/explorer/coco/import"
DONE = "done"

Json = dict[str, Any]


def _png(path: Path, size: tuple[int, int] = (100, 50)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (1, 2, 3)).save(path)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def yolo(tmp_path: Path, client: TestClient) -> Path:
    root = tmp_path / "ds"
    _png(root / "images" / "train" / "a.png")
    _png(root / "images" / "val" / "b.png")
    for split, stem, text in (("train", "a", "0 0.5 0.5 0.2 0.4\n"), ("val", "b", "")):
        label = root / "labels" / split / f"{stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(text, encoding="utf-8")
    (root / "classes.txt").write_text("cat\n", encoding="utf-8")
    # The explorer must have opened it first.
    opened = client.post(
        "/api/explorer/summary", json={"base_dir": str(root), "task": "detection"}
    )
    assert opened.is_success
    return root


@pytest.fixture
def coco(tmp_path: Path) -> Path:
    root = tmp_path / "cocoset"
    _png(root / "val2017" / "one.png")
    (root / "annotations").mkdir()
    body: Json = {
        "images": [{"id": 1, "file_name": "one.png", "width": 100, "height": 50}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 4, "bbox": [10, 5, 20, 10]}
        ],
        "categories": [{"id": 4, "name": "car"}],
    }
    (root / "annotations" / "instances_val2017.json").write_text(
        json.dumps(body), encoding="utf-8"
    )
    return root


@pytest.fixture(autouse=True)
def _idle_jobs() -> Iterator[None]:
    yield
    for job in list(coco_routes._JOBS._jobs.values()):
        if job.state == "running":
            job.state = "cancelled"


def _wait(client: TestClient, url: str, job_id: str) -> Json:
    body: Json = {}

    def _finished() -> bool:
        nonlocal body
        body = client.get(f"{url}/{job_id}").json()
        state: str = body["state"]
        return state != "running"

    wait_until(_finished, message="the job to finish")
    return body


def _preview(client: TestClient, url: str, **body: str) -> Json:
    res = client.post(f"{url}/preview", json=body)
    assert res.status_code == 200, res.text
    return _wait(client, url, res.json()["job_id"])


# ── export ────────────────────────────────────────────────────────────────────


def test_export_preview_is_refused_when_the_explorer_never_opened_the_dataset(
    client: TestClient, tmp_path: Path
) -> None:
    root = tmp_path / "never-opened"
    root.mkdir()
    res = client.post(f"{EXPORT}/preview", json={"base_dir": str(root)})
    assert res.status_code == 403


def test_export_preview_of_a_missing_folder_is_a_bad_request(
    client: TestClient, tmp_path: Path
) -> None:
    res = client.post(f"{EXPORT}/preview", json={"base_dir": str(tmp_path / "no")})
    assert res.status_code == 400


def test_export_preview_counts_images_boxes_and_suggests_a_name(
    client: TestClient, yolo: Path
) -> None:
    job = _preview(client, EXPORT, base_dir=str(yolo))
    assert job["state"] == DONE
    result = job["result"]
    assert result["suggested_name"] == "ds-coco"
    assert result["images"] == 2
    assert result["boxes"] == 1
    assert result["categories"] == 1
    assert result["per_split"]["train"] == {"images": 1, "boxes": 1}


def test_export_plan_checks_the_destination_name(
    client: TestClient, yolo: Path
) -> None:
    job_id = _preview(client, EXPORT, base_dir=str(yolo))["job_id"]
    ok = client.post(f"{EXPORT}/{job_id}/plan", json={"dest_name": "fresh"})
    assert ok.status_code == 200
    assert ok.json()["dest_name"] == "fresh"
    bad = client.post(f"{EXPORT}/{job_id}/plan", json={"dest_name": "a/b"})
    assert bad.status_code == 400
    (yolo.parent / "taken").mkdir()
    taken = client.post(f"{EXPORT}/{job_id}/plan", json={"dest_name": "taken"})
    assert taken.status_code == 409


def test_export_apply_writes_a_new_dataset(client: TestClient, yolo: Path) -> None:
    job_id = _preview(client, EXPORT, base_dir=str(yolo))["job_id"]
    res = client.post(f"{EXPORT}/{job_id}/apply", json={"dest_name": "ds-coco"})
    assert res.status_code == 200, res.text
    done = _wait(client, EXPORT, res.json()["job_id"])
    assert done["state"] == DONE
    assert done["result"]["path"] == str(yolo.parent / "ds-coco")
    assert done["result"]["images"] == 2
    assert (yolo.parent / "ds-coco" / "annotations" / "instances_train.json").is_file()
    assert (yolo.parent / "ds-coco" / "images" / "val" / "b.png").is_file()


def test_export_apply_refuses_a_source_that_changed(
    client: TestClient, yolo: Path
) -> None:
    job_id = _preview(client, EXPORT, base_dir=str(yolo))["job_id"]
    _png(yolo / "images" / "val" / "late.png")
    res = client.post(f"{EXPORT}/{job_id}/apply", json={"dest_name": "ds-coco"})
    assert res.status_code == 409
    assert not (yolo.parent / "ds-coco").exists()


def test_export_apply_refuses_a_taken_or_bad_name(
    client: TestClient, yolo: Path
) -> None:
    job_id = _preview(client, EXPORT, base_dir=str(yolo))["job_id"]
    (yolo.parent / "taken").mkdir()
    taken = client.post(f"{EXPORT}/{job_id}/apply", json={"dest_name": "taken"})
    assert taken.status_code == 409
    bad = client.post(f"{EXPORT}/{job_id}/apply", json={"dest_name": "a/b"})
    assert bad.status_code == 400


def test_export_apply_needs_a_finished_export_preview(
    client: TestClient, yolo: Path, coco: Path
) -> None:
    import_id = _preview(client, IMPORT, source_dir=str(coco))["job_id"]
    res = client.post(f"{EXPORT}/{import_id}/apply", json={"dest_name": "x"})
    assert res.status_code == 409
    missing = client.post(f"{EXPORT}/nope/apply", json={"dest_name": "x"})
    assert missing.status_code == 404


def test_a_running_job_makes_the_next_one_wait(
    client: TestClient, yolo: Path, coco: Path
) -> None:
    running = coco_routes._JOBS.register("preview")
    try:
        assert (
            client.post(f"{EXPORT}/preview", json={"base_dir": str(yolo)}).status_code
            == 409
        )
        assert (
            client.post(f"{IMPORT}/preview", json={"source_dir": str(coco)}).status_code
            == 409
        )
    finally:
        running.state = "cancelled"


def test_unknown_jobs_are_not_found(client: TestClient) -> None:
    assert client.get(f"{EXPORT}/nope").status_code == 404
    assert client.get(f"{IMPORT}/nope").status_code == 404
    assert client.post(f"{EXPORT}/nope/cancel").status_code == 404
    assert client.post(f"{IMPORT}/nope/cancel").status_code == 404


def test_cancel_acknowledges_a_known_job(client: TestClient, yolo: Path) -> None:
    job_id = _preview(client, EXPORT, base_dir=str(yolo))["job_id"]
    res = client.post(f"{EXPORT}/{job_id}/cancel")
    assert res.status_code == 200
    assert res.json() == {"cancelling": True}


# ── import ────────────────────────────────────────────────────────────────────


def test_import_preview_of_a_missing_folder_is_a_bad_request(
    client: TestClient, tmp_path: Path
) -> None:
    res = client.post(f"{IMPORT}/preview", json={"source_dir": str(tmp_path / "no")})
    assert res.status_code == 400


def test_import_preview_without_annotation_files_fails_the_job(
    client: TestClient, tmp_path: Path
) -> None:
    (tmp_path / "empty").mkdir()
    job = _preview(client, IMPORT, source_dir=str(tmp_path / "empty"))
    assert job["state"] == "failed"
    assert job["error"]


def test_import_preview_describes_each_json(client: TestClient, coco: Path) -> None:
    job = _preview(client, IMPORT, source_dir=str(coco))
    assert job["state"] == DONE
    result = job["result"]
    assert result["suggested_name"] == "cocoset-yolo"
    source = result["sources"][0]
    assert source["name"] == "annotations/instances_val2017.json"
    assert source["detected_split"] == "val"
    assert source["images_dir"] == "val2017"
    assert (source["images_kept"], source["boxes"]) == (1, 1)


def test_import_plan_maps_the_categories_and_flags_unassigned_files(
    client: TestClient, coco: Path
) -> None:
    (coco / "annotations" / "instances_val2017.json").rename(
        coco / "annotations" / "mine.json"
    )
    job_id = _preview(client, IMPORT, source_dir=str(coco))["job_id"]
    first = client.post(f"{IMPORT}/{job_id}/plan", json={}).json()
    assert first["ready"] is False
    assert first["unassigned"] == ["annotations/mine.json"]
    second = client.post(
        f"{IMPORT}/{job_id}/plan", json={"splits": {"annotations/mine.json": "test"}}
    ).json()
    assert second["ready"] is True
    assert second["mapping"] == [
        {"coco_id": 4, "yolo_id": 0, "name": "car", "original_name": "car"}
    ]
    assert second["per_split"]["test"] == {"images": 1, "boxes": 1}


def test_import_plan_refuses_two_files_on_one_split(
    client: TestClient, coco: Path
) -> None:
    body = (coco / "annotations" / "instances_val2017.json").read_text()
    (coco / "annotations" / "val.json").write_text(body, encoding="utf-8")
    job_id = _preview(client, IMPORT, source_dir=str(coco))["job_id"]
    res = client.post(f"{IMPORT}/{job_id}/plan", json={})
    assert res.status_code == 400


def test_import_plan_checks_the_destination_name(
    client: TestClient, coco: Path
) -> None:
    job_id = _preview(client, IMPORT, source_dir=str(coco))["job_id"]
    (coco.parent / "taken").mkdir()
    res = client.post(f"{IMPORT}/{job_id}/plan", json={"dest_name": "taken"})
    assert res.status_code == 409


def test_import_apply_writes_a_dataset_the_explorer_can_open(
    client: TestClient, coco: Path
) -> None:
    job_id = _preview(client, IMPORT, source_dir=str(coco))["job_id"]
    res = client.post(f"{IMPORT}/{job_id}/apply", json={"dest_name": "cocoset-yolo"})
    assert res.status_code == 200, res.text
    done = _wait(client, IMPORT, res.json()["job_id"])
    assert done["state"] == DONE
    out = coco.parent / "cocoset-yolo"
    assert done["result"]["path"] == str(out)
    assert (
        done["result"]["images"],
        done["result"]["boxes"],
        done["result"]["classes"],
    ) == (1, 1, 1)
    summary = client.post(
        "/api/explorer/summary", json={"base_dir": str(out), "task": "detection"}
    )
    assert summary.status_code == 200
    assert summary.json()["class_names"] == ["car"]


def test_import_apply_refuses_an_unassigned_file(
    client: TestClient, coco: Path
) -> None:
    (coco / "annotations" / "instances_val2017.json").rename(
        coco / "annotations" / "mine.json"
    )
    job_id = _preview(client, IMPORT, source_dir=str(coco))["job_id"]
    res = client.post(f"{IMPORT}/{job_id}/apply", json={"dest_name": "out"})
    assert res.status_code == 400
    assert not (coco.parent / "out").exists()


def test_import_apply_refuses_a_source_that_changed(
    client: TestClient, coco: Path
) -> None:
    job_id = _preview(client, IMPORT, source_dir=str(coco))["job_id"]
    _png(coco / "val2017" / "late.png")
    res = client.post(f"{IMPORT}/{job_id}/apply", json={"dest_name": "out"})
    assert res.status_code == 409
    assert not (coco.parent / "out").exists()


def test_import_apply_refuses_a_taken_name(client: TestClient, coco: Path) -> None:
    job_id = _preview(client, IMPORT, source_dir=str(coco))["job_id"]
    (coco.parent / "taken").mkdir()
    res = client.post(f"{IMPORT}/{job_id}/apply", json={"dest_name": "taken"})
    assert res.status_code == 409
