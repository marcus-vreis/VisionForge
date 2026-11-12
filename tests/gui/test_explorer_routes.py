"""Dataset explorer endpoints (ADR-165): summary, gallery, thumbnails, scan job."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.helpers import wait_until
from visionforge.gui.server import app


def _png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    _png(tmp_path / "train/cat/a.png", (255, 0, 0))
    _png(tmp_path / "train/dog/b.png", (0, 255, 0))
    _png(tmp_path / "val/cat/a.png", (255, 0, 0))  # leaks from train
    (tmp_path / "train/cat/bad.png").write_bytes(b"junk")
    return tmp_path


def _wait(client: TestClient, job_id: str) -> dict[str, Any]:
    body: dict[str, Any] = {}

    def _finished() -> bool:
        nonlocal body
        body = client.get(f"/api/explorer/scan/{job_id}").json()
        state: str = body["state"]
        return state != "running"

    wait_until(_finished, message="the scan to finish")
    return body


def test_summary_counts(client: TestClient, dataset: Path) -> None:
    body = client.post("/api/explorer/summary", json={"base_dir": str(dataset)}).json()
    assert body["class_names"] == ["cat", "dog"]
    assert body["splits"]["train"]["per_class"] == {"cat": 2, "dog": 1}
    assert body["splits"]["val"]["images"] == 1


def test_summary_of_missing_folder_is_400(client: TestClient, tmp_path: Path) -> None:
    res = client.post("/api/explorer/summary", json={"base_dir": str(tmp_path / "no")})
    assert res.status_code == 400


def test_images_filter_and_page(client: TestClient, dataset: Path) -> None:
    body = client.post(
        "/api/explorer/images",
        json={
            "base_dir": str(dataset),
            "split": "train",
            "class_name": "cat",
            "limit": 1,
        },
    ).json()
    assert body["total"] == 2
    assert len(body["items"]) == 1
    assert body["items"][0]["path"] == "train/cat/a.png"
    assert body["items"][0]["classes"] == ["cat"]


def test_thumb_serves_a_small_jpeg(client: TestClient, dataset: Path) -> None:
    client.post("/api/explorer/summary", json={"base_dir": str(dataset)})
    res = client.get(
        "/api/explorer/thumb",
        params={"root": str(dataset), "path": "train/cat/a.png", "size": 16},
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    with Image.open(io.BytesIO(res.content)) as img:
        assert max(img.size) <= 32  # the minimum size is clamped up to 32


def test_thumb_refuses_unopened_root_and_traversal(
    client: TestClient, dataset: Path, tmp_path: Path
) -> None:
    other = tmp_path / "other"
    _png(other / "x.png", (1, 1, 1))
    res = client.get(
        "/api/explorer/thumb", params={"root": str(other), "path": "x.png"}
    )
    assert res.status_code == 403
    client.post("/api/explorer/summary", json={"base_dir": str(dataset)})
    res = client.get(
        "/api/explorer/thumb", params={"root": str(dataset), "path": "../x.png"}
    )
    assert res.status_code == 400


def test_thumb_of_corrupted_image_is_422(client: TestClient, dataset: Path) -> None:
    client.post("/api/explorer/summary", json={"base_dir": str(dataset)})
    res = client.get(
        "/api/explorer/thumb",
        params={"root": str(dataset), "path": "train/cat/bad.png"},
    )
    assert res.status_code == 422


def test_scan_job_reports_corruption_and_leakage(
    client: TestClient, dataset: Path
) -> None:
    started = client.post("/api/explorer/scan", json={"base_dir": str(dataset)}).json()
    done = _wait(client, started["job_id"])
    assert done["state"] == "done"
    result = done["result"]
    assert [c["path"] for c in result["corrupted"]] == ["train/cat/bad.png"]
    assert result["leakage_group_count"] == 1
    assert done["done"] == done["total"] == 4


def test_scan_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/api/explorer/scan/nope").status_code == 404
    assert client.post("/api/explorer/scan/nope/cancel").status_code == 404


def test_second_scan_while_running_is_409(
    client: TestClient, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from visionforge.gui.api import explorer_jobs

    gate = threading.Event()
    real = explorer_jobs.scan_integrity

    def slow(*args, **kwargs):  # type: ignore[no-untyped-def]
        gate.wait(5)
        return real(*args, **kwargs)

    monkeypatch.setattr(explorer_jobs, "scan_integrity", slow)
    first = client.post("/api/explorer/scan", json={"base_dir": str(dataset)})
    assert first.status_code == 200
    second = client.post("/api/explorer/scan", json={"base_dir": str(dataset)})
    assert second.status_code == 409
    gate.set()
    _wait(client, first.json()["job_id"])


def test_cancel_marks_job_cancelled(
    client: TestClient, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from visionforge.gui.api import explorer_jobs

    gate = threading.Event()
    real = explorer_jobs.scan_integrity

    def gated(index, **kwargs):  # type: ignore[no-untyped-def]
        gate.wait(5)
        return real(index, **kwargs)

    monkeypatch.setattr(explorer_jobs, "scan_integrity", gated)
    job = client.post("/api/explorer/scan", json={"base_dir": str(dataset)}).json()
    assert client.post(f"/api/explorer/scan/{job['job_id']}/cancel").status_code == 200
    gate.set()
    done = _wait(client, job["job_id"])
    assert done["state"] == "cancelled"
    assert done["result"]["cancelled"] is True
    assert done["result"]["scanned"] == 0
