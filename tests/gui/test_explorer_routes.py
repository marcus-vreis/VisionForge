"""Dataset explorer endpoints (ADR-165): summary, gallery, thumbnails, scan job."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.helpers import wait_until
from visionforge.gui.api import explorer_routes
from visionforge.gui.server import app


def _png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # A finished scan is kept on disk (ADR-231): never in the repository's outputs.
    monkeypatch.setattr(explorer_routes, "_SCAN_STORE", tmp_path / "scans")
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


def _saved(client: TestClient, dataset: Path) -> dict[str, Any]:
    body: dict[str, Any] = client.post(
        "/api/explorer/scan/saved", json={"base_dir": str(dataset)}
    ).json()
    return body


def test_a_finished_scan_is_kept_and_marked_stale_when_the_dataset_changes(
    client: TestClient, dataset: Path
) -> None:
    assert _saved(client, dataset) == {"report": None, "saved_at": None, "stale": False}

    started = client.post("/api/explorer/scan", json={"base_dir": str(dataset)}).json()
    report = _wait(client, started["job_id"])["result"]
    kept = _saved(client, dataset)
    _png(dataset / "val/dog/new.png", (0, 0, 255))
    changed = _saved(client, dataset)

    assert kept["report"] == report
    assert kept["stale"] is False
    assert kept["saved_at"]
    assert changed["stale"] is True
    assert changed["report"] == report


def test_another_split_layout_is_another_dataset(
    client: TestClient, dataset: Path
) -> None:
    started = client.post("/api/explorer/scan", json={"base_dir": str(dataset)}).json()
    _wait(client, started["job_id"])
    other = {"base_dir": str(dataset), "val_dir": "train"}

    assert client.post("/api/explorer/scan/saved", json=other).json()["report"] is None


def test_the_scan_finds_near_duplicates_across_splits_on_request(
    client: TestClient, tmp_path: Path
) -> None:
    field = Image.radial_gradient("L").resize((32, 32)).convert("RGB")
    root = tmp_path / "near"
    for split, image in (
        ("train", field),
        ("val", field.point(lambda v: min(255, v + 8))),
    ):
        (root / split / "cat").mkdir(parents=True)
        image.save(root / split / "cat" / "a.png")
    body = {"base_dir": str(root)}

    plain = _wait(client, client.post("/api/explorer/scan", json=body).json()["job_id"])
    near = _wait(
        client,
        client.post(
            "/api/explorer/scan", json={**body, "near_duplicates": True}
        ).json()["job_id"],
    )

    assert plain["result"]["near_checked"] is False
    assert near["result"]["near_checked"] is True
    assert near["result"]["near_groups"] == [
        {"splits": ["train", "val"], "paths": ["train/cat/a.png", "val/cat/a.png"]}
    ]
    # Two passes over two images.
    assert near["done"] == near["total"] == 4


def test_the_summary_says_whether_the_splits_share_the_class_mix(
    client: TestClient, dataset: Path
) -> None:
    # train: cat 2 (a, bad) and dog 1; val: cat 1. dog is missing from val.
    body = client.post("/api/explorer/summary", json={"base_dir": str(dataset)}).json()

    balance = body["balance"]
    assert balance["splits"] == ["train", "val"]
    assert balance["verdict"] == "missing_class"
    dog = next(r for r in balance["classes"] if r["name"] == "dog")
    assert dog["missing_in"] == ["val"]
    assert dog["gap_pp"] == pytest.approx(100 / 3)


def _seg_dataset(root: Path) -> Path:
    """train: one 4x4 mask, half class 1. val: one mask with an id past 2 classes."""
    for split, value in (("train", 1), ("val", 9)):
        (root / split / "images").mkdir(parents=True)
        (root / split / "masks").mkdir(parents=True)
        Image.new("RGB", (4, 4)).save(root / split / "images" / "a.jpg")
        mask = Image.new("L", (4, 4), 0)
        for x in range(4):
            for y in range(2):
                mask.putpixel((x, y), value)
        mask.save(root / split / "masks" / "a.png")
    return root


def test_mask_analysis_runs_as_a_job_and_is_kept(
    client: TestClient, tmp_path: Path
) -> None:
    root = _seg_dataset(tmp_path / "seg")
    body = {"base_dir": str(root), "num_classes": 2}
    assert client.post("/api/explorer/masks/saved", json=body).json()["report"] is None

    started = client.post("/api/explorer/masks", json=body).json()
    report = _wait(client, started["job_id"])["result"]
    assert report["task"] == "segmentation"
    assert report["scanned"] == 2
    assert report["out_of_range_ids"] == [9]
    by_id = {c["id"]: c["shares"] for c in report["classes"]}
    assert by_id[1] == {"train": 50.0, "val": 0}

    kept = client.post("/api/explorer/masks/saved", json=body).json()
    assert kept["report"] == report
    assert kept["stale"] is False
    # Another class count is another analysis: what is out of range changes.
    other = {**body, "num_classes": 10}
    assert client.post("/api/explorer/masks/saved", json=other).json()["report"] is None


def test_mask_analysis_of_a_missing_folder_is_400(
    client: TestClient, tmp_path: Path
) -> None:
    res = client.post("/api/explorer/masks", json={"base_dir": str(tmp_path / "no")})
    assert res.status_code == 400


def test_target_analysis_runs_as_a_job_and_is_kept(
    client: TestClient, tmp_path: Path
) -> None:
    root = tmp_path / "reg"
    (root / "images").mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(root / "images" / "a.png")
    (root / "train.csv").write_text(
        "image,y\na.png,1\na.png,2\na.png,3\n", encoding="utf-8"
    )
    body = {"base_dir": str(root), "target_columns": ["y"]}
    assert (
        client.post("/api/explorer/targets/saved", json=body).json()["report"] is None
    )

    started = client.post("/api/explorer/targets", json=body).json()
    report = _wait(client, started["job_id"])["result"]
    assert report["task"] == "regression"
    assert report["targets"][0]["splits"]["train"]["mean"] == 2.0
    assert report["splits"]["val"]["present"] is False

    kept = client.post("/api/explorer/targets/saved", json=body).json()
    assert (kept["report"], kept["stale"]) == (report, False)
    (root / "train.csv").write_text("image,y\na.png,9\n", encoding="utf-8")
    assert client.post("/api/explorer/targets/saved", json=body).json()["stale"] is True


def test_target_analysis_refuses_a_csv_outside_the_dataset(
    client: TestClient, tmp_path: Path
) -> None:
    (tmp_path / "reg").mkdir()
    res = client.post(
        "/api/explorer/targets",
        json={"base_dir": str(tmp_path / "reg"), "train_csv": "../x.csv"},
    )
    assert res.status_code == 400
    assert "../x.csv" in res.json()["detail"]


def test_anomaly_analysis_runs_as_a_job_and_is_kept(
    client: TestClient, tmp_path: Path
) -> None:
    root = tmp_path / "mvtec"
    _png(root / "train/good/a.png", (10, 20, 30))
    _png(root / "test/good/b.png", (40, 50, 60))
    _png(root / "test/crack/c.png", (70, 80, 90))
    body = {"base_dir": str(root)}
    assert (
        client.post("/api/explorer/anomaly/saved", json=body).json()["report"] is None
    )

    started = client.post("/api/explorer/anomaly", json=body).json()
    report = _wait(client, started["job_id"])["result"]
    assert report["counts"] == {"defect:crack": 1, "test_normal": 1, "train": 1}
    assert report["defect_kinds"] == ["crack"]

    kept = client.post("/api/explorer/anomaly/saved", json=body).json()
    assert (kept["report"], kept["stale"]) == (report, False)
    _png(root / "test/crack/d.png", (1, 2, 3))
    assert client.post("/api/explorer/anomaly/saved", json=body).json()["stale"] is True


def _mean_colour_embedder() -> Any:
    """Mean RGB as the features, in place of the ImageNet network (ADR-239)."""
    import numpy as np

    def embed(images: list[Image.Image]) -> Any:
        return np.array(
            [np.asarray(im, dtype=np.float32).mean(axis=(0, 1)) for im in images]
        )

    return embed


def test_class_comparison_runs_as_a_job_and_is_kept(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from visionforge.core import class_similarity

    monkeypatch.setattr(class_similarity, "pretrained_embedder", _mean_colour_embedder)
    root = tmp_path / "cls"
    # cat and dog are both red, sky is blue: cat and dog look alike.
    for i in range(3):
        _png(root / f"train/cat/c{i}.png", (200, i, 0))
        _png(root / f"train/dog/d{i}.png", (200, i + 3, 0))
        _png(root / f"train/sky/s{i}.png", (0, i, 200))
    body = {"base_dir": str(root)}
    assert (
        client.post("/api/explorer/classes/saved", json=body).json()["report"] is None
    )

    started = client.post("/api/explorer/classes", json=body).json()
    report = _wait(client, started["job_id"])["result"]
    assert report["described"] == 9
    assert [(p["a"], p["b"]) for p in report["pairs"] if p["flagged"]] == [
        ("cat", "dog")
    ]

    kept = client.post("/api/explorer/classes/saved", json=body).json()
    assert kept["report"] == report
    # Another sample size is another comparison.
    other = {**body, "per_class": 2}
    assert (
        client.post("/api/explorer/classes/saved", json=other).json()["report"] is None
    )


def test_class_comparison_is_for_classification_only(
    client: TestClient, dataset: Path
) -> None:
    body = {"base_dir": str(dataset), "task": "detection"}
    res = client.post("/api/explorer/classes", json=body)
    assert res.status_code == 400
    assert "classifica" in res.json()["detail"].lower()
