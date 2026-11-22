"""Suspicious labels of a run (ADR-237): the routes, with the inference faked.

The batch prediction block is replaced by one that writes the probabilities a
test chose, so the route's reading, ranking and keeping are what is tested.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import visionforge.gui.api.routes as routes_mod
from visionforge.blocks.batch_prediction import BatchPredictionBlock
from visionforge.gui.server import app

RUN_ID = "20261122_100000_000000"
# p(cat), p(dog) per file name; val/cat/c3.png is labeled cat but looks like a dog.
PROBS = {
    "c1.png": (0.9, 0.1),
    "c2.png": (0.8, 0.2),
    "c3.png": (0.02, 0.98),
    "d1.png": (0.1, 0.9),
    "d2.png": (0.2, 0.8),
}


def _dataset(root: Path) -> Path:
    for split in ("train", "val"):
        for name in PROBS:
            folder = root / split / ("cat" if name.startswith("c") else "dog")
            folder.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(b"x")
    return root


def _run(tmp: Path, dataset: Path, task: str = "multiclass") -> Path:
    run_dir = tmp / "models" / "exp" / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "best_model.pth").write_bytes(b"fake")
    data = {
        "id": f"exp_{RUN_ID}",
        "experiment": "exp",
        "timestamp": "2026-11-22T10:00:00",
        "status": "completed",
        "class_names": ["cat", "dog"],
        "config": {
            "name": "exp",
            "task": task,
            "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
            "training": {
                "learning_rate": 0.001,
                "epochs": 1,
                "batch_size": 2,
                "seed": 0,
            },
            "data": {"base_dir": str(dataset)},
        },
        "metrics": {},
        "history": [],
        "artifacts": {"model": str(run_dir / "best_model.pth")},
        "tests": [],
    }
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
    return run_dir


@pytest.fixture
def fake_block() -> Any:
    """Writes the CSV the real block would, from ``PROBS``."""
    state: dict[str, Any] = {}

    def setup(self: BatchPredictionBlock, config: Any) -> None:
        state["bp"] = config.batch_prediction

    def run(self: BatchPredictionBlock) -> None:
        bp = state["bp"]
        with Path(bp.output_csv).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["filename", "predicted_class", "prob_class_0", "prob_class_1"]
            )
            for path in sorted(Path(bp.input_dir).rglob("*.png")):
                p0, p1 = PROBS[path.name]
                writer.writerow([str(path), "x", p0, p1])

    def report(self: BatchPredictionBlock) -> dict[str, Any]:
        return {"failed_files": []}

    with (
        patch.object(BatchPredictionBlock, "setup", setup),
        patch.object(BatchPredictionBlock, "run", run),
        patch.object(BatchPredictionBlock, "report", report),
    ):
        yield


def _client(tmp: Path) -> Any:
    return patch.object(routes_mod, "_MODELS_DIR", tmp / "models")


@pytest.mark.usefixtures("fake_block")
def test_the_disputed_image_is_found_and_the_report_kept(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, _dataset(tmp_path / "ds"))
    with _client(tmp_path):
        client = TestClient(app)
        assert client.get(f"/api/runs/{RUN_ID}/label_suspects").json() == {
            "report": None
        }
        res = client.post(
            f"/api/runs/{RUN_ID}/label_suspects", json={"splits": ["val"]}
        )
        assert res.status_code == 200, res.text
        report = res.json()
        kept = client.get(f"/api/runs/{RUN_ID}/label_suspects").json()["report"]
    val = report["splits"]["val"]
    # Thresholds: cat (0.9 + 0.8 + 0.02) / 3, dog (0.9 + 0.8) / 2 = 0.85.
    assert val["images"] == 5
    assert val["suspect_count"] == 1
    assert val["suspects"][0] == {
        "path": "cat/c3.png",
        "given": "cat",
        "suspected": "dog",
        "p_given": 0.02,
        "p_suspected": 0.98,
    }
    assert val["in_sample"] is False
    assert report["missing_splits"] == []
    assert kept == report
    assert (run_dir / "label_suspects" / "val.csv").is_file()


@pytest.mark.usefixtures("fake_block")
def test_a_split_without_a_folder_is_named(tmp_path: Path) -> None:
    _run(tmp_path, _dataset(tmp_path / "ds"))
    with _client(tmp_path):
        res = TestClient(app).post(
            f"/api/runs/{RUN_ID}/label_suspects", json={"splits": ["test"]}
        )
    assert res.json()["missing_splits"] == ["test"]


def test_refused_for_another_task_or_split(tmp_path: Path) -> None:
    _run(tmp_path, _dataset(tmp_path / "ds"), task="regression")
    with _client(tmp_path):
        client = TestClient(app)
        assert (
            client.post(f"/api/runs/{RUN_ID}/label_suspects", json={}).status_code
            == 400
        )
        assert client.post("/api/runs/nope/label_suspects", json={}).status_code == 404
        assert client.get("/api/runs/nope/label_suspects").status_code == 404


def test_an_unknown_split_is_a_bad_request(tmp_path: Path) -> None:
    _run(tmp_path, _dataset(tmp_path / "ds"))
    with _client(tmp_path):
        res = TestClient(app).post(
            f"/api/runs/{RUN_ID}/label_suspects", json={"splits": ["holdout"]}
        )
    assert res.status_code == 400
    assert "holdout" in res.json()["detail"]
