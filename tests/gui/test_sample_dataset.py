"""The synthetic sample dataset for the "Primeiro treino" guide (ADR-115).

Every test runs with the working directory moved to ``tmp_path``: the folder is
``datasets/exemplo-classificacao/`` relative to where the GUI was started.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visionforge.core.data import DataModule
from visionforge.gui.api.sample_dataset import (
    SampleDatasetExistsError,
    SampleTaskError,
    create_sample_dataset,
)
from visionforge.utils.config import ExperimentConfig


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(routes_mod, "_SAMPLE_DATASETS_DIR", Path("datasets"))
    yield TestClient(app, raise_server_exceptions=True, client=("127.0.0.1", 50000))


def _datamodule(base: Path, tmp_path: Path) -> DataModule:
    config = ExperimentConfig.model_validate(
        {
            "name": "sample",
            "block": "classification",
            "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
            "training": {"epochs": 1, "batch_size": 4, "seed": 0},
            "data": {
                "base_dir": str(base),
                "num_workers": 0,
                "pin_memory": False,
                "transforms": {"image_size": 32},
            },
            "output": {
                "models_dir": str(tmp_path / "models"),
                "reports_dir": str(tmp_path / "reports"),
                "graphics_dir": str(tmp_path / "graphics"),
                "logs_dir": str(tmp_path / "logs"),
            },
            "device": {"kind": "cpu"},
        }
    )
    return DataModule(config)


class TestEndpoint:
    def test_creates_a_layout_the_classification_datamodule_loads(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        resp = client.post("/api/sample-dataset", json={"task": "classification"})

        assert resp.status_code == 201, resp.text
        body = resp.json()
        base = Path(body["path"])
        assert base == (tmp_path / "datasets" / "exemplo-classificacao").resolve()
        assert body["classes"] == ["class_a", "class_b"]
        assert body["counts"] == {"train": 12, "val": 12, "test": 12}
        data = _datamodule(base, tmp_path)
        images, labels = next(iter(data.train_loader()))
        assert images.shape[1:] == (3, 32, 32)
        assert set(labels.tolist()) <= {0, 1}

    def test_task_defaults_to_classification(self, client: TestClient) -> None:
        resp = client.post("/api/sample-dataset", json={})

        assert resp.status_code == 201, resp.text

    def test_creates_datasets_when_the_folder_is_missing(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        assert not (tmp_path / "datasets").exists()

        client.post("/api/sample-dataset", json={"task": "classification"})

        assert (tmp_path / "datasets" / "exemplo-classificacao" / "train").is_dir()

    def test_second_call_is_409_and_returns_the_path(self, client: TestClient) -> None:
        first = client.post("/api/sample-dataset", json={"task": "classification"})
        second = client.post("/api/sample-dataset", json={"task": "classification"})

        assert second.status_code == 409
        body = second.json()
        assert body["path"] == first.json()["path"]
        assert "already exists" in body["detail"]

    def test_files_of_the_researcher_are_never_touched(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        mine = tmp_path / "datasets" / "exemplo-classificacao"
        mine.mkdir(parents=True)
        (mine / "notes.txt").write_text("notes", encoding="utf-8")

        resp = client.post("/api/sample-dataset", json={"task": "classification"})

        assert resp.status_code == 409
        assert [p.name for p in mine.iterdir()] == ["notes.txt"]

    def test_an_empty_folder_left_by_hand_is_taken(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        (tmp_path / "datasets" / "exemplo-classificacao").mkdir(parents=True)

        resp = client.post("/api/sample-dataset", json={"task": "classification"})

        assert resp.status_code == 201, resp.text

    @pytest.mark.parametrize(
        "task", ["detection", "segmentation", "regression", "anomaly"]
    )
    def test_other_known_tasks_are_400_not_yet(
        self, client: TestClient, task: str
    ) -> None:
        resp = client.post("/api/sample-dataset", json={"task": task})

        assert resp.status_code == 400
        assert "classification" in resp.json()["detail"]

    def test_unknown_task_is_400(self, client: TestClient, tmp_path: Path) -> None:
        resp = client.post("/api/sample-dataset", json={"task": "../etc"})

        assert resp.status_code == 400
        assert not (tmp_path / "datasets").exists()

    def test_the_dataset_thumbnails_can_be_served(self, client: TestClient) -> None:
        body = client.post(
            "/api/sample-dataset", json={"task": "classification"}
        ).json()
        image = Path(body["path"]) / "train" / "class_a" / "img0.png"

        resp = client.get("/api/dataset/file", params={"path": str(image)})

        assert resp.status_code == 200, resp.text


class TestBuilder:
    def test_leaves_no_scratch_folder_behind(self, tmp_path: Path) -> None:
        create_sample_dataset(tmp_path / "datasets", "classification")

        assert [p.name for p in (tmp_path / "datasets").iterdir()] == [
            "exemplo-classificacao"
        ]

    def test_a_failed_build_leaves_nothing_the_next_call_would_reuse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import visionforge.gui.api.sample_dataset as mod

        def boom(base: Path) -> Path:
            (base / "train").mkdir(parents=True)
            raise OSError("disk full")

        monkeypatch.setattr(mod, "build_classification_dataset", boom)
        with pytest.raises(OSError, match="disk full"):
            create_sample_dataset(tmp_path / "datasets", "classification")

        assert list((tmp_path / "datasets").iterdir()) == []

    def test_exception_types(self, tmp_path: Path) -> None:
        create_sample_dataset(tmp_path, "classification")

        with pytest.raises(SampleDatasetExistsError) as exists:
            create_sample_dataset(tmp_path, "classification")
        assert exists.value.path == (tmp_path / "exemplo-classificacao").resolve()
        with pytest.raises(SampleTaskError):
            create_sample_dataset(tmp_path, "detection")
        with pytest.raises(SampleTaskError):
            create_sample_dataset(tmp_path, "nonsense")
