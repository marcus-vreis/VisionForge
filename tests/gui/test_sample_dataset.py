"""The synthetic sample datasets for the "Primeiro treino" guides (ADR-115, ADR-117).

Every test runs with the working directory moved to ``tmp_path``: the folder is
``datasets/exemplo-<tarefa>/`` relative to where the GUI was started. Each task's
layout is read back by the loader that task trains with, or by the dataset check
its panel runs, so a folder the endpoint calls finished is one that trains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from fastapi.testclient import TestClient

from visionforge.core.anomaly_data import AnomalyDataModule
from visionforge.core.data import DataModule
from visionforge.core.detection_data import DetectionDataModule, resolve_yolo_split
from visionforge.core.detection_dataset import DetectionDataset
from visionforge.core.regression_data import RegressionDataModule
from visionforge.core.segmentation_data import SegmentationDataModule
from visionforge.gui.api.sample_dataset import (
    SAMPLE_DIR_NAMES,
    SampleDatasetExistsError,
    SampleTaskError,
    create_sample_dataset,
)
from visionforge.utils.anomaly_config import AnomalyConfig
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.detection_config import DetectionConfig
from visionforge.utils.regression_config import RegressionConfig
from visionforge.utils.segmentation_config import SegmentationConfig

EN = {"X-VF-Lang": "en"}
TASKS = ["classification", "detection", "regression", "segmentation", "anomaly"]


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(routes_mod, "_SAMPLE_DATASETS_DIR", Path("datasets"))
    return TestClient(app, raise_server_exceptions=True, client=("127.0.0.1", 50000))


def _output(tmp_path: Path) -> dict[str, str]:
    return {
        "models_dir": str(tmp_path / "models"),
        "reports_dir": str(tmp_path / "reports"),
        "graphics_dir": str(tmp_path / "graphics"),
        "logs_dir": str(tmp_path / "logs"),
    }


def _build(client: TestClient, task: str) -> dict[str, Any]:
    resp = client.post("/api/sample-dataset", json={"task": task})
    assert resp.status_code == 201, resp.text
    body: dict[str, Any] = resp.json()
    return body


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
            "output": _output(tmp_path),
            "device": {"kind": "cpu"},
        }
    )
    return DataModule(config)


class TestEndpoint:
    def test_creates_a_layout_the_classification_datamodule_loads(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        body = _build(client, "classification")

        base = Path(body["path"])
        assert base == (tmp_path / "datasets" / "exemplo-classificacao").resolve()
        assert body["task"] == "classification"
        assert body["classes"] == ["class_a", "class_b"]
        assert body["counts"] == {"train": 12, "val": 12, "test": 12}
        assert body["targets"] == []
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

    def test_files_of_the_researcher_are_never_touched(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        mine = tmp_path / "datasets" / "exemplo-classificacao"
        mine.mkdir(parents=True)
        (mine / "notes.txt").write_text("notes", encoding="utf-8")

        resp = client.post("/api/sample-dataset", json={"task": "classification"})

        assert resp.status_code == 409
        assert [p.name for p in mine.iterdir()] == ["notes.txt"]

    @pytest.mark.parametrize("task", TASKS)
    def test_an_empty_folder_left_by_hand_is_taken(
        self, client: TestClient, tmp_path: Path, task: str
    ) -> None:
        (tmp_path / "datasets" / SAMPLE_DIR_NAMES[task]).mkdir(parents=True)

        resp = client.post("/api/sample-dataset", json={"task": task})

        assert resp.status_code == 201, resp.text

    @pytest.mark.parametrize("task", ["custom", "nonsense", "../etc", ""])
    def test_a_task_without_a_sample_is_400_and_creates_nothing(
        self, client: TestClient, tmp_path: Path, task: str
    ) -> None:
        resp = client.post("/api/sample-dataset", json={"task": task})

        assert resp.status_code == 400
        assert not (tmp_path / "datasets").exists()

    def test_the_dataset_thumbnails_can_be_served(self, client: TestClient) -> None:
        body = _build(client, "classification")
        image = Path(body["path"]) / "train" / "class_a" / "img0.png"

        resp = client.get("/api/dataset/file", params={"path": str(image)})

        assert resp.status_code == 200, resp.text


class TestEveryTask:
    """The folder, the response and the 409 of each of the five tasks."""

    @pytest.mark.parametrize(
        ("task", "folder"),
        [
            ("classification", "exemplo-classificacao"),
            ("detection", "exemplo-deteccao"),
            ("regression", "exemplo-regressao"),
            ("segmentation", "exemplo-segmentacao"),
            ("anomaly", "exemplo-anomalia"),
        ],
    )
    def test_each_task_has_its_own_portuguese_folder(
        self, client: TestClient, tmp_path: Path, task: str, folder: str
    ) -> None:
        body = _build(client, task)

        assert Path(body["path"]) == (tmp_path / "datasets" / folder).resolve()
        assert body["task"] == task

    @pytest.mark.parametrize("task", TASKS)
    def test_a_second_call_is_409_with_the_path_and_the_language_of_the_page(
        self, client: TestClient, task: str
    ) -> None:
        first = _build(client, task)

        en = client.post("/api/sample-dataset", json={"task": task}, headers=EN)
        pt = client.post("/api/sample-dataset", json={"task": task})

        assert en.status_code == pt.status_code == 409
        assert en.json()["path"] == pt.json()["path"] == first["path"]
        assert "already exists" in en.json()["detail"]
        assert "já existe" in pt.json()["detail"]

    @pytest.mark.parametrize("task", TASKS)
    def test_nothing_hidden_is_left_beside_the_folder(
        self, client: TestClient, tmp_path: Path, task: str
    ) -> None:
        _build(client, task)

        assert [p.name for p in (tmp_path / "datasets").iterdir()] == [
            SAMPLE_DIR_NAMES[task]
        ]

    def test_an_unknown_task_says_so_in_the_language_of_the_page(
        self, client: TestClient
    ) -> None:
        en = client.post("/api/sample-dataset", json={"task": "x"}, headers=EN)
        pt = client.post("/api/sample-dataset", json={"task": "x"})

        assert en.json()["detail"] == "Unknown task 'x'."
        assert pt.json()["detail"] == "Tarefa desconhecida 'x'."

    def test_detection_response(self, client: TestClient) -> None:
        body = _build(client, "detection")

        assert body["classes"] == ["object"]
        assert body["counts"] == {"train": 4, "val": 4}
        assert body["targets"] == []

    def test_regression_response(self, client: TestClient) -> None:
        body = _build(client, "regression")

        assert body["classes"] == []
        assert body["targets"] == ["target"]
        assert body["counts"] == {"train": 12, "val": 6, "test": 4}

    def test_segmentation_response(self, client: TestClient) -> None:
        body = _build(client, "segmentation")

        # Pixel values found in the masks: 0 is background, 1 and 2 the classes.
        assert body["classes"] == ["0", "1", "2"]
        assert body["counts"] == {"train": 6, "val": 3, "test": 2}

    def test_anomaly_response(self, client: TestClient) -> None:
        body = _build(client, "anomaly")

        # Train holds the normals only; the subfolders of test are good and defect.
        assert body["classes"] == ["defect", "good"]
        assert body["counts"] == {"train": 8, "test": 8}


class TestLayoutsLoad:
    """Each layout, read by what the task trains with or checks with."""

    def test_detection_resolves_a_data_yaml_and_a_torchvision_dataset(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        base = Path(_build(client, "detection")["path"])
        config = DetectionConfig.model_validate(
            {
                "name": "det",
                "model": {"name": "yolo11n", "num_classes": 1},
                "data": {"base_dir": str(base)},
                "training": {"epochs": 1},
                "output": _output(tmp_path),
            }
        )

        spec = yaml.safe_load(
            DetectionDataModule(config)
            .resolve_data_yaml(tmp_path / "spec")
            .read_text(encoding="utf-8")
        )

        assert spec["train"] == "images/train"
        assert spec["val"] == "images/val"
        assert spec["names"] == ["object"]
        resolved = resolve_yolo_split(base, "train")
        assert resolved is not None
        image, target = DetectionDataset(*resolved)[0]
        assert image.shape == (3, 128, 128)
        assert target["boxes"].shape == (1, 4)

    def test_detection_passes_the_panels_dataset_check(
        self, client: TestClient
    ) -> None:
        base = Path(_build(client, "detection")["path"])

        stats = client.post(
            "/api/detection/dataset/stats", json={"base_dir": str(base)}
        ).json()

        assert stats["class_names"] == ["object"]
        assert stats["splits"]["train"]["total_images"] == 4
        assert stats["splits"]["train"]["total_annotations"] == 4
        assert stats["splits"]["train"]["unlabeled_images"] == 0
        assert stats["splits"]["val"]["total_images"] == 4
        assert stats["splits"]["test"]["missing"] is True

    def test_regression_datamodule_loads_and_the_check_finds_every_image(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        base = Path(_build(client, "regression")["path"])
        config = RegressionConfig.model_validate(
            {
                "name": "reg",
                "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
                "data": {
                    "base_dir": str(base),
                    "target_columns": ["target"],
                    "transforms": {"image_size": 32},
                    "num_workers": 0,
                    "pin_memory": False,
                },
                "training": {"epochs": 1, "batch_size": 4, "seed": 0},
                "output": _output(tmp_path),
                "device": {"kind": "cpu"},
            }
        )

        data = RegressionDataModule(config)
        images, targets = next(iter(data.train_loader()))
        stats = client.post(
            "/api/regression/dataset/stats", json={"base_dir": str(base)}
        ).json()

        assert images.shape == (4, 3, 32, 32)
        assert targets.shape == (4, 1)
        assert torch.all((targets >= 0) & (targets <= 1))
        assert data.test_loader() is not None
        assert {k: v["rows"] for k, v in stats["splits"].items()} == {
            "train": 12,
            "val": 6,
            "test": 4,
        }
        assert all(v["missing_images"] == 0 for v in stats["splits"].values())
        assert all(v["missing_columns"] == [] for v in stats["splits"].values())

    def test_segmentation_datamodule_loads_and_every_image_has_its_mask(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        base = Path(_build(client, "segmentation")["path"])
        config = SegmentationConfig.model_validate(
            {
                "name": "seg",
                "model": {"name": "unet", "num_classes": 3, "pretrained": False},
                "data": {
                    "base_dir": str(base),
                    "image_size": 64,
                    "num_workers": 0,
                    "pin_memory": False,
                },
                "training": {"epochs": 1, "batch_size": 2, "seed": 0},
                "output": _output(tmp_path),
                "device": {"kind": "cpu"},
            }
        )

        data = SegmentationDataModule(config)
        images, masks = next(iter(data.train_loader()))
        stats = client.post(
            "/api/segmentation/dataset/stats", json={"base_dir": str(base)}
        ).json()

        assert images.shape == (2, 3, 64, 64)
        assert masks.shape == (2, 64, 64)
        # 255 is the ignore index the augmentation fills the corners it rotates in.
        assert set(masks.unique().tolist()) - {255} <= {0, 1, 2}
        assert data.test_loader() is not None
        assert stats["mask_class_ids"] == [0, 1, 2]
        for split, paired in {"train": 6, "val": 3, "test": 2}.items():
            assert stats["splits"][split]["paired"] == paired
            assert stats["splits"][split]["unpaired_images"] == 0
            assert stats["splits"][split]["unpaired_masks"] == 0

    def test_anomaly_trains_on_normals_only_and_tests_on_both(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        base = Path(_build(client, "anomaly")["path"])
        config = AnomalyConfig.model_validate(
            {
                "name": "anom",
                "model": {"name": "autoencoder", "latent_dim": 16},
                "data": {
                    "base_dir": str(base),
                    "image_size": 64,
                    "num_workers": 0,
                    "pin_memory": False,
                },
                "training": {"epochs": 1, "batch_size": 2, "seed": 0},
                "output": _output(tmp_path),
                "device": {"kind": "cpu"},
            }
        )

        data = AnomalyDataModule(config)
        train_labels = [int(y) for _, ys in data.train_loader() for y in ys]
        test_labels = [int(y) for _, ys in data.test_loader() for y in ys]
        stats = client.post(
            "/api/anomaly/dataset/stats", json={"base_dir": str(base)}
        ).json()

        assert train_labels == [0] * 8
        assert sorted(test_labels) == [0] * 4 + [1] * 4
        assert stats["train_normal"] == 8
        assert stats["test_normal"] == 4
        assert stats["test_anomalous"] == {"defect": 4}
        assert stats["missing_train"] is False
        assert stats["missing_test"] is False


class TestBuilder:
    @pytest.mark.parametrize("task", TASKS)
    def test_leaves_no_scratch_folder_behind(self, tmp_path: Path, task: str) -> None:
        create_sample_dataset(tmp_path / "datasets", task)

        assert [p.name for p in (tmp_path / "datasets").iterdir()] == [
            SAMPLE_DIR_NAMES[task]
        ]

    @pytest.mark.parametrize(
        ("task", "builder"),
        [
            ("classification", "build_classification_dataset"),
            ("detection", "build_detection_dataset"),
            ("regression", "build_regression_dataset"),
            ("segmentation", "build_segmentation_dataset"),
            ("anomaly", "build_anomaly_dataset"),
        ],
    )
    def test_a_failed_build_leaves_nothing_the_next_call_would_reuse(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        task: str,
        builder: str,
    ) -> None:
        import visionforge.gui.api.sample_dataset as mod

        def boom(base: Path) -> Path:
            (base / "train").mkdir(parents=True)
            raise OSError("disk full")

        monkeypatch.setattr(mod, builder, boom)
        with pytest.raises(OSError, match="disk full"):
            create_sample_dataset(tmp_path / "datasets", task)

        assert list((tmp_path / "datasets").iterdir()) == []

    def test_exception_types(self, tmp_path: Path) -> None:
        create_sample_dataset(tmp_path, "classification")

        with pytest.raises(SampleDatasetExistsError) as exists:
            create_sample_dataset(tmp_path, "classification")
        assert exists.value.path == (tmp_path / "exemplo-classificacao").resolve()
        with pytest.raises(SampleTaskError):
            create_sample_dataset(tmp_path, "nonsense")

    def test_every_built_in_task_has_a_sample(self) -> None:
        assert sorted(SAMPLE_DIR_NAMES) == sorted(TASKS)
