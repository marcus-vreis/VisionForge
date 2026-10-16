"""O submit recusa dataset vazio ou incompatível com o config, com 422 traduzido.

Antes, a recusa vinha da fila: o job entrava, falhava na primeira linha do
treino e a pessoa lia uma exceção crua em inglês no log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from visionforge.utils.selftest_data import (
    build_classification_dataset,
    build_detection_dataset,
    build_regression_dataset,
    build_segmentation_dataset,
)

from .conftest import occupy_queue, release_queue


@pytest.fixture
def client_and_routes():  # type: ignore[no-untyped-def]
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    routes_mod._current_run = None
    return TestClient(app, raise_server_exceptions=True), routes_mod


def _output(tmp_path: Path) -> dict[str, str]:
    return {
        "models_dir": str(tmp_path / "models"),
        "reports_dir": str(tmp_path / "reports"),
    }


def _detection(tmp_path: Path, base: Path) -> dict[str, Any]:
    return {
        "name": "det_val",
        "model": {"backend": "ultralytics", "name": "yolo11n", "num_classes": 1},
        "data": {"base_dir": str(base), "image_size": 640},
        "training": {"epochs": 1, "batch_size": 8, "learning_rate": 0.01},
        "output": _output(tmp_path),
    }


def _segmentation(tmp_path: Path, base: Path) -> dict[str, Any]:
    return {
        "name": "seg_val",
        "model": {"name": "unet", "num_classes": 3, "pretrained": False},
        "data": {"base_dir": str(base)},
        "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.001},
        "output": _output(tmp_path),
    }


def _regression(
    tmp_path: Path, base: Path, columns: list[str] | None = None
) -> dict[str, Any]:
    cols = columns or ["target"]
    return {
        "name": "reg_val",
        "model": {"name": "resnet18", "num_targets": len(cols), "pretrained": False},
        "data": {"base_dir": str(base), "target_columns": cols},
        "training": {"epochs": 1, "batch_size": 8, "learning_rate": 0.001},
        "output": _output(tmp_path),
    }


def _classification(
    tmp_path: Path, base: Path, num_classes: int = 2, task: str = "multiclass"
) -> dict[str, Any]:
    return {
        "name": "cls_val",
        "task": task,
        "model": {"name": "resnet18", "num_classes": num_classes, "pretrained": False},
        "data": {"base_dir": str(base), "num_workers": 0},
        "output": _output(tmp_path),
    }


@pytest.fixture
def busy_queue(client_and_routes: tuple):  # type: ignore[no-untyped-def]
    """Accepted jobs stay queued, so a 200 here never trains anything."""
    _, routes_mod = client_and_routes
    occupy_queue(routes_mod)
    yield
    release_queue(routes_mod)


@pytest.mark.usefixtures("busy_queue")
class TestDetectionSubmit:
    def test_valid_dataset_is_accepted(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        client, _ = client_and_routes
        base = build_detection_dataset(tmp_path / "ds")
        resp = client.post("/api/detection/run", json=_detection(tmp_path, base))
        assert resp.status_code == 200, resp.text

    def test_empty_folder_is_refused_in_portuguese(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        client, _ = client_and_routes
        base = tmp_path / "ds"
        base.mkdir()
        resp = client.post("/api/detection/run", json=_detection(tmp_path, base))
        assert resp.status_code == 422
        assert "Nenhum split YOLO" in resp.json()["detail"]

    def test_train_split_without_images_is_refused(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        client, _ = client_and_routes
        base = tmp_path / "ds"
        (base / "images" / "train").mkdir(parents=True)
        (base / "images" / "val").mkdir(parents=True)
        resp = client.post("/api/detection/run", json=_detection(tmp_path, base))
        assert resp.status_code == 422
        assert "treino" in resp.json()["detail"]

    def test_language_header_selects_english(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        from visionforge.utils.messages import LANG_HEADER

        client, _ = client_and_routes
        base = tmp_path / "ds"
        base.mkdir()
        resp = client.post(
            "/api/detection/run",
            json=_detection(tmp_path, base),
            headers={LANG_HEADER: "en"},
        )
        assert resp.status_code == 422
        assert "No YOLO split" in resp.json()["detail"]


@pytest.mark.usefixtures("busy_queue")
class TestSegmentationSubmit:
    def test_valid_dataset_is_accepted(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        client, _ = client_and_routes
        base = build_segmentation_dataset(tmp_path / "ds")
        resp = client.post("/api/segmentation/run", json=_segmentation(tmp_path, base))
        assert resp.status_code == 200, resp.text

    def test_empty_folder_is_refused_in_portuguese(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        client, _ = client_and_routes
        base = tmp_path / "ds"
        base.mkdir()
        resp = client.post("/api/segmentation/run", json=_segmentation(tmp_path, base))
        assert resp.status_code == 422
        assert "Nenhum split encontrado" in resp.json()["detail"]

    def test_train_without_any_pair_is_refused(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        client, _ = client_and_routes
        base = tmp_path / "ds"
        (base / "train" / "images").mkdir(parents=True)
        (base / "train" / "masks").mkdir(parents=True)
        resp = client.post("/api/segmentation/run", json=_segmentation(tmp_path, base))
        assert resp.status_code == 422
        assert "par imagem" in resp.json()["detail"]


@pytest.mark.usefixtures("busy_queue")
class TestRegressionSubmit:
    def test_valid_dataset_is_accepted(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        client, _ = client_and_routes
        base = build_regression_dataset(tmp_path / "ds")
        resp = client.post("/api/regression/run", json=_regression(tmp_path, base))
        assert resp.status_code == 200, resp.text

    def test_missing_target_column_names_the_column(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        client, _ = client_and_routes
        base = build_regression_dataset(tmp_path / "ds")
        resp = client.post(
            "/api/regression/run", json=_regression(tmp_path, base, ["altura"])
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "altura" in detail
        assert "Colunas" in detail

    def test_missing_train_csv_is_refused(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        client, _ = client_and_routes
        base = tmp_path / "ds"
        base.mkdir()
        resp = client.post("/api/regression/run", json=_regression(tmp_path, base))
        assert resp.status_code == 422
        assert "train.csv" in resp.json()["detail"]

    def test_header_only_manifest_is_refused(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        client, _ = client_and_routes
        base = tmp_path / "ds"
        base.mkdir()
        (base / "train.csv").write_text("image,target\n", encoding="utf-8")
        resp = client.post("/api/regression/run", json=_regression(tmp_path, base))
        assert resp.status_code == 422
        assert "sem linhas" in resp.json()["detail"]


@pytest.mark.usefixtures("busy_queue")
class TestClassificationSubmit:
    def test_matching_num_classes_is_accepted(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        client, _ = client_and_routes
        base = build_classification_dataset(tmp_path / "ds")
        resp = client.post(
            "/api/experiment/run", json=_classification(tmp_path, base, 2)
        )
        assert resp.status_code == 200, resp.text

    def test_binary_takes_one_output_for_two_folders(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        client, _ = client_and_routes
        base = build_classification_dataset(tmp_path / "ds")
        resp = client.post(
            "/api/experiment/run",
            json=_classification(tmp_path, base, 1, task="binary"),
        )
        assert resp.status_code == 200, resp.text

    def test_more_outputs_than_folders_is_refused(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        client, _ = client_and_routes
        base = build_classification_dataset(tmp_path / "ds")
        resp = client.post(
            "/api/experiment/run", json=_classification(tmp_path, base, 5)
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "num_classes" in detail
        assert "class_a" in detail

    def test_binary_with_three_folders_is_refused(
        self, client_and_routes: tuple, tmp_path: Path
    ) -> None:
        client, _ = client_and_routes
        base = build_classification_dataset(tmp_path / "ds")
        (base / "train" / "class_c").mkdir()
        resp = client.post(
            "/api/experiment/run",
            json=_classification(tmp_path, base, 1, task="binary"),
        )
        assert resp.status_code == 422
