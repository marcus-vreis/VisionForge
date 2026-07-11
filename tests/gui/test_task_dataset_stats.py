from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from visionforge.gui.api.routes import (
    _collect_anomaly_dataset_stats,
    _collect_regression_dataset_stats,
    _collect_segmentation_dataset_stats,
)
from visionforge.gui.api.schemas import (
    AnomalyDatasetStatsRequest,
    RegressionDatasetStatsRequest,
    SegmentationDatasetStatsRequest,
)


def _write_image(path: Path, values: int | list[int] = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(values, int):
        arr = np.full((8, 8), values, dtype=np.uint8)
    else:
        arr = np.array(values, dtype=np.uint8).reshape(1, -1)
        arr = np.tile(arr, (8, max(1, 8 // arr.shape[1])))
    Image.fromarray(arr, "L").save(path)


class TestSegmentationDatasetStats:
    def test_pairing_and_class_ids(self, tmp_path: Path) -> None:
        for stem in ("a", "b", "c"):
            _write_image(tmp_path / "train" / "images" / f"{stem}.png")
        _write_image(tmp_path / "train" / "masks" / "a.png", [0, 1])
        _write_image(tmp_path / "train" / "masks" / "b.png", [0, 255])
        _write_image(tmp_path / "train" / "masks" / "zz.png", [1])

        resp = _collect_segmentation_dataset_stats(
            SegmentationDatasetStatsRequest(base_dir=str(tmp_path))
        )
        train = resp.splits["train"]
        assert train.images == 3
        assert train.masks == 3
        assert train.paired == 2  # a + b; c has no mask, zz has no image
        assert train.unpaired_images == 1
        assert train.unpaired_masks == 1
        assert resp.splits["val"].missing is True
        # sampled class ids include the ignore_index-style 255
        assert resp.mask_class_ids == [0, 1, 255]

    def test_missing_base_dir_reports_message(self, tmp_path: Path) -> None:
        resp = _collect_segmentation_dataset_stats(
            SegmentationDatasetStatsRequest(base_dir=str(tmp_path / "nope"))
        )
        assert resp.splits == {}
        assert resp.message is not None


class TestAnomalyDatasetStats:
    def test_counts_normal_and_defect_subdirs(self, tmp_path: Path) -> None:
        for i in range(3):
            _write_image(tmp_path / "train" / "good" / f"n{i}.png")
        for i in range(2):
            _write_image(tmp_path / "test" / "good" / f"n{i}.png")
        for i in range(4):
            _write_image(tmp_path / "test" / "crack" / f"d{i}.png")
        _write_image(tmp_path / "test" / "scratch" / "d0.png")

        resp = _collect_anomaly_dataset_stats(
            AnomalyDatasetStatsRequest(base_dir=str(tmp_path))
        )
        assert resp.train_normal == 3
        assert resp.test_normal == 2
        assert resp.test_anomalous == {"crack": 4, "scratch": 1}
        assert resp.message is None

    def test_empty_normal_train_flags_a_message(self, tmp_path: Path) -> None:
        (tmp_path / "train" / "good").mkdir(parents=True)
        (tmp_path / "test" / "good").mkdir(parents=True)
        resp = _collect_anomaly_dataset_stats(
            AnomalyDatasetStatsRequest(base_dir=str(tmp_path))
        )
        assert resp.train_normal == 0
        assert resp.message is not None


class TestRegressionDatasetStats:
    def _write_manifest(
        self, path: Path, rows: list[dict[str, str]], header: list[str]
    ) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)

    def test_rows_targets_and_missing_images(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "images" / "x0.png")
        self._write_manifest(
            tmp_path / "train.csv",
            [
                {"image": "x0.png", "target": "1.0"},
                {"image": "x1.png", "target": "3.0"},
                {"image": "x2.png", "target": "notanumber"},
            ],
            ["image", "target"],
        )
        resp = _collect_regression_dataset_stats(
            RegressionDatasetStatsRequest(base_dir=str(tmp_path))
        )
        train = resp.splits["train"]
        assert train.rows == 3
        assert train.missing_columns == []
        assert train.checked_images == 3
        assert train.missing_images == 2  # x1 + x2 don't exist on disk
        stats = train.targets["target"]
        assert stats.count == 2  # the non-numeric row is skipped
        assert stats.mean == 2.0
        assert stats.min == 1.0
        assert stats.max == 3.0
        assert resp.splits["val"].missing is True

    def test_missing_target_column_is_reported(self, tmp_path: Path) -> None:
        self._write_manifest(
            tmp_path / "train.csv",
            [{"image": "a.png", "value": "1"}],
            ["image", "value"],
        )
        resp = _collect_regression_dataset_stats(
            RegressionDatasetStatsRequest(base_dir=str(tmp_path))
        )
        assert resp.splits["train"].missing_columns == ["target"]

    def test_missing_base_dir_reports_message(self, tmp_path: Path) -> None:
        resp = _collect_regression_dataset_stats(
            RegressionDatasetStatsRequest(base_dir=str(tmp_path / "nope"))
        )
        assert resp.message is not None


class TestEndpointsExposed:
    def test_all_three_stats_routes_registered(self) -> None:
        from visionforge.gui.server import app

        # getattr: newer Starlette exposes included routers without a .path
        paths = {getattr(route, "path", None) for route in app.routes}
        for task in ("segmentation", "anomaly", "regression"):
            assert f"/api/{task}/dataset/stats" in paths
