"""Tests for the YOLO-layout detection dataset stats helper + endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from visionforge.gui.api.routes import _collect_detection_dataset_stats
from visionforge.gui.api.schemas import DetectionDatasetStatsRequest


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_yolo_dataset(base: Path, *, classes_txt: bool = True) -> Path:
    """images/<split> + labels/<split> with cat(0)=4, dog(1)=1 instances."""
    # train: img1 → class0, class1 ; img2 → class0 x3
    _write(base / "images" / "train" / "img1.jpg")
    _write(base / "images" / "train" / "img2.jpg")
    _write(
        base / "labels" / "train" / "img1.txt", "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n"
    )
    _write(
        base / "labels" / "train" / "img2.txt",
        "0 0.5 0.5 0.2 0.2\n0 0.3 0.3 0.1 0.1\n0 0.6 0.6 0.1 0.1\n",
    )
    # val: img3 has no label → unlabeled
    _write(base / "images" / "val" / "img3.jpg")
    if classes_txt:
        _write(base / "classes.txt", "cat\ndog\n")
    return base


class TestCollectDetectionDatasetStats:
    def test_counts_instances_per_class(self, tmp_path: Path) -> None:
        base = _make_yolo_dataset(tmp_path / "ds")
        resp = _collect_detection_dataset_stats(
            DetectionDatasetStatsRequest(base_dir=str(base))
        )
        train = resp.splits["train"]
        assert train.total_images == 2
        assert train.total_annotations == 5
        assert train.class_counts == {"cat": 4, "dog": 1}
        assert train.unlabeled_images == 0
        assert resp.class_names == ["cat", "dog"]

    def test_unlabeled_images_counted(self, tmp_path: Path) -> None:
        base = _make_yolo_dataset(tmp_path / "ds")
        resp = _collect_detection_dataset_stats(
            DetectionDatasetStatsRequest(base_dir=str(base))
        )
        val = resp.splits["val"]
        assert val.total_images == 1
        assert val.total_annotations == 0
        assert val.unlabeled_images == 1

    def test_missing_split_flagged(self, tmp_path: Path) -> None:
        base = _make_yolo_dataset(tmp_path / "ds")
        resp = _collect_detection_dataset_stats(
            DetectionDatasetStatsRequest(base_dir=str(base))
        )
        assert resp.splits["test"].missing is True

    def test_imbalance_detected(self, tmp_path: Path) -> None:
        base = _make_yolo_dataset(tmp_path / "ds")
        resp = _collect_detection_dataset_stats(
            DetectionDatasetStatsRequest(base_dir=str(base))
        )
        # cat=4 vs dog=1 → ratio 4 > 2.0
        assert resp.imbalanced is True

    def test_generated_names_without_classes_txt(self, tmp_path: Path) -> None:
        base = _make_yolo_dataset(tmp_path / "ds", classes_txt=False)
        resp = _collect_detection_dataset_stats(
            DetectionDatasetStatsRequest(base_dir=str(base))
        )
        assert resp.class_names == ["class_0", "class_1"]
        assert resp.splits["train"].class_counts == {"class_0": 4, "class_1": 1}

    def test_missing_base_dir_returns_message(self, tmp_path: Path) -> None:
        resp = _collect_detection_dataset_stats(
            DetectionDatasetStatsRequest(base_dir=str(tmp_path / "nope"))
        )
        assert resp.splits == {}
        assert resp.message is not None

    def test_no_yolo_splits_returns_message(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        resp = _collect_detection_dataset_stats(
            DetectionDatasetStatsRequest(base_dir=str(tmp_path / "empty"))
        )
        assert all(s.missing for s in resp.splits.values())
        assert resp.message is not None


class TestDetectionDatasetStatsEndpoint:
    def test_endpoint_returns_stats(self, tmp_path: Path) -> None:
        from visionforge.gui.server import app

        base = _make_yolo_dataset(tmp_path / "ds")
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post("/api/detection/dataset/stats", json={"base_dir": str(base)})
        assert resp.status_code == 200
        body = resp.json()
        assert body["class_names"] == ["cat", "dog"]
        assert body["splits"]["train"]["class_counts"] == {"cat": 4, "dog": 1}
        assert body["imbalanced"] is True
