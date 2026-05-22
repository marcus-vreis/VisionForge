"""Coverage for /api/dataset/stats and the underlying _collect_dataset_stats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from visionforge.gui.api.routes import (
    _collect_dataset_samples,
    _collect_dataset_stats,
)
from visionforge.gui.api.schemas import DatasetSamplesRequest, DatasetStatsRequest


def _save_images(folder: Path, count: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        img.save(folder / f"img_{i}.png")


def _balanced_dataset(root: Path) -> Path:
    for split in ("train", "val", "test"):
        for cls in ("class_a", "class_b"):
            _save_images(root / split / cls, 5)
    return root


def _imbalanced_dataset(root: Path) -> Path:
    """class_a has 10x more images than class_b — clearly imbalanced."""
    for split in ("train", "val", "test"):
        _save_images(root / split / "class_a", 20)
        _save_images(root / split / "class_b", 2)
    return root


class TestCollectDatasetStats:
    def test_balanced_dataset(self, tmp_path: Path) -> None:
        root = _balanced_dataset(tmp_path)
        resp = _collect_dataset_stats(DatasetStatsRequest(base_dir=str(root)))
        assert resp.imbalanced is False
        assert set(resp.class_names) == {"class_a", "class_b"}
        assert resp.splits["train"].total_images == 10
        assert resp.splits["train"].classes == {"class_a": 5, "class_b": 5}
        assert resp.splits["val"].missing is False
        assert resp.splits["test"].missing is False

    def test_imbalanced_dataset_flagged(self, tmp_path: Path) -> None:
        root = _imbalanced_dataset(tmp_path)
        resp = _collect_dataset_stats(DatasetStatsRequest(base_dir=str(root)))
        assert resp.imbalanced is True
        assert resp.splits["train"].classes["class_a"] == 20
        assert resp.splits["train"].classes["class_b"] == 2

    def test_missing_split_reported_not_raised(self, tmp_path: Path) -> None:
        """Datasets without a test/ folder still return stats for the splits present."""
        _save_images(tmp_path / "train" / "class_a", 3)
        _save_images(tmp_path / "train" / "class_b", 3)
        resp = _collect_dataset_stats(DatasetStatsRequest(base_dir=str(tmp_path)))
        assert resp.splits["train"].missing is False
        assert resp.splits["val"].missing is True
        assert resp.splits["test"].missing is True
        assert resp.imbalanced is False

    def test_nonexistent_base_dir_returns_message(self, tmp_path: Path) -> None:
        resp = _collect_dataset_stats(
            DatasetStatsRequest(base_dir=str(tmp_path / "nope"))
        )
        assert resp.splits == {}
        assert resp.class_names == []
        assert resp.message is not None
        assert "não encontrado" in resp.message

    def test_ignores_non_image_files(self, tmp_path: Path) -> None:
        """README.md and similar artifacts must not inflate counts."""
        class_dir = tmp_path / "train" / "class_a"
        _save_images(class_dir, 4)
        (class_dir / "README.md").write_text("notes", encoding="utf-8")
        (class_dir / "manifest.json").write_text("{}", encoding="utf-8")
        resp = _collect_dataset_stats(DatasetStatsRequest(base_dir=str(tmp_path)))
        assert resp.splits["train"].classes["class_a"] == 4


class TestCollectDatasetSamples:
    def test_returns_up_to_per_class_paths(self, tmp_path: Path) -> None:
        root = _balanced_dataset(tmp_path)
        resp = _collect_dataset_samples(
            DatasetSamplesRequest(base_dir=str(root), split="train", per_class=3)
        )
        assert set(resp.samples.keys()) == {"class_a", "class_b"}
        assert all(len(paths) <= 3 for paths in resp.samples.values())
        # The returned paths must be absolute and point at real files.
        for paths in resp.samples.values():
            for p in paths:
                assert Path(p).is_file()
                assert Path(p).is_absolute()

    def test_returns_empty_when_split_missing(self, tmp_path: Path) -> None:
        """A split that does not exist on disk must produce an empty samples dict."""
        _save_images(tmp_path / "train" / "class_a", 2)
        resp = _collect_dataset_samples(
            DatasetSamplesRequest(base_dir=str(tmp_path), split="test")
        )
        assert resp.samples == {}
        assert resp.message and "não encontrado" in resp.message


class TestDatasetSamplesEndpoint:
    def test_endpoint_returns_paths(self, tmp_path: Path) -> None:
        from visionforge.gui.server import app

        root = _balanced_dataset(tmp_path)
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/dataset/samples",
            json={"base_dir": str(root), "split": "train", "per_class": 2},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert set(body["samples"].keys()) == {"class_a", "class_b"}


class TestServeDatasetFile:
    def test_serves_valid_image(self, tmp_path: Path) -> None:
        from visionforge.gui.server import app

        root = _balanced_dataset(tmp_path)
        sample = next((root / "train" / "class_a").iterdir())
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get(f"/api/dataset/file?path={sample.resolve()}")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/")

    def test_rejects_non_image_extension(self, tmp_path: Path) -> None:
        from visionforge.gui.server import app

        txt = tmp_path / "leak.txt"
        txt.write_text("secret", encoding="utf-8")
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get(f"/api/dataset/file?path={txt.resolve()}")
        assert resp.status_code == 400

    def test_404_on_missing_file(self, tmp_path: Path) -> None:
        from visionforge.gui.server import app

        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get(f"/api/dataset/file?path={tmp_path / 'nope.png'}")
        assert resp.status_code == 404


class TestDatasetStatsEndpoint:
    def test_post_returns_balanced(self, tmp_path: Path) -> None:
        from visionforge.gui.server import app

        root = _balanced_dataset(tmp_path)
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post("/api/dataset/stats", json={"base_dir": str(root)})
        assert resp.status_code == 200
        body = resp.json()
        assert body["imbalanced"] is False
        assert body["class_names"] == ["class_a", "class_b"]
        assert body["splits"]["train"]["total_images"] == 10
