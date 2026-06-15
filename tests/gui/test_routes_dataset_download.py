from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from visionforge.gui.api.dataset_download import (
    download_dataset,
    download_torchvision,
)


class _FakeCIFAR:
    """Stand-in for a torchvision built-in: 2 classes, (PIL, label) items."""

    classes = ["cat", "dog"]

    def __init__(self, root: str, train: bool, download: bool) -> None:
        self._n = 4 if train else 2

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return Image.new("RGB", (8, 8), (idx * 10, 0, 0)), idx % 2


@pytest.fixture
def _fake_cifar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torchvision.datasets.CIFAR10", _FakeCIFAR)


class TestTorchvisionDownload:
    def test_materializes_imagefolder(self, _fake_cifar: None, tmp_path: Path) -> None:
        out = tmp_path / "ds"
        result = download_torchvision("cifar10", out, splits=("train", "test"))

        assert result.total_images == 6  # 4 train + 2 test
        assert result.splits == {"train": 4, "test": 2}
        assert result.classes == ["cat", "dog"]
        # ImageFolder layout: <out>/<split>/<class>/*.png
        assert (out / "train" / "cat").is_dir()
        assert (out / "train" / "dog").is_dir()
        pngs = list((out / "train").rglob("*.png"))
        assert len(pngs) == 4

    def test_limit_caps_per_class(self, _fake_cifar: None, tmp_path: Path) -> None:
        result = download_torchvision(
            "cifar10", tmp_path / "ds", splits=("train",), limit=1
        )
        assert result.total_images == 2  # 1 per class (cat, dog)

    def test_unknown_dataset_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown torchvision dataset"):
            download_torchvision("not_a_dataset", tmp_path)


class TestDispatcher:
    def test_routes_to_torchvision(self, _fake_cifar: None, tmp_path: Path) -> None:
        result = download_dataset(
            "torchvision", dataset="cifar10", out_dir=str(tmp_path), splits=("test",)
        )
        assert result.provider == "torchvision"
        assert result.total_images == 2

    @pytest.mark.parametrize("provider", ["roboflow", "kaggle", "huggingface"])
    def test_unimplemented_providers_raise(self, provider: str, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not implemented"):
            download_dataset(provider, dataset="x", out_dir=str(tmp_path))

    def test_unknown_provider_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown dataset provider"):
            download_dataset("bogus", dataset="x", out_dir=str(tmp_path))


class TestExecuteRoute:
    def test_execute_builds_response(self, _fake_cifar: None, tmp_path: Path) -> None:
        from visionforge.gui.api.routes import _execute_dataset_download
        from visionforge.gui.api.schemas import DatasetDownloadRequest

        resp = _execute_dataset_download(
            DatasetDownloadRequest(
                provider="torchvision",
                dataset="cifar10",
                out_dir=str(tmp_path / "ds"),
                splits=["train", "test"],
            )
        )
        assert resp.provider == "torchvision"
        assert resp.dataset == "cifar10"
        assert resp.total_images == 6
        assert resp.classes == ["cat", "dog"]
