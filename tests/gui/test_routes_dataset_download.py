from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from visionforge.gui.api.dataset_download import (
    download_dataset,
    download_huggingface,
    download_kaggle,
    download_roboflow,
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

    def test_unknown_provider_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown dataset provider"):
            download_dataset("bogus", dataset="x", out_dir=str(tmp_path))


def _install_fake_roboflow(
    monkeypatch: pytest.MonkeyPatch, rec: dict[str, Any]
) -> None:
    """Inject a fake roboflow SDK whose download() writes a couple of images."""

    class FakeVersion:
        def __init__(self, num: int) -> None:
            self.num = num

        def download(self, fmt: str, location: str) -> None:
            rec["format"] = fmt
            rec["location"] = location
            d = Path(location) / "train" / "cat"
            d.mkdir(parents=True, exist_ok=True)
            (d / "a.jpg").write_bytes(b"x")
            (d / "b.jpg").write_bytes(b"x")

    class FakeProject:
        def version(self, num: int) -> FakeVersion:
            rec["version"] = num
            return FakeVersion(num)

    class FakeWorkspace:
        def project(self, name: str) -> FakeProject:
            rec["project"] = name
            return FakeProject()

    class FakeRoboflow:
        def __init__(self, api_key: str) -> None:
            rec["api_key"] = api_key

        def workspace(self, ws: str) -> FakeWorkspace:
            rec["workspace"] = ws
            return FakeWorkspace()

    fake_mod = types.ModuleType("roboflow")
    fake_mod.Roboflow = FakeRoboflow  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "roboflow", fake_mod)


class TestRoboflowDownload:
    def test_downloads_and_counts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rec: dict[str, Any] = {}
        _install_fake_roboflow(monkeypatch, rec)
        result = download_roboflow(
            "ws/proj",
            tmp_path / "ds",
            api_key="KEY",
            version=3,
            dataset_format="folder",
        )
        assert result.provider == "roboflow"
        assert result.dataset == "ws/proj:v3"
        assert result.total_images == 2
        assert result.splits == {"train": 2}
        assert rec == {
            "api_key": "KEY",
            "workspace": "ws",
            "project": "proj",
            "version": 3,
            "format": "folder",
            "location": str(tmp_path / "ds"),
        }

    def test_dispatcher_routes_to_roboflow(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fake_roboflow(monkeypatch, {})
        result = download_dataset(
            "roboflow",
            dataset="ws/proj",
            out_dir=str(tmp_path),
            api_key="KEY",
            version=1,
        )
        assert result.provider == "roboflow"
        assert result.total_images == 2

    def test_missing_api_key_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="api_key"):
            download_roboflow("ws/proj", tmp_path, api_key=None, version=1)

    def test_missing_version_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="version"):
            download_roboflow("ws/proj", tmp_path, api_key="KEY", version=None)

    def test_malformed_dataset_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="workspace/project"):
            download_roboflow("justproject", tmp_path, api_key="KEY", version=1)


def _install_fake_kaggle(monkeypatch: pytest.MonkeyPatch, rec: dict[str, Any]) -> None:
    """Inject a fake kaggle SDK whose dataset_download_files() writes images."""

    class FakeKaggleApi:
        def authenticate(self) -> None:
            rec["authenticated"] = True

        def dataset_download_files(self, dataset: str, path: str, unzip: bool) -> None:
            rec["dataset"] = dataset
            rec["unzip"] = unzip
            d = Path(path) / "images"
            d.mkdir(parents=True, exist_ok=True)
            for n in ("a.png", "b.png", "c.png"):
                (d / n).write_bytes(b"x")

    pkg = types.ModuleType("kaggle")
    api_mod = types.ModuleType("kaggle.api")
    ext = types.ModuleType("kaggle.api.kaggle_api_extended")
    ext.KaggleApi = FakeKaggleApi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kaggle", pkg)
    monkeypatch.setitem(sys.modules, "kaggle.api", api_mod)
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", ext)


class TestKaggleDownload:
    def test_downloads_and_counts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rec: dict[str, Any] = {}
        _install_fake_kaggle(monkeypatch, rec)
        result = download_kaggle("owner/slug", tmp_path / "ds")
        assert result.provider == "kaggle"
        assert result.dataset == "owner/slug"
        assert result.total_images == 3
        assert result.splits == {"images": 3}
        assert rec["authenticated"] is True
        assert rec["dataset"] == "owner/slug"
        assert rec["unzip"] is True

    def test_dispatcher_routes_to_kaggle(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fake_kaggle(monkeypatch, {})
        result = download_dataset("kaggle", dataset="owner/slug", out_dir=str(tmp_path))
        assert result.provider == "kaggle"
        assert result.total_images == 3

    def test_malformed_dataset_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="owner/dataset-slug"):
            download_kaggle("justslug", tmp_path)


def _install_fake_datasets(
    monkeypatch: pytest.MonkeyPatch, with_label: bool = True
) -> None:
    """Inject a fake `datasets` module: load_dataset → DatasetDict of image+label."""
    from PIL import Image as PilImage

    class Image:  # class name is what the feature introspection matches on
        pass

    class ClassLabel:
        def __init__(self, names: list[str]) -> None:
            self.names = names

    class FakeSplit:
        def __init__(self, n: int) -> None:
            self._n = n
            if with_label:
                self.features = {"image": Image(), "label": ClassLabel(["cat", "dog"])}
            else:
                self.features = {"text": object()}

        def __iter__(self):
            for i in range(self._n):
                yield {
                    "image": PilImage.new("RGB", (8, 8), (i, 0, 0)),
                    "label": i % 2,
                }

    def load_dataset(name: str, token: str | None = None) -> dict[str, Any]:
        return {"train": FakeSplit(4), "test": FakeSplit(2)}

    fake_mod = types.ModuleType("datasets")
    fake_mod.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)


class TestHuggingFaceDownload:
    def test_materializes_imagefolder(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fake_datasets(monkeypatch)
        result = download_huggingface("owner/ds", tmp_path / "ds", token="tok")
        assert result.provider == "huggingface"
        assert result.total_images == 6  # 4 train + 2 test
        assert result.splits == {"train": 4, "test": 2}
        assert result.classes == ["cat", "dog"]
        assert (tmp_path / "ds" / "train" / "cat").is_dir()

    def test_dispatcher_routes_to_huggingface(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fake_datasets(monkeypatch)
        result = download_dataset(
            "huggingface", dataset="owner/ds", out_dir=str(tmp_path), token="t"
        )
        assert result.provider == "huggingface"
        assert result.total_images == 6

    def test_no_image_label_features_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fake_datasets(monkeypatch, with_label=False)
        with pytest.raises(ValueError, match="no image\\+label features"):
            download_huggingface("owner/text-ds", tmp_path)


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
