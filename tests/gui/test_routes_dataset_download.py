from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from loguru import logger
from PIL import Image

from visionforge.gui.api.dataset_download import (
    _parse_roboflow_dataset,
    credentials_in_play,
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


class _BigFakeCIFAR(_FakeCIFAR):
    """Enough images per class that a 20% slice is a whole number."""

    def __init__(self, root: str, train: bool, download: bool) -> None:
        self._n = 20 if train else 6


@pytest.fixture
def _big_fake_cifar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torchvision.datasets.CIFAR10", _BigFakeCIFAR)


class TestValidationSplit:
    """torchvision ships train/test; every VisionForge task wants train/val/test.

    Without this the smoothest possible first run — download a built-in dataset,
    point the picker at it — reported "Faltando: validação" and made the user
    resolve it by hand.
    """

    def test_val_is_carved_out_of_train(
        self, _big_fake_cifar: None, tmp_path: Path
    ) -> None:
        out = tmp_path / "ds"
        result = download_torchvision("cifar10", out, splits=("train", "test"))

        assert set(result.splits) == {"train", "val", "test"}
        assert result.splits["val"] == 4  # 20 train = 10/class, 20% = 2/class
        assert result.splits["train"] == 16
        # Nothing is lost or duplicated in the move.
        assert result.splits["train"] + result.splits["val"] == 20
        assert (out / "val" / "cat").is_dir()
        assert (out / "val" / "dog").is_dir()

    def test_every_class_is_represented_in_val(
        self, _big_fake_cifar: None, tmp_path: Path
    ) -> None:
        """Stratified, so a rare class does not vanish from validation."""
        out = tmp_path / "ds"
        download_torchvision("cifar10", out, splits=("train",))
        per_class = {
            d.name: len(list(d.iterdir())) for d in sorted((out / "val").iterdir())
        }
        assert per_class == {"cat": 2, "dog": 2}

    def test_split_is_reproducible(self, _big_fake_cifar: None, tmp_path: Path) -> None:
        """Sorted, not random: downloading twice gives the same val set."""
        first = tmp_path / "a"
        second = tmp_path / "b"
        download_torchvision("cifar10", first, splits=("train",))
        download_torchvision("cifar10", second, splits=("train",))
        names = lambda root: sorted(p.name for p in (root / "val").rglob("*.png"))  # noqa: E731
        assert names(first) == names(second)

    def test_zero_fraction_keeps_the_original_two_splits(
        self, _big_fake_cifar: None, tmp_path: Path
    ) -> None:
        out = tmp_path / "ds"
        result = download_torchvision(
            "cifar10", out, splits=("train", "test"), val_fraction=0.0
        )
        assert set(result.splits) == {"train", "test"}
        assert not (out / "val").exists()

    def test_rejects_a_fraction_that_would_empty_train(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="val_fraction"):
            download_torchvision("cifar10", tmp_path, val_fraction=1.0)


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

        def download(self, fmt: str, location: str, overwrite: bool = False) -> None:
            rec["format"] = fmt
            rec["location"] = location
            rec["overwrite"] = overwrite
            # Faithful to roboflow/core/version.py: an existing location with
            # `overwrite=False` returns immediately, writing nothing. We create
            # the output folder ourselves before calling, so without the flag
            # this branch is always the one taken.
            if Path(location).exists() and not overwrite:
                return
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
        # `overwrite` is the whole ballgame: without it the client returns
        # early on the folder we just created and writes nothing.
        assert rec["overwrite"] is True
        assert rec == {
            "api_key": "KEY",
            "overwrite": True,
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


class TestRoboflowDatasetString:
    """What a person actually pastes, versus what the field asks for.

    The obvious move is to copy the project URL from the browser; the next most
    obvious is the path with its leading slash. Both used to pass the "has a
    slash" check and then split into nonsense — an empty workspace, or one
    called `https:` — so Roboflow answered about a workspace nobody asked for
    and the error read as ours.
    """

    @pytest.mark.parametrize(
        "typed",
        [
            "ws/proj",
            "/ws/proj",
            "ws/proj/",
            "  ws/proj  ",
            "https://app.roboflow.com/ws/proj",
            "http://app.roboflow.com/ws/proj",
        ],
    )
    def test_every_shape_finds_the_same_pair(self, typed: str) -> None:
        workspace, project, _version = _parse_roboflow_dataset(typed)

        assert (workspace, project) == ("ws", "proj")

    def test_a_pasted_url_carries_its_version(self) -> None:
        """So pasting the URL fills the version field's job too."""
        assert _parse_roboflow_dataset("https://app.roboflow.com/ws/proj/7") == (
            "ws",
            "proj",
            7,
        )

    def test_an_explicit_version_wins_over_the_url(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rec: dict[str, Any] = {}
        _install_fake_roboflow(monkeypatch, rec)

        download_roboflow(
            "https://app.roboflow.com/ws/proj/7",
            tmp_path / "ds",
            api_key="KEY",
            version=2,
        )

        assert rec["version"] == 2

    @pytest.mark.parametrize("typed", ["justproject", "/", "  ", "ws"])
    def test_a_string_with_no_pair_is_refused(self, typed: str) -> None:
        with pytest.raises(ValueError, match="workspace/project"):
            _parse_roboflow_dataset(typed)


class TestAnEmptyDownloadIsNotSuccess:
    """A finished export that produced nothing must say so.

    Roboflow prints "export complete" and Kaggle unzips without complaint, so
    the provider's own output looks fine; we then counted zero images and
    returned a result that also looked fine. The only clue was a `0 images` log
    line next to a green outcome, which is not something anyone can act on.
    """

    @staticmethod
    def _empty_roboflow(monkeypatch: pytest.MonkeyPatch) -> None:
        """A Roboflow whose download() writes an export with no images in it."""

        class FakeVersion:
            def download(
                self, fmt: str, location: str, overwrite: bool = False
            ) -> None:
                Path(location).mkdir(parents=True, exist_ok=True)
                (Path(location) / "README.roboflow.txt").write_text("empty")

        class FakeProject:
            def version(self, num: int) -> FakeVersion:
                return FakeVersion()

        class FakeWorkspace:
            def project(self, name: str) -> FakeProject:
                return FakeProject()

        class FakeRoboflow:
            def __init__(self, api_key: str) -> None:
                pass

            def workspace(self, ws: str) -> FakeWorkspace:
                return FakeWorkspace()

        mod = types.ModuleType("roboflow")
        mod.Roboflow = FakeRoboflow  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "roboflow", mod)

    def test_roboflow_names_the_folder_it_looked_in(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._empty_roboflow(monkeypatch)
        out = tmp_path / "ds"

        with pytest.raises(ValueError, match="no images landed") as exc:
            download_roboflow("ws/proj", out, api_key="KEY", version=1)

        assert str(out) in str(exc.value)

    def test_it_lists_what_did_arrive(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """So "the export has no images" is separable from "we cannot read them"."""
        self._empty_roboflow(monkeypatch)

        with pytest.raises(ValueError, match=r"\.txt"):
            download_roboflow("ws/proj", tmp_path / "ds", api_key="KEY", version=1)

    def test_a_download_with_images_still_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fake_roboflow(monkeypatch, {})

        result = download_roboflow("ws/proj", tmp_path / "ds", api_key="KEY", version=1)

        assert result.total_images == 2


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

    def test_saved_token_reaches_the_client_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The client reads `KAGGLE_API_TOKEN`, and only at import time.

        It used to be `KAGGLE_USERNAME` + `KAGGLE_KEY`; that pair appears zero
        times in kaggle 2.2.3, so setting it authenticated as nobody.
        """
        _install_fake_kaggle(monkeypatch, {})
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
        monkeypatch.setattr(
            "visionforge.gui.api.dataset_download.load_credential",
            lambda _provider: "KGAT_exemplo",
        )

        download_kaggle("owner/slug", tmp_path / "ds")

        assert os.environ["KAGGLE_API_TOKEN"] == "KGAT_exemplo"

    def test_the_old_username_key_pair_is_refused_with_an_explanation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Someone with the old credential saved gets told what to do.

        Passing it through would have failed inside Kaggle's client with an
        error about the token file, which points nowhere near the real cause.
        """
        _install_fake_kaggle(monkeypatch, {})
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
        monkeypatch.setattr(
            "visionforge.gui.api.dataset_download.load_credential",
            lambda _provider: "meu-usuario:minha-chave",
        )

        with pytest.raises(ValueError, match="KGAT_"):
            download_kaggle("owner/slug", tmp_path / "ds")

    def test_an_explicit_env_token_wins_over_the_saved_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fake_kaggle(monkeypatch, {})
        monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_do_ambiente")
        monkeypatch.setattr(
            "visionforge.gui.api.dataset_download.load_credential",
            lambda _provider: "KGAT_salvado",
        )

        download_kaggle("owner/slug", tmp_path / "ds")

        assert os.environ["KAGGLE_API_TOKEN"] == "KGAT_do_ambiente"


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


# --- credentials never leave in an error ------------------------------------

# A made-up value: every test here uses it, none uses a real credential.
SECRET = "SECRET123"


class _HTTPError(Exception):
    """Stands in for `requests.HTTPError` / `huggingface_hub`'s, by name only.

    The tests assert the exception *type name* survives redaction; a local class
    keeps `requests` (which ships no type stubs) out of the test imports.
    """


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Saved credentials live under VISIONFORGE_HOME; never touch the real one.
    monkeypatch.setenv("VISIONFORGE_HOME", str(tmp_path / "home"))
    from visionforge.gui.server import app

    return TestClient(app)


@pytest.fixture
def server_log() -> Any:
    """Everything loguru emitted while the test ran, tracebacks included.

    `diagnose=True` is deliberate: it is the setting that prints variable
    values inside a traceback, so it is the worst case for a key held in a local.
    """
    messages: list[Any] = []
    sink = logger.add(
        messages.append,
        level="DEBUG",
        backtrace=True,
        diagnose=True,
        format="{message}",
    )

    class _Log:
        @property
        def text(self) -> str:
            return "\n".join(str(m) for m in messages)

    yield _Log()
    logger.remove(sink)


def _post_download(client: TestClient, tmp_path: Path, **body: Any) -> Any:
    payload = {"dataset": "ws/proj", "out_dir": str(tmp_path / "ds"), **body}
    return client.post("/api/dataset/download", json=payload)


def _use_stored_kaggle_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Save the token the way the GUI does, and keep the env var from outliving the test.

    `download_kaggle` copies a saved token into `os.environ` directly, so the
    variable is registered with monkeypatch first (set, then delete) to be
    restored to "absent" at teardown.
    """
    from visionforge.utils.credentials import save_credential

    monkeypatch.setenv("KAGGLE_API_TOKEN", "placeholder")
    monkeypatch.delenv("KAGGLE_API_TOKEN")
    save_credential("kaggle", SECRET)


class TestADownloadErrorDoesNotLeakTheCredential:
    """The Roboflow client puts the key in the URL, so a dropped connection spells it out.

    `requests` reports "Max retries exceeded with url: /?api_key=<KEY>", and the
    route used to forward that text as the HTTP detail (shown in the GUI) and to
    `logger.exception` (written to the server log, traceback and all).
    """

    @pytest.mark.parametrize(
        "message",
        [
            f"Max retries exceeded with url: /?api_key={SECRET} (Caused by X)",
            f"401 Client Error: Unauthorized for url: https://hf.co/api/x?token={SECRET}",
            f"GET /v1/export?format=folder&access_token={SECRET}&v=1 failed",
            f"request headers {{'Authorization': 'Bearer {SECRET}'}}",
            f"the server rejected {SECRET} as invalid",
        ],
        ids=["api_key", "token", "access_token", "authorization", "literal"],
    )
    def test_the_detail_and_the_log_are_clean(
        self,
        client: TestClient,
        server_log: Any,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        message: str,
    ) -> None:
        def boom(*args: Any, **kwargs: Any) -> None:
            raise ConnectionError(message)

        monkeypatch.setattr("visionforge.gui.api.routes.download_dataset", boom)

        resp = _post_download(
            client, tmp_path, provider="roboflow", api_key=SECRET, version=1
        )

        assert resp.status_code == 500
        assert SECRET not in resp.text
        assert SECRET not in server_log.text

    def test_the_error_stays_useful(
        self,
        client: TestClient,
        server_log: Any,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Redacted is not silenced: type and context reach the screen and the log."""

        def boom(*args: Any, **kwargs: Any) -> None:
            raise ConnectionError(
                "HTTPSConnectionPool(host='api.roboflow.com', port=443): Max "
                f"retries exceeded with url: /?api_key={SECRET}"
            )

        monkeypatch.setattr("visionforge.gui.api.routes.download_dataset", boom)

        resp = _post_download(
            client, tmp_path, provider="roboflow", api_key=SECRET, version=1
        )

        detail = resp.json()["detail"]
        assert detail.startswith("ConnectionError: ")
        assert "api.roboflow.com" in detail
        assert "Max retries exceeded" in detail
        # The operator still gets the failure, with provider and cause.
        assert "roboflow" in server_log.text
        assert "ConnectionError" in server_log.text
        assert "Max retries exceeded" in server_log.text

    @pytest.mark.parametrize("exc_type", [ValueError, FileNotFoundError, ImportError])
    def test_a_400_detail_is_clean_too(
        self,
        client: TestClient,
        server_log: Any,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        exc_type: type[Exception],
    ) -> None:
        def boom(*args: Any, **kwargs: Any) -> None:
            raise exc_type(f"cannot fetch /?api_key={SECRET} for {SECRET}")

        monkeypatch.setattr("visionforge.gui.api.routes.download_dataset", boom)

        resp = _post_download(
            client, tmp_path, provider="roboflow", api_key=SECRET, version=1
        )

        assert resp.status_code == 400
        assert SECRET not in resp.text
        assert SECRET not in server_log.text

    @pytest.mark.parametrize("source", ["typed", "saved"])
    def test_roboflow_offline_with_the_real_download_path(
        self,
        client: TestClient,
        server_log: Any,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        source: str,
    ) -> None:
        """The key in use is either typed into the request or read from the store."""

        class OfflineRoboflow:
            def __init__(self, api_key: str) -> None:
                try:
                    raise OSError(f"getaddrinfo failed for /?api_key={api_key}")
                except OSError as inner:
                    raise ConnectionError(
                        "HTTPSConnectionPool(host='api.roboflow.com', port=443): Max "
                        f"retries exceeded with url: /?api_key={api_key}"
                    ) from inner

        fake = types.ModuleType("roboflow")
        fake.Roboflow = OfflineRoboflow  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "roboflow", fake)
        body: dict[str, Any] = {"provider": "roboflow", "version": 1}
        if source == "typed":
            body["api_key"] = SECRET
        else:
            from visionforge.utils.credentials import save_credential

            save_credential("roboflow", SECRET)

        resp = _post_download(client, tmp_path, **body)

        assert resp.status_code == 500
        assert "ConnectionError" in resp.json()["detail"]
        assert SECRET not in resp.text
        assert SECRET not in server_log.text

    @pytest.mark.parametrize("source", ["environment", "saved"])
    def test_kaggle_token_in_an_error(
        self,
        client: TestClient,
        server_log: Any,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        source: str,
    ) -> None:
        """Kaggle's token comes from the environment or the store, never the request."""
        if source == "environment":
            monkeypatch.setenv("KAGGLE_API_TOKEN", SECRET)
        else:
            _use_stored_kaggle_token(monkeypatch)

        class FailingKaggleApi:
            def authenticate(self) -> None:
                return None

            def dataset_download_files(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError(
                    f"401 Unauthorized: token {os.environ['KAGGLE_API_TOKEN']} rejected"
                )

        pkg = types.ModuleType("kaggle")
        api_mod = types.ModuleType("kaggle.api")
        ext = types.ModuleType("kaggle.api.kaggle_api_extended")
        ext.KaggleApi = FailingKaggleApi  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "kaggle", pkg)
        monkeypatch.setitem(sys.modules, "kaggle.api", api_mod)
        monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", ext)

        resp = _post_download(client, tmp_path, provider="kaggle", dataset="owner/slug")

        assert resp.status_code == 500
        assert "RuntimeError" in resp.json()["detail"]
        assert SECRET not in resp.text
        assert SECRET not in server_log.text

    @pytest.mark.parametrize("source", ["typed", "saved"])
    def test_huggingface_token_in_an_error(
        self,
        client: TestClient,
        server_log: Any,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        source: str,
    ) -> None:
        def load_dataset(name: str, token: str | None = None) -> Any:
            raise _HTTPError(
                f"401 Client Error: Unauthorized for url: https://huggingface.co/api/x?token={token}"
            )

        fake = types.ModuleType("datasets")
        fake.load_dataset = load_dataset  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "datasets", fake)
        body: dict[str, Any] = {"provider": "huggingface", "dataset": "owner/ds"}
        if source == "typed":
            body["token"] = SECRET
        else:
            from visionforge.utils.credentials import save_credential

            save_credential("huggingface", SECRET)

        resp = _post_download(client, tmp_path, **body)

        assert resp.status_code == 500
        assert "_HTTPError" in resp.json()["detail"]
        assert SECRET not in resp.text
        assert SECRET not in server_log.text

    @pytest.mark.parametrize("variable", ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"])
    def test_huggingface_token_read_from_the_environment(
        self,
        client: TestClient,
        server_log: Any,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        variable: str,
    ) -> None:
        """`load_dataset(token=None)` falls back to HF_TOKEN, so the request never carries it."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.setenv(variable, SECRET)

        def load_dataset(name: str, token: str | None = None) -> Any:
            raise _HTTPError(f"401 Unauthorized: {os.environ[variable]} is not valid")

        fake = types.ModuleType("datasets")
        fake.load_dataset = load_dataset  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "datasets", fake)

        resp = _post_download(
            client, tmp_path, provider="huggingface", dataset="owner/ds"
        )

        assert resp.status_code == 500
        assert "_HTTPError" in resp.json()["detail"]
        assert SECRET not in resp.text
        assert SECRET not in server_log.text


class TestCredentialsInPlay:
    """What the redaction is told to look for."""

    @pytest.fixture(autouse=True)
    def _clean_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VISIONFORGE_HOME", str(tmp_path / "home"))
        for name in ("KAGGLE_API_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            monkeypatch.delenv(name, raising=False)

    def test_the_request_values_are_included(self) -> None:
        assert credentials_in_play("rf_request_key", "hf_request_tok") == [
            "rf_request_key",
            "hf_request_tok",
        ]

    def test_saved_credentials_are_included(self) -> None:
        from visionforge.utils.credentials import save_credential

        save_credential("roboflow", "rf_saved_key_123")

        assert "rf_saved_key_123" in credentials_in_play(None, None)

    @pytest.mark.parametrize(
        "variable",
        ["KAGGLE_API_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"],
    )
    def test_the_environment_variables_the_clients_fall_back_to(
        self, monkeypatch: pytest.MonkeyPatch, variable: str
    ) -> None:
        monkeypatch.setenv(variable, "value_from_the_env")

        assert "value_from_the_env" in credentials_in_play(None, None)

    def test_nothing_in_play_is_an_empty_list(self) -> None:
        assert credentials_in_play(None, None) == []
