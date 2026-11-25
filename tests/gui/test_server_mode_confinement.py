"""Server mode: a request cannot write outside, nor probe, what lies beyond its scope.

A download writes each split under the folder it names, and nothing above it,
even when the split names come from the request or from the provider
(ADR-219). The routes that serve a file, a thumbnail or an audit answer the
same for a place outside the scope whether it exists or not (ADR-214).
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from visionforge.gui.api import routes
from visionforge.gui.server_mode import ServerMode, user_slug

PROXY = "10.0.0.5"
ANA = {"X-Forwarded-User": "ana@lab.org"}
BIA = {"X-Forwarded-User": "bia@lab.org"}


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from visionforge.gui.server import app

    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    shared = tmp_path / "shared"
    shared.mkdir()
    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    app.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY], datasets_dirs=[shared]
    )
    yield TestClient(app, base_url="http://localhost", client=(PROXY, 50000))
    if had:
        app.state.server_mode = saved
    else:
        del app.state.server_mode


def _own(client: TestClient, tmp_path: Path, headers: dict[str, str]) -> Path:
    client.get("/api/profiles", headers=headers)  # makes the profile and its folder
    slug = user_slug(headers["X-Forwarded-User"])
    return (tmp_path / "profiles" / slug / "datasets").resolve()


def _png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (200, 30, 30)).save(path)
    return path


def _files(root: Path) -> set[Path]:
    return {p for p in root.rglob("*") if p.is_file()}


class _FakeMNIST:
    """Stands in for ``torchvision.datasets.MNIST``: two classes, no network."""

    classes = ["0", "1"]

    def __init__(self, root: str, train: bool, download: bool) -> None:
        self._n = 2

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return Image.new("RGB", (4, 4), (idx, 0, 0)), idx % 2


def _install_fake_hf(
    monkeypatch: pytest.MonkeyPatch, split_names: list[str], class_names: list[str]
) -> None:
    """Fake ``datasets`` and ``huggingface_hub`` whose names come from the Hub."""

    class HfApi:
        def dataset_info(self, repo_id: str, **_: Any) -> Any:
            return types.SimpleNamespace(sha="a" * 40)

    class Image_:  # noqa: N801 - the feature introspection matches the class name
        pass

    Image_.__name__ = "Image"

    class ClassLabel:
        def __init__(self, names: list[str]) -> None:
            self.names = names

    class FakeSplit:
        features = {"image": Image_(), "label": ClassLabel(class_names)}

        def __iter__(self) -> Iterator[dict[str, Any]]:
            for i in range(len(class_names)):
                yield {"image": Image.new("RGB", (4, 4)), "label": i}

    def load_dataset(name: str, **_: Any) -> dict[str, Any]:
        return {split: FakeSplit() for split in split_names}

    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = HfApi  # type: ignore[attr-defined]
    fake = types.ModuleType("datasets")
    fake.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "datasets", fake)


@pytest.mark.parametrize("where", ["climbing", "absolute"])
def test_a_download_split_cannot_leave_the_target_folder(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    where: str,
) -> None:
    monkeypatch.setattr("torchvision.datasets.MNIST", _FakeMNIST)
    own = _own(client, tmp_path, ANA)
    theirs = _own(client, tmp_path, BIA)
    split = {
        "climbing": f"../../../{theirs.parent.name}/datasets/stolen",
        "absolute": str(tmp_path / "anywhere"),
    }[where]
    body = {
        "provider": "torchvision",
        "dataset": "mnist",
        "out_dir": str(own / "mnist"),
        "splits": ["train", split],
        "limit": 1,
    }
    response = client.post("/api/dataset/download", json=body, headers=ANA)
    assert response.status_code == 422, response.text
    assert _files(theirs) == set()
    assert not (tmp_path / "anywhere").exists()
    assert not (own / "mnist").exists()


@pytest.mark.parametrize(
    ("split_names", "class_names"),
    [
        (["train", "../../escaped"], ["cat", "dog"]),
        (["train"], ["cat", ".."]),
        (["train"], ["cat", "..\\..\\..\\escaped"]),
    ],
    ids=["split-climbs", "class-is-parent", "class-backslash-climbs"],
)
def test_names_from_the_hub_cannot_leave_the_target_folder(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    split_names: list[str],
    class_names: list[str],
) -> None:
    _install_fake_hf(monkeypatch, split_names, class_names)
    own = _own(client, tmp_path, ANA)
    out = own / "hub"
    body = {"provider": "huggingface", "dataset": "o/d", "out_dir": str(out)}
    before = {p for p in _files(tmp_path) if not p.is_relative_to(out)}
    response = client.post("/api/dataset/download", json=body, headers=ANA)
    assert response.status_code == 400, response.text
    after = {p for p in _files(tmp_path) if not p.is_relative_to(out)}
    assert after == before
    # Every image that was written sits in a class folder of a split folder.
    for image in out.rglob("*.png"):
        assert image.relative_to(out).parts[0] in split_names
        assert len(image.relative_to(out).parts) == 3


def test_a_file_outside_the_scope_gets_one_answer_whether_it_exists_or_not(
    client: TestClient, tmp_path: Path
) -> None:
    theirs = _own(client, tmp_path, BIA)
    image = _png(theirs / "beans" / "a.png")
    profile = theirs.parent / "profile.json"
    assert profile.is_file()
    answers = set()
    for path in (image, profile, theirs / "beans" / "missing.png", tmp_path / "no"):
        got = client.get("/api/dataset/file", params={"path": str(path)}, headers=ANA)
        answers.add((got.status_code, got.text))
    assert len(answers) == 1, answers
    assert next(iter(answers))[0] == 403


def test_a_thumbnail_root_outside_the_scope_gets_one_answer(
    client: TestClient, tmp_path: Path
) -> None:
    theirs = _own(client, tmp_path, BIA)
    _png(theirs / "beans" / "a.png")
    answers = set()
    for root in (theirs / "beans", theirs / "missing"):
        got = client.get(
            "/api/explorer/thumb",
            params={"root": str(root), "path": "a.png"},
            headers=ANA,
        )
        answers.add((got.status_code, got.text))
    assert len(answers) == 1, answers
    assert next(iter(answers))[0] == 403


def test_an_audit_split_cannot_probe_outside_the_dataset(
    client: TestClient, tmp_path: Path
) -> None:
    own = _own(client, tmp_path, ANA)
    theirs = _own(client, tmp_path, BIA)
    _png(own / "beans" / "images" / "train" / "a.png")
    (theirs / "secret" / "images").mkdir(parents=True)
    answers = set()
    for name in ("secret", "missing"):
        body = {
            "base_dir": str(own / "beans"),
            "layout": "yolo",
            "splits": {"train": f"../../../{theirs.parent.name}/datasets/{name}"},
        }
        got = client.post("/api/dataset/audit", json=body, headers=ANA)
        answers.add((got.status_code, got.text.replace(name, "<name>")))
    assert len(answers) == 1, answers
    assert next(iter(answers))[0] == 422
