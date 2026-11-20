"""Server mode: a person lists and deletes the datasets of their own folder (ADR-227).

The own folder counts against the disk quota (ADR-225); deleting a dataset is
how a person gets under it again without asking the operator.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from visionforge.gui.api import execution, routes
from visionforge.gui.disk_quota import folder_bytes
from visionforge.gui.server_mode import ROUTE_CLASSES, ServerMode, answers, user_slug

PROXY = "10.0.0.5"
ANA = {"X-Forwarded-User": "ana@lab.org"}
BIA = {"X-Forwarded-User": "bia@lab.org"}
MIB = 1024 * 1024


def _serve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quota_gb: float | None = None
) -> Iterator[TestClient]:
    from visionforge.gui.server import app

    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    shared = tmp_path / "shared"
    shared.mkdir()
    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    app.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY], datasets_dirs=[shared], quota_gb=quota_gb
    )
    yield TestClient(app, base_url="http://localhost", client=(PROXY, 50000))
    if had:
        app.state.server_mode = saved
    else:
        del app.state.server_mode


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    yield from _serve(tmp_path, monkeypatch)


@pytest.fixture
def tight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """The same server with one mebibyte per person."""
    yield from _serve(tmp_path, monkeypatch, quota_gb=1 / 1024)


def _own(client: TestClient, tmp_path: Path, headers: dict[str, str]) -> Path:
    client.get("/api/profiles", headers=headers)  # makes the profile and its folder
    slug = user_slug(headers["X-Forwarded-User"])
    return tmp_path / "profiles" / slug / "datasets"


def _dataset(own: Path, name: str, size: int = 1000) -> Path:
    folder = own / name
    (folder / "train" / "a").mkdir(parents=True)
    (folder / "train" / "a" / "img.bin").write_bytes(b"\0" * size)
    return folder


def _names(client: TestClient, headers: dict[str, str]) -> list[str]:
    listed = client.get("/api/datasets/own", headers=headers)
    assert listed.status_code == 200, listed.text
    return [d["name"] for d in listed.json()["datasets"]]


def test_the_route_classes() -> None:
    for key in ("GET /api/datasets/own", "DELETE /api/datasets/own/{name}"):
        assert ROUTE_CLASSES[key] == "owner"
        assert answers(key)


def test_a_person_lists_the_datasets_of_their_own_folder_with_sizes(
    client: TestClient, tmp_path: Path
) -> None:
    own = _own(client, tmp_path, ANA)
    _dataset(own, "beans", 3000)
    _dataset(own, "Apples", 500)
    (own / "notes.txt").write_text("not a dataset", encoding="utf-8")
    _dataset(_own(client, tmp_path, BIA), "secret")
    _dataset(tmp_path / "shared", "everyone")

    listed = client.get("/api/datasets/own", headers=ANA).json()["datasets"]
    assert [d["name"] for d in listed] == ["Apples", "beans"]
    assert [d["size_bytes"] for d in listed] == [500, 3000]
    assert Path(listed[1]["path"]) == own / "beans"
    assert _names(client, BIA) == ["secret"]


def test_deleting_removes_the_folder_and_only_that_one(
    client: TestClient, tmp_path: Path
) -> None:
    own = _own(client, tmp_path, ANA)
    beans = _dataset(own, "beans")
    _dataset(own, "apples")
    theirs = _dataset(_own(client, tmp_path, BIA), "beans")

    gone = client.delete("/api/datasets/own/beans", headers=ANA)
    assert gone.status_code == 200, gone.text
    assert gone.json() == {"name": "beans", "status": "deleted"}
    assert not beans.exists()
    assert _names(client, ANA) == ["apples"]
    # Bia's dataset of the same name is hers.
    assert theirs.is_dir()
    again = client.delete("/api/datasets/own/beans", headers=ANA)
    assert again.status_code == 404
    assert "beans" in again.json()["detail"]


@pytest.mark.parametrize(
    "name",
    ["..", ".", "..%2F..%2Fmodels", "%2E%2E", "a%5Cb", "C%3A", "notes.txt", "nope"],
)
def test_a_name_that_is_not_one_of_their_folders_deletes_nothing(
    client: TestClient, tmp_path: Path, name: str
) -> None:
    own = _own(client, tmp_path, ANA)
    kept = _dataset(own, "beans")
    (own / "notes.txt").write_text("keep", encoding="utf-8")
    before = folder_bytes(own.parent)

    refused = client.delete(f"/api/datasets/own/{name}", headers=ANA)

    assert refused.status_code == 404, refused.text
    assert kept.is_dir()
    assert (own / "notes.txt").is_file()
    assert (own.parent / "models").is_dir()
    assert folder_bytes(own.parent) == before


def test_a_link_out_of_the_folder_is_neither_listed_nor_followed(
    client: TestClient, tmp_path: Path
) -> None:
    own = _own(client, tmp_path, ANA)
    shared = _dataset(tmp_path / "shared", "everyone")
    try:
        (own / "link").symlink_to(shared, target_is_directory=True)
    except OSError:
        pytest.skip("this account cannot create symbolic links")
    assert _names(client, ANA) == []
    assert client.delete("/api/datasets/own/link", headers=ANA).status_code == 404
    assert shared.is_dir()


def test_a_dataset_is_kept_while_a_training_of_theirs_runs_or_waits(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    own = _own(client, tmp_path, ANA)
    beans = _dataset(own, "beans")
    slug = user_slug(ANA["X-Forwarded-User"])
    lanes: list[dict[str, Any]] = [
        {"lane": "local", "active": None, "pending": [{"profile": slug}]}
    ]
    monkeypatch.setattr(execution._LANES, "snapshots", lambda: lanes)

    held = client.delete("/api/datasets/own/beans", headers=ANA)
    assert held.status_code == 409, held.text
    assert "fila" in held.json()["detail"]
    assert beans.is_dir()

    # Somebody else's training does not hold hers.
    lanes[0] = {"lane": "local", "active": {"profile": "someone-else"}, "pending": []}
    assert client.delete("/api/datasets/own/beans", headers=ANA).status_code == 200
    assert not beans.exists()


def test_deleting_a_dataset_gets_a_full_folder_under_the_quota(
    tight: TestClient, tmp_path: Path
) -> None:
    own = _own(tight, tmp_path, ANA)
    _dataset(own, "heavy", 2 * MIB)
    body = {"task": "classification"}
    assert tight.post("/api/sample-dataset", json=body, headers=ANA).status_code == 507

    assert tight.delete("/api/datasets/own/heavy", headers=ANA).status_code == 200

    made = tight.post("/api/sample-dataset", json=body, headers=ANA)
    assert made.status_code == 201, made.text


def test_a_local_install_has_no_own_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from visionforge.gui.server import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "datasets" / "mine").mkdir(parents=True)
    local = TestClient(app, base_url="http://localhost")
    assert local.get("/api/datasets/own").json() == {"datasets": []}
    assert local.delete("/api/datasets/own/mine").status_code == 404
    assert (tmp_path / "datasets" / "mine").is_dir()
