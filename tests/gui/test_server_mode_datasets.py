"""Server mode: a person puts a dataset on the server, in their own folder only (ADR-219).

The sample dataset and a download are the two ways. Both write under the
signed-in person's dataset folder, and a download uses the key the request
brings, never the one saved on the machine, which is the operator's.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from visionforge.gui.api import dataset_download, datasets_routes, routes
from visionforge.gui.api.dataset_download import DatasetDownloadResult
from visionforge.gui.api.dataset_scope import DatasetScope, dataset_scope
from visionforge.gui.api.schemas import DatasetDownloadRequest
from visionforge.gui.server_mode import ServerMode, user_slug

PROXY = "10.0.0.5"
ANA = {"X-Forwarded-User": "ana@lab.org"}
BIA = {"X-Forwarded-User": "bia@lab.org"}


@pytest.fixture
def folders(tmp_path: Path) -> dict[str, Path]:
    shared = tmp_path / "shared"
    shared.mkdir()
    own = tmp_path / "profiles" / user_slug("ana@lab.org") / "datasets"
    return {"shared": shared.resolve(), "own": own.resolve(), "root": tmp_path}


def _scope(folders: dict[str, Path]) -> DatasetScope:
    folders["own"].mkdir(parents=True, exist_ok=True)
    return DatasetScope(
        readable=(folders["own"], folders["shared"]), owner="ana", own=folders["own"]
    )


def test_a_dataset_is_written_only_inside_the_persons_own_folder(
    folders: dict[str, Path], tmp_path: Path
) -> None:
    def request(out_dir: Path | str) -> DatasetDownloadRequest:
        return DatasetDownloadRequest(dataset="cifar10", out_dir=str(out_dir))

    with dataset_scope(_scope(folders)):
        inside = request(folders["own"] / "cifar10").out_dir
        assert Path(inside) == folders["own"] / "cifar10"
        for refused in (
            folders["own"],  # the folder itself: two downloads would mix
            folders["shared"] / "cifar10",  # readable, not writable
            folders["own"] / ".." / "other" / "x",
            tmp_path / "elsewhere",
            "cifar10",
            "~/cifar10",
            " ",
        ):
            with pytest.raises(ValidationError):
                request(refused)
    assert request("cifar10").out_dir == "cifar10"  # the local tool writes anywhere


def test_a_scope_with_no_own_folder_writes_nowhere(folders: dict[str, Path]) -> None:
    with (
        dataset_scope(DatasetScope(readable=(folders["shared"],))),
        pytest.raises(ValidationError),
    ):
        DatasetDownloadRequest(dataset="x", out_dir=str(folders["shared"] / "x"))


@pytest.fixture
def providers(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The provider clients are recorded; the machine has a key saved for each."""
    calls: list[dict[str, Any]] = []

    def record(provider: str) -> Any:
        def download(dataset: str, out_dir: str, **kw: Any) -> DatasetDownloadResult:
            calls.append({"provider": provider, **kw})
            return DatasetDownloadResult(
                provider=provider, dataset=dataset, out_dir=out_dir, total_images=1
            )

        return download

    for name in ("roboflow", "huggingface", "kaggle"):
        monkeypatch.setattr(dataset_download, f"download_{name}", record(name))
    monkeypatch.setattr(dataset_download, "load_credential", lambda _: "OPERATOR-KEY")
    monkeypatch.setattr(
        dataset_download, "resolve_credential", lambda _, given: given or "OPERATOR-KEY"
    )
    return calls


def test_on_a_server_a_download_never_uses_the_key_saved_on_the_machine(
    folders: dict[str, Path], providers: list[dict[str, Any]]
) -> None:
    fetch = dataset_download.download_dataset
    with dataset_scope(_scope(folders)):
        fetch("roboflow", dataset="w/p", out_dir="x", api_key=None, version=1)
        fetch("huggingface", dataset="o/d", out_dir="x", token=None)
        fetch("roboflow", dataset="w/p", out_dir="x", api_key=" mine ", version=1)
        fetch("huggingface", dataset="o/d", out_dir="x", token="mine")
    assert [call.get("api_key", call.get("token")) for call in providers] == [
        None,  # Roboflow then answers "a key is needed"
        False,  # tells the Hub client to send no token, not to look for one
        "mine",
        "mine",
    ]


def test_on_a_local_install_the_saved_key_is_still_used(
    providers: list[dict[str, Any]],
) -> None:
    dataset_download.download_dataset(
        "roboflow", dataset="w/p", out_dir="x", api_key=None, version=1
    )
    dataset_download.download_dataset("kaggle", dataset="o/d", out_dir="x")
    assert providers[0]["api_key"] == "OPERATOR-KEY"
    assert providers[1]["provider"] == "kaggle"


def test_kaggle_is_refused_on_a_server(
    folders: dict[str, Path], providers: list[dict[str, Any]]
) -> None:
    """Its client reads the key from the process environment, which is shared."""
    with dataset_scope(_scope(folders)), pytest.raises(ValueError, match="Kaggle"):
        dataset_download.download_dataset("kaggle", dataset="o/d", out_dir="x")
    assert providers == []


@pytest.fixture
def served(
    folders: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, list[dict[str, Any]]]]:
    """The real app in server mode; a download is recorded, never fetched."""
    from visionforge.gui.server import app

    fetched: list[dict[str, Any]] = []

    def fetch(provider: str, **kw: Any) -> DatasetDownloadResult:
        fetched.append({"provider": provider, **kw})
        return DatasetDownloadResult(
            provider=provider,
            dataset=kw["dataset"],
            out_dir=kw["out_dir"],
            total_images=3,
        )

    monkeypatch.setattr(datasets_routes, "download_dataset", fetch)
    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    app.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY], datasets_dirs=[folders["shared"]]
    )
    yield (
        TestClient(app, base_url="http://localhost", client=(PROXY, 50000)),
        fetched,
    )
    if had:
        app.state.server_mode = saved
    else:
        del app.state.server_mode


def test_a_download_into_the_own_folder_reaches_the_provider(
    served: tuple[TestClient, list[dict[str, Any]]], folders: dict[str, Path]
) -> None:
    client, fetched = served
    body = {"dataset": "cifar10", "out_dir": str(folders["own"] / "cifar10")}
    response = client.post("/api/dataset/download", json=body, headers=ANA)
    assert response.status_code == 200, response.text
    assert [Path(call["out_dir"]) for call in fetched] == [folders["own"] / "cifar10"]


@pytest.mark.parametrize("where", ["shared", "someone_else", "outside"])
def test_a_download_anywhere_else_is_refused_before_anything_is_fetched(
    served: tuple[TestClient, list[dict[str, Any]]],
    folders: dict[str, Path],
    where: str,
) -> None:
    client, fetched = served
    client.get("/api/session", headers=BIA)  # bia's folder exists
    target = {
        "shared": folders["shared"] / "cifar10",
        "someone_else": folders["own"].parent.parent
        / user_slug("bia@lab.org")
        / "datasets"
        / "cifar10",
        "outside": folders["root"] / "elsewhere",
    }[where]
    body = {"dataset": "cifar10", "out_dir": str(target)}
    body["api_key"] = "SECRET-KEY"
    response = client.post("/api/dataset/download", json=body, headers=ANA)
    assert (response.status_code, fetched) == (422, [])
    # The answer says which rule, and repeats nothing the request carried.
    assert "datasets" in response.json()["detail"]
    assert "SECRET-KEY" not in response.text
    assert str(target) not in response.text


def test_a_malformed_download_still_says_nothing_of_its_body(
    served: tuple[TestClient, list[dict[str, Any]]], folders: dict[str, Path]
) -> None:
    client, fetched = served
    body = {"dataset": 7, "out_dir": str(folders["own"] / "x"), "api_key": "SECRET"}
    response = client.post("/api/dataset/download", json=body, headers=ANA)
    assert (response.status_code, fetched) == (422, [])
    assert "SECRET" not in response.text


def test_the_sample_dataset_lands_in_the_persons_own_folder(
    served: tuple[TestClient, list[dict[str, Any]]], folders: dict[str, Path]
) -> None:
    client, _ = served
    response = client.post(
        "/api/sample-dataset", json={"task": "classification"}, headers=ANA
    )
    assert response.status_code == 201, response.text
    written = Path(response.json()["path"])
    assert written.parent == folders["own"]
    assert (written / "train").is_dir()
    # Each person gets their own copy: nothing of ana's is reused for bia.
    theirs = client.post(
        "/api/sample-dataset", json={"task": "classification"}, headers=BIA
    )
    assert theirs.status_code == 201
    assert Path(theirs.json()["path"]).parent != folders["own"]
    # And what was written is a dataset the person can read back.
    stats = client.post(
        "/api/dataset/stats", json={"base_dir": str(written)}, headers=ANA
    )
    assert stats.status_code == 200


def test_the_session_no_longer_lists_these_routes_as_closed(
    served: tuple[TestClient, list[dict[str, Any]]],
) -> None:
    client, _ = served
    closed = client.get("/api/session", headers=ANA).json()["closed"]
    assert "POST /api/dataset/download" not in closed
    assert "POST /api/sample-dataset" not in closed
    assert "POST /api/credentials" in closed
    assert "POST /api/explorer/summary" in closed
