"""Server mode: a run leaves the server through its owner only (ADR-221).

The ZIP, the Markdown card and the three publishers (Hugging Face Hub, Zenodo,
GitHub) take a run id, package files from inside that run's folder and, for a
publisher, use the token its owner saved. DVC writes into a folder of the
server's own disk with the server's git, so it stays off.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from visionforge.gui.api import github_routes, hub_routes, routes, zenodo_routes
from visionforge.gui.server_mode import ROUTE_CLASSES, ServerMode, answers, user_slug

PROXY = "10.0.0.5"
ANA = {"X-Forwarded-User": "ana@lab.org"}
BIA = {"X-Forwarded-User": "bia@lab.org"}
RUN_ID = "20261119_150000_000001"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """The real app in server mode; tokens are kept under ``tmp_path``."""
    from visionforge.gui.api import profile_tokens
    from visionforge.gui.server import app

    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(profile_tokens, "config_dir", lambda: tmp_path / "config")
    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    app.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY]
    )
    yield TestClient(app, base_url="http://localhost", client=(PROXY, 50000))
    if had:
        app.state.server_mode = saved
    else:
        del app.state.server_mode


@pytest.fixture
def run(client: TestClient, tmp_path: Path) -> Path:
    """A finished run in ana's folder, with a checkpoint and a plot of its own."""
    client.get("/api/session", headers=ANA)
    client.get("/api/session", headers=BIA)
    models = tmp_path / "profiles" / user_slug("ana@lab.org") / "models"
    run_dir = models / "beans" / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "best_model.pth").write_bytes(b"weights")
    (run_dir / "loss.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    record = {
        "run_id": RUN_ID,
        "experiment": "beans",
        "status": "completed",
        "config": {
            "name": "beans",
            "task": "multiclass",
            "model": {"name": "resnet18"},
        },
        "metrics": {"accuracy": 0.9},
        "history": [],
        "artifacts": {
            "model": str(run_dir / "best_model.pth"),
            "graphics": [str(run_dir / "loss.png")],
            "report": None,
        },
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return run_dir


def test_the_owner_downloads_the_run_as_a_zip_and_nobody_else_does(
    client: TestClient, run: Path
) -> None:
    mine = client.get(f"/api/runs/{RUN_ID}/bundle", headers=ANA)
    assert mine.status_code == 200, mine.text
    names = zipfile.ZipFile(io.BytesIO(mine.content)).namelist()
    assert "run.json" in names
    assert client.get(f"/api/runs/{RUN_ID}/bundle", headers=BIA).status_code == 404


def test_the_owner_downloads_the_markdown_card_and_nobody_else_does(
    client: TestClient, run: Path
) -> None:
    mine = client.get(f"/api/runs/{RUN_ID}/export_md", headers=ANA)
    assert mine.status_code == 200, mine.text
    assert "beans" in mine.text
    assert mine.headers["content-type"].startswith("text/markdown")
    assert client.get(f"/api/runs/{RUN_ID}/export_md", headers=BIA).status_code == 404


@pytest.mark.parametrize("service", ["hub", "zenodo", "github"])
def test_a_preview_of_the_package_is_the_owners(
    client: TestClient, run: Path, service: str
) -> None:
    url = f"/api/{service}/preview"
    assert client.get(url, params={"run_id": RUN_ID}, headers=BIA).status_code == 404
    mine = client.get(url, params={"run_id": RUN_ID}, headers=ANA)
    # Read offline from inside the run folder; a tiny fake run may be refused
    # as a package (422), which is still an answer about ana's own run.
    assert mine.status_code in (200, 422), mine.text


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every upload is recorded, never made."""
    calls: list[dict[str, Any]] = []

    def hub(directory: Path, repo_id: str, token: str, digest: str) -> dict[str, Any]:
        calls.append({"service": "hub", "run": directory.name, "token": token})
        return {"url": "https://example.invalid"}

    def zenodo(directory: Path, token: str, digest: str, **kw: Any) -> dict[str, Any]:
        calls.append({"service": "zenodo", "run": directory.name, "token": token})
        return {"url": "https://example.invalid"}

    def github(directory: Path, repo: str, token: str, digest: str) -> dict[str, Any]:
        calls.append({"service": "github", "run": directory.name, "token": token})
        return {"url": "https://example.invalid"}

    monkeypatch.setattr(hub_routes, "export_run", hub)
    monkeypatch.setattr(zenodo_routes, "export_run", zenodo)
    monkeypatch.setattr(github_routes, "export_run", github)
    return calls


BODIES = {
    "hub": {"run_id": RUN_ID, "repo_id": "ana/beans", "digest": "d"},
    "zenodo": {"run_id": RUN_ID, "digest": "d"},
    "github": {"run_id": RUN_ID, "repo": "ana/beans", "digest": "d"},
}


@pytest.mark.parametrize("service", sorted(BODIES))
def test_a_run_is_published_with_its_owners_token_and_only_by_its_owner(
    client: TestClient, run: Path, sent: list[dict[str, Any]], service: str
) -> None:
    export = f"/api/{service}/export"
    # No token saved yet: nothing is sent, and nobody else's token is borrowed.
    client.put(f"/api/{service}/token", json={"token": "BIA-TOKEN"}, headers=BIA)
    assert client.post(export, json=BODIES[service], headers=ANA).status_code == 409
    # Bia has a token, and ana's run does not exist for her.
    assert client.post(export, json=BODIES[service], headers=BIA).status_code == 404
    assert sent == []

    saved = client.put(
        f"/api/{service}/token", json={"token": "ANA-TOKEN"}, headers=ANA
    )
    assert saved.status_code in (200, 204), saved.text
    published = client.post(export, json=BODIES[service], headers=ANA)
    assert published.status_code == 200, published.text
    assert sent == [{"service": service, "run": RUN_ID, "token": "ANA-TOKEN"}]


def test_dvc_stays_off_on_a_server(client: TestClient, run: Path) -> None:
    """It runs the server's git and DVC on a folder of the server's disk."""
    for key in (
        "POST /api/dvc/export",
        "GET /api/dvc/preview",
        "POST /api/dvc/prepare",
    ):
        assert (ROUTE_CLASSES[key], answers(key)) == ("off", False)
    preview = client.get(
        "/api/dvc/preview", params={"run_id": RUN_ID, "repo": "/srv/x"}, headers=ANA
    )
    assert preview.status_code == 404


def test_the_session_lists_what_still_does_not_leave_the_server(
    client: TestClient,
) -> None:
    closed = client.get("/api/session", headers=ANA).json()["closed"]
    for key in (
        "GET /api/runs/{run_id}/bundle",
        "GET /api/runs/{run_id}/export_md",
        "POST /api/hub/export",
        "POST /api/zenodo/export",
        "POST /api/github/export",
    ):
        assert key not in closed
    assert "POST /api/dvc/export" in closed
    assert "POST /api/runs/{run_id}/mlflow/retry" in closed
