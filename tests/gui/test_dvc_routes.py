"""HTTP checks for versioning a History run with DVC (ADR-176).

git is the real one; DVC is the ``FakeDvc`` of the core tests.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.core.test_dvc_export import FakeDvc
from visionforge.core.dvc_export import Tools
from visionforge.gui.api import dvc_routes, routes
from visionforge.gui.server import app

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture
def fake() -> FakeDvc:
    return FakeDvc()


@pytest.fixture
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeDvc
) -> TestClient:
    monkeypatch.setenv("VISIONFORGE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(dvc_routes, "is_loopback_host", lambda _: True)
    tools = Tools(git=["git"], dvc=["fake-dvc"], run=fake)
    monkeypatch.setattr(dvc_routes, "find_tools", lambda: tools)
    empty = tmp_path / "gitconfig"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Ana")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "ana@example.org")
    run_dir = tmp_path / "models" / "gatos_001"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_bytes(b"weights")
    record = {
        "experiment": "gatos",
        "config": {"task": "binary"},
        "metrics": {"accuracy": 0.9},
        "artifacts": {"model": str(run_dir / "best.pt")},
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return TestClient(app)


def _repo(tmp_path: Path) -> str:
    return str(tmp_path / "experimentos")


def _preview(client: TestClient, repo: str) -> Any:
    return client.get("/api/dvc/preview", params={"run_id": "gatos_001", "repo": repo})


def test_prepare_then_export_commits_the_run(
    client: TestClient, tmp_path: Path
) -> None:
    repo = _repo(tmp_path)

    prepared = client.post("/api/dvc/prepare", json={"repo": repo})
    preview = _preview(client, repo).json()
    exported = client.post(
        "/api/dvc/export",
        json={"run_id": "gatos_001", "repo": repo, "digest": preview["digest"]},
    )

    assert prepared.status_code == 200
    assert prepared.json()["state"] == "ready"
    assert preview["folder"] == "runs/gatos/gatos_001"
    assert {"path": "model.pt", "size": 7, "storage": "dvc"} in preview["files"]
    assert exported.status_code == 200
    body = exported.json()
    assert len(body["commit"]) == 40
    assert body["pushed"] is None
    assert (tmp_path / "experimentos" / "runs/gatos/gatos_001/model.pt.dvc").is_file()


def test_the_preview_says_the_folder_is_not_ready(
    client: TestClient, tmp_path: Path
) -> None:
    response = _preview(client, _repo(tmp_path))

    assert response.status_code == 200
    assert response.json()["repository"]["state"] == "missing"
    assert response.json()["repository"]["tools"] == {"git": True, "dvc": True}


def test_an_export_to_a_folder_that_is_not_ready_is_a_conflict(
    client: TestClient, tmp_path: Path
) -> None:
    repo = _repo(tmp_path)
    digest = _preview(client, repo).json()["digest"]

    response = client.post(
        "/api/dvc/export", json={"run_id": "gatos_001", "repo": repo, "digest": digest}
    )

    assert response.status_code == 409
    assert "Preparar pasta" in response.json()["detail"]


def test_a_stale_preview_is_a_conflict(client: TestClient, tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    client.post("/api/dvc/prepare", json={"repo": repo})

    response = client.post(
        "/api/dvc/export", json={"run_id": "gatos_001", "repo": repo, "digest": "x"}
    )

    assert response.status_code == 409
    assert "mudaram depois da prévia" in response.json()["detail"]


def test_a_relative_path_is_refused(client: TestClient) -> None:
    response = client.post("/api/dvc/prepare", json={"repo": "experimentos"})

    assert response.status_code == 422
    assert "caminho completo" in response.json()["detail"]


def test_a_file_cannot_be_prepared(client: TestClient, tmp_path: Path) -> None:
    file = tmp_path / "file.txt"
    file.write_text("x", encoding="utf-8")

    response = client.post("/api/dvc/prepare", json={"repo": str(file)})

    assert response.status_code == 409
    assert "é um arquivo" in response.json()["detail"]


def test_an_unknown_run_is_not_found(client: TestClient, tmp_path: Path) -> None:
    response = client.get(
        "/api/dvc/preview", params={"run_id": "x", "repo": _repo(tmp_path)}
    )

    assert response.status_code == 404


def test_missing_dvc_is_named(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dvc_routes, "find_tools", lambda: Tools(git=["git"], dvc=None))

    response = client.post("/api/dvc/prepare", json={"repo": _repo(tmp_path)})

    assert response.status_code == 409
    assert "visionforge-studio[dvc]" in response.json()["detail"]


def test_a_failed_dvc_add_is_a_server_error_without_its_output(
    client: TestClient, tmp_path: Path, fake: FakeDvc
) -> None:
    repo = _repo(tmp_path)
    client.post("/api/dvc/prepare", json={"repo": repo})
    digest = _preview(client, repo).json()["digest"]
    fake.add_ok = False

    response = client.post(
        "/api/dvc/export", json={"run_id": "gatos_001", "repo": repo, "digest": digest}
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "O DVC falhou em add. O log do servidor tem a saída."
    )


def test_a_client_on_another_machine_is_refused(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dvc_routes, "is_loopback_host", lambda _: False)

    preview = _preview(client, _repo(tmp_path))
    prepare = client.post("/api/dvc/prepare", json={"repo": _repo(tmp_path)})

    assert (preview.status_code, prepare.status_code) == (403, 403)
    assert "computador onde o VisionForge roda" in prepare.json()["detail"]
    assert not (tmp_path / "experimentos").exists()


def test_a_page_of_another_site_is_refused(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/api/dvc/prepare",
        json={"repo": _repo(tmp_path)},
        headers={"Origin": "https://example.com"},
    )

    assert response.status_code == 403
    assert not (tmp_path / "experimentos").exists()


def test_a_malformed_body_does_not_echo_it(client: TestClient) -> None:
    response = client.post("/api/dvc/export", json={"repo": "C:/x"})

    assert response.status_code == 422
    assert response.json()["detail"] == "Pedido inválido para o DVC."
