"""HTTP checks for committing a History run to a private GitHub repository (ADR-175)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from visionforge.core.github_export import GitHubExportError
from visionforge.core.hub_export import PrivateRepositoryRequiredError
from visionforge.gui.api import github_routes, profile_tokens, routes
from visionforge.gui.server import app

_TOKEN = "github-test-token-123456"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("VISIONFORGE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(profile_tokens, "is_loopback_host", lambda _: True)
    run_dir = tmp_path / "models" / "gatos_001"
    run_dir.mkdir(parents=True)
    record = {"experiment": "gatos", "metrics": {"accuracy": 0.9}, "artifacts": {}}
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return TestClient(app)


def _export(client: TestClient, repo: str = "ana/runs") -> Any:
    digest = client.get("/api/github/preview", params={"run_id": "gatos_001"}).json()[
        "digest"
    ]
    return client.post(
        "/api/github/export",
        json={"run_id": "gatos_001", "repo": repo, "digest": digest},
    )


def test_the_preview_lists_the_folder_and_the_report(client: TestClient) -> None:
    response = client.get("/api/github/preview", params={"run_id": "gatos_001"})

    assert response.status_code == 200
    body = response.json()
    assert body["folder"] == "runs/gatos/gatos_001"
    assert [file["path"] for file in body["files"]] == ["README.md", "run.json"]
    assert body["readme"].startswith("# gatos_001")


def test_an_unknown_run_is_not_found(client: TestClient) -> None:
    assert client.get("/api/github/preview", params={"run_id": "x"}).status_code == 404


def test_a_token_is_needed_before_the_commit(client: TestClient) -> None:
    response = _export(client)
    assert response.status_code == 409
    assert "token do GitHub" in response.json()["detail"]


def test_a_malformed_repository_is_refused(client: TestClient) -> None:
    client.put("/api/github/token", json={"token": _TOKEN})
    response = _export(client, repo="sem-barra")
    assert response.status_code == 422
    assert "usuário/repositório" in response.json()["detail"]


def test_the_commit_uses_the_profile_token(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_export(directory: Path, repo: str, token: str, digest: str) -> Any:
        calls.append((repo, token))
        return {"url": "https://github.com/ana/runs/tree/main/runs/gatos/gatos_001"}

    monkeypatch.setattr(github_routes, "export_run", fake_export)
    client.put("/api/github/token", json={"token": _TOKEN})

    response = _export(client)

    assert response.status_code == 200
    assert calls == [("ana/runs", _TOKEN)]


@pytest.mark.parametrize(
    ("error", "code", "message"),
    [
        (GitHubExportError("x", 401), 502, "recusou o token"),
        (GitHubExportError("x", 403), 502, "Contents"),
        (GitHubExportError("x", 404), 502, "Crie o repositório privado"),
        (GitHubExportError("x", 422), 502, "falhou"),
        (GitHubExportError("x"), 502, "falhou"),
        (PrivateRepositoryRequiredError("x"), 409, "precisa ser privado"),
    ],
)
def test_each_failure_has_its_own_message(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    code: int,
    message: str,
) -> None:
    def refuse(*_: Any, **__: Any) -> Any:
        raise error

    monkeypatch.setattr(github_routes, "export_run", refuse)
    client.put("/api/github/token", json={"token": _TOKEN})

    response = _export(client)

    assert response.status_code == code
    assert message in response.json()["detail"]
    assert _TOKEN not in response.text


def test_a_changed_run_asks_for_a_new_preview(client: TestClient) -> None:
    client.put("/api/github/token", json={"token": _TOKEN})
    response = client.post(
        "/api/github/export",
        json={"run_id": "gatos_001", "repo": "ana/runs", "digest": "stale"},
    )
    assert response.status_code == 409
    assert "prévia" in response.json()["detail"]


def test_a_missing_or_invalid_record(client: TestClient, tmp_path: Path) -> None:
    record = tmp_path / "models" / "gatos_001" / "run.json"
    record.write_text("[]", encoding="utf-8")
    assert (
        client.get("/api/github/preview", params={"run_id": "gatos_001"}).status_code
        == 422
    )
    record.unlink()
    assert (
        client.get("/api/github/preview", params={"run_id": "gatos_001"}).status_code
        == 404
    )


def test_a_malformed_body_never_echoes_the_token(client: TestClient) -> None:
    response = client.put("/api/github/token", json={"token": [_TOKEN]})
    assert response.status_code == 422
    assert _TOKEN not in response.text
