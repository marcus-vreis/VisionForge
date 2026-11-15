"""HTTP integration checks for profile-owned private Hub exports."""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from visionforge.core.hub_export import (
    PreviewMismatchError,
    PrivateRepositoryRequiredError,
)
from visionforge.gui.api import hub_routes, profile_tokens, routes
from visionforge.gui.server import app
from visionforge.utils.messages import tr


def test_credentials_are_private_to_workspace_and_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONFORGE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    # The real bind host lives on app.state, which newer Starlette does not let
    # monkeypatch restore; the route only asks whether that host is loopback.
    loopback = {"value": False}
    monkeypatch.setattr(profile_tokens, "is_loopback_host", lambda _: loopback["value"])
    client = TestClient(app)
    assert client.post("/api/profiles", json={"name": "Lab"}).status_code == 201
    url = "/api/hub/token"
    headers = {"X-VF-Profile": "lab"}
    response = client.put(url, headers=headers, json={"token": "example-private-token"})
    assert response.json() == {"configured": True, "shared": True}
    assert client.get(url).json() == {"configured": False, "shared": True}
    assert client.delete(url).status_code == 200
    assert client.get(url, headers=headers).json()["configured"]
    loopback["value"] = True
    assert client.delete(url, headers=headers).json() == {
        "configured": False,
        "shared": False,
    }
    assert not any(p.is_file() for p in (tmp_path / "home").rglob("*"))
    assert client.put(url, json={"token": " "}).status_code == 422


def test_export_never_uses_ambient_login(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "ambient-test-token")
    monkeypatch.setenv("VISIONFORGE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    directory = tmp_path / "models" / "run"
    directory.mkdir(parents=True)
    (directory / "run.json").write_text("{}", encoding="utf-8")
    client = TestClient(app)
    payload = {"run_id": "run", "repo_id": "owner/model", "digest": "test"}
    assert client.post("/api/hub/export", json=payload).status_code == 409
    assert client.get("/api/hub/preview", params={"run_id": "other"}).status_code == 404


def test_malformed_token_payload_never_echoes_secret():
    client = TestClient(app)
    for body in ({"value": "canary-secret"}, {"token": ["canary-secret"]}):
        response = client.put("/api/hub/token", json=body)
        assert response.status_code == 422
        assert "canary-secret" not in response.text


def _client_with_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("VISIONFORGE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    directory = tmp_path / "models" / "run"
    directory.mkdir(parents=True)
    (directory / "run.json").write_text("{}", encoding="utf-8")
    return TestClient(app)


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        (FileNotFoundError("model.pt"), 404, tr("hub.artifact_missing")),
        (ValueError("no card"), 422, tr("hub.package_invalid", reason="no card")),
    ],
)
def test_preview_failures_map_to_statuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status: int,
    detail: str,
) -> None:
    client = _client_with_run(tmp_path, monkeypatch)

    def fail(directory: Path) -> dict[str, Any]:
        raise error

    monkeypatch.setattr(hub_routes, "preview_run", fail)
    response = client.get("/api/hub/preview", params={"run_id": "run"})
    assert response.status_code == status
    assert response.json()["detail"] == detail


def test_preview_returns_the_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client = _client_with_run(tmp_path, monkeypatch)
    seen: list[Path] = []

    def preview(directory: Path) -> dict[str, Any]:
        seen.append(directory)
        return {"digest": "abc"}

    monkeypatch.setattr(hub_routes, "preview_run", preview)
    response = client.get("/api/hub/preview", params={"run_id": "run"})
    assert response.json() == {"digest": "abc"}
    assert seen == [tmp_path / "models" / "run"]


def test_an_invalid_repo_id_is_refused_before_the_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_with_run(tmp_path, monkeypatch)
    payload = {"run_id": "missing", "repo_id": "not a repo", "digest": "d"}
    response = client.post("/api/hub/export", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"] == tr("hub.repo_invalid")


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        (FileNotFoundError("model.pt"), 404, tr("hub.artifact_missing")),
        (PreviewMismatchError("changed"), 409, tr("hub.preview_changed")),
        (PrivateRepositoryRequiredError("public"), 409, tr("hub.private_required")),
        (ValueError("no card"), 422, tr("hub.package_invalid", reason="no card")),
        (RuntimeError("network"), 502, tr("hub.export_failed")),
    ],
)
def test_export_failures_map_to_statuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status: int,
    detail: str,
) -> None:
    client = _client_with_run(tmp_path, monkeypatch)
    monkeypatch.setattr(hub_routes, "read_token", lambda kind, profile: "profile-tok")

    def fail(directory: Path, repo_id: str, token: str, digest: str) -> dict[str, Any]:
        raise error

    monkeypatch.setattr(hub_routes, "export_run", fail)
    payload = {"run_id": "run", "repo_id": "owner/model", "digest": "d"}
    response = client.post("/api/hub/export", json=payload)
    assert response.status_code == status
    assert response.json()["detail"] == detail


def test_export_sends_with_the_profile_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client_with_run(tmp_path, monkeypatch)
    monkeypatch.setattr(hub_routes, "read_token", lambda kind, profile: "profile-tok")
    sent: list[tuple[Path, str, str, str]] = []

    def export(
        directory: Path, repo_id: str, token: str, digest: str
    ) -> dict[str, Any]:
        sent.append((directory, repo_id, token, digest))
        return {"url": "https://huggingface.co/owner/model"}

    monkeypatch.setattr(hub_routes, "export_run", export)
    payload = {"run_id": "run", "repo_id": "owner/model", "digest": "d"}
    response = client.post("/api/hub/export", json=payload)
    assert response.json() == {"url": "https://huggingface.co/owner/model"}
    assert sent == [(tmp_path / "models" / "run", "owner/model", "profile-tok", "d")]
