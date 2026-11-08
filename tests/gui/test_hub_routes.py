"""HTTP integration checks for profile-owned private Hub exports."""

from pathlib import Path

from fastapi.testclient import TestClient

from visionforge.gui.api import routes
from visionforge.gui.server import app


def test_credentials_are_private_to_workspace_and_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONFORGE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(app.state, "bind_host", "0.0.0.0", raising=False)
    client = TestClient(app)
    assert client.post("/api/profiles", json={"name": "Lab"}).status_code == 201
    url = "/api/hub/token"
    headers = {"X-VF-Profile": "lab"}
    response = client.put(url, headers=headers, json={"token": "example-private-token"})
    assert response.json() == {"configured": True, "shared": True}
    assert client.get(url).json() == {"configured": False, "shared": True}
    assert client.delete(url).status_code == 200
    assert client.get(url, headers=headers).json()["configured"]
    monkeypatch.setattr(app.state, "bind_host", "127.0.0.1")
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
