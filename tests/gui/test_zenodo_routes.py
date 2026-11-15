"""HTTP checks for Zenodo drafts of a History run (ADR-174)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from visionforge.core.zenodo_export import ZenodoExportError
from visionforge.gui.api import profile_tokens, routes, zenodo_routes
from visionforge.gui.server import app

_TOKEN = "zenodo-test-token-123456"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("VISIONFORGE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(profile_tokens, "is_loopback_host", lambda _: True)
    run_dir = tmp_path / "models" / "gatos_001"
    run_dir.mkdir(parents=True)
    (run_dir / "best_model.pth").write_bytes(b"weights")
    record = {"artifacts": {"model": "best_model.pth"}, "metrics": {"accuracy": 0.9}}
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return TestClient(app)


def _export(client: TestClient, sandbox: bool = False) -> Any:
    digest = client.get("/api/zenodo/preview", params={"run_id": "gatos_001"}).json()[
        "digest"
    ]
    return client.post(
        "/api/zenodo/export",
        json={"run_id": "gatos_001", "digest": digest, "sandbox": sandbox},
    )


def test_the_preview_names_the_bundle(client: TestClient) -> None:
    response = client.get("/api/zenodo/preview", params={"run_id": "gatos_001"})

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "visionforge-gatos_001.zip"
    assert body["files"] == ["best_model.pth", "manifest.json", "run.json"]


def test_an_unknown_run_is_not_found(client: TestClient) -> None:
    response = client.get("/api/zenodo/preview", params={"run_id": "other"})
    assert response.status_code == 404


def test_each_site_has_its_own_token(client: TestClient) -> None:
    saved = client.put("/api/zenodo/sandbox/token", json={"token": _TOKEN})

    assert saved.json() == {"configured": True, "shared": False}
    assert client.get("/api/zenodo/token").json()["configured"] is False
    assert _export(client).status_code == 409


def test_the_draft_uses_the_token_of_the_chosen_site(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_export(directory: Path, token: str, digest: str, *, sandbox: bool) -> Any:
        calls.append({"run": directory.name, "token": token, "sandbox": sandbox})
        return {
            "id": 1,
            "url": "https://sandbox.zenodo.org/deposit/1",
            "sandbox": sandbox,
        }

    monkeypatch.setattr(zenodo_routes, "export_run", fake_export)
    client.put("/api/zenodo/sandbox/token", json={"token": _TOKEN})

    response = _export(client, sandbox=True)

    assert response.status_code == 200
    assert response.json()["url"] == "https://sandbox.zenodo.org/deposit/1"
    assert calls == [{"run": "gatos_001", "token": _TOKEN, "sandbox": True}]


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (401, "recusou o token"),
        (403, "recusou o token"),
        (500, "falhou"),
        (None, "falhou"),
    ],
)
def test_a_zenodo_failure_is_a_bad_gateway_with_a_clear_message(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    status: int | None,
    message: str,
) -> None:
    def refuse(*_: Any, **__: Any) -> Any:
        raise ZenodoExportError("Zenodo answered", status)

    monkeypatch.setattr(zenodo_routes, "export_run", refuse)
    client.put("/api/zenodo/token", json={"token": _TOKEN})

    response = _export(client)

    assert response.status_code == 502
    assert message in response.json()["detail"]
    assert _TOKEN not in response.text


def test_a_changed_run_asks_for_a_new_preview(client: TestClient) -> None:
    client.put("/api/zenodo/token", json={"token": _TOKEN})
    response = client.post(
        "/api/zenodo/export", json={"run_id": "gatos_001", "digest": "stale"}
    )
    assert response.status_code == 409
    assert "prévia" in response.json()["detail"]


def test_a_run_without_a_record_cannot_be_previewed(
    client: TestClient, tmp_path: Path
) -> None:
    (tmp_path / "models" / "gatos_001" / "run.json").unlink()
    response = client.get("/api/zenodo/preview", params={"run_id": "gatos_001"})
    assert response.status_code == 404


def test_an_invalid_record_is_unprocessable(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "models" / "gatos_001" / "run.json").write_text("[]", encoding="utf-8")
    response = client.get("/api/zenodo/preview", params={"run_id": "gatos_001"})
    assert response.status_code == 422
    client.put("/api/zenodo/token", json={"token": _TOKEN})
    exported = client.post(
        "/api/zenodo/export", json={"run_id": "gatos_001", "digest": "x"}
    )
    assert exported.status_code == 422


def test_a_malformed_body_never_echoes_the_token(client: TestClient) -> None:
    for url in ("/api/zenodo/token", "/api/zenodo/sandbox/token"):
        response = client.put(url, json={"token": [_TOKEN]})
        assert response.status_code == 422
        assert _TOKEN not in response.text
