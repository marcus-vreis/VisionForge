"""Static requests cannot escape the SPA directory."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visionforge.gui import server


@pytest.mark.parametrize(
    "url",
    ["/%2e%2e/canary.txt", "/..%5ccanary.txt", "/assets/%2e%2e/%2e%2e/canary.txt"],
)
def test_parent_paths_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    """A readable file outside the root is never an asset."""
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("INDEX", encoding="utf-8")
    (tmp_path / "canary.txt").write_text("PRIVATE", encoding="utf-8")
    monkeypatch.setattr(server, "STATIC_DIR", static)

    response = TestClient(server.app).get(url)

    assert response.status_code == 404
    assert "PRIVATE" not in response.text


def test_internal_asset_and_deep_link_still_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Containment preserves assets and client-side routing."""
    (tmp_path / "index.html").write_text("INDEX", encoding="utf-8")
    (tmp_path / "app.js").write_text("ASSET", encoding="utf-8")
    monkeypatch.setattr(server, "STATIC_DIR", tmp_path)
    client = TestClient(server.app)

    assert client.get("/app.js").text == "ASSET"
    assert client.get("/history/run-1").text == "INDEX"
