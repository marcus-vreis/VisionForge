"""Static requests cannot escape the SPA directory."""

import re
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


@pytest.mark.parametrize("meta", ["description", "theme-color", "color-scheme"])
def test_built_index_declares_its_metas(meta: str) -> None:
    """The index.html the server delivers carries a description, theme-color and color-scheme."""
    html = server.STATIC_DIR.joinpath("index.html").read_text(encoding="utf-8")

    assert re.search(rf'<meta\s+name="{meta}"\s+content="[^"]+"', html)


_CHUNK_BUDGET_BYTES = 500_000


def test_no_bundled_script_exceeds_the_chunk_budget() -> None:
    """The SPA ships split by panel: no single script outgrows 500 kB (GUI-12)."""
    scripts = list(server.STATIC_DIR.joinpath("assets").glob("*.js"))
    assert scripts, "the SPA must be built for this test to mean anything"

    largest = max(scripts, key=lambda p: p.stat().st_size)

    assert largest.stat().st_size <= _CHUNK_BUDGET_BYTES, largest.name


def test_index_html_points_at_exactly_one_entry_script() -> None:
    """Lazy chunks are fetched by the entry, never listed beside it in index.html."""
    html = server.STATIC_DIR.joinpath("index.html").read_text(encoding="utf-8")

    entries = re.findall(r"assets/index-[A-Za-z0-9_-]+\.js", html)

    assert len(entries) == 1, entries
    assert server.STATIC_DIR.joinpath(entries[0]).is_file()
