"""index.html must never be cached, or a new build serves the old app.

Vite fingerprints asset filenames, so a bundle is safe to cache forever. The one
file whose name never changes is index.html, and it is the file that names the
current bundle — so caching it is how a browser keeps running yesterday's app
against today's server. That has been mistaken for a broken build twice.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from visionforge.gui.server import app


class TestIndexCaching:
    def test_index_is_served_with_no_cache(self) -> None:
        client = TestClient(app)

        resp = client.get("/")

        assert resp.status_code == 200
        assert "no-cache" in resp.headers.get("cache-control", "")

    def test_an_unknown_route_falls_back_to_a_no_cache_index(self) -> None:
        """Deep links hit the SPA fallback, which must not be cached either."""
        client = TestClient(app)

        resp = client.get("/alguma/rota/do/spa")

        assert resp.status_code == 200
        assert "no-cache" in resp.headers.get("cache-control", "")

    def test_a_fingerprinted_asset_is_not_forced_to_revalidate(self) -> None:
        """Those names change per build, so caching them is the point."""
        from visionforge.gui.server import STATIC_DIR

        assets = list((STATIC_DIR / "assets").glob("index-*.js"))
        assert assets, "the SPA must be built for this test to mean anything"

        client = TestClient(app)
        resp = client.get(f"/assets/{assets[0].name}")

        assert resp.status_code == 200
        assert "no-cache" not in resp.headers.get("cache-control", "")


class TestHealthReportsTheBootBundle:
    """The signal that a server outlived the build it is serving."""

    def test_health_names_the_bundle_index_html_points_at(self) -> None:
        import re

        from visionforge.gui.server import STATIC_DIR

        index_html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        expected = re.search(r"assets/(index-[A-Za-z0-9_-]+\.js)", index_html)
        assert expected, "the SPA must be built for this test to mean anything"

        resp = TestClient(app).get("/api/health")

        assert resp.status_code == 200
        assert resp.json()["spa_bundle"] == expected.group(1)

    def test_health_reports_the_version(self) -> None:
        from visionforge import __version__

        assert TestClient(app).get("/api/health").json()["version"] == __version__

    def test_a_missing_build_reports_an_empty_bundle_rather_than_raising(
        self, tmp_path: Path
    ) -> None:
        """An unbuilt checkout must not warn; it has no name to compare."""
        from unittest.mock import patch

        from visionforge.gui import server as server_mod

        with patch.object(server_mod, "STATIC_DIR", tmp_path):
            assert server_mod._read_spa_bundle() == ""


class TestHealthReportsTheUser:
    """The greeting's name comes from the OS, not from a settings field."""

    def test_health_carries_a_user(self) -> None:
        body = TestClient(app).get("/api/health").json()

        assert "user" in body
        assert isinstance(body["user"], str)

    def test_an_unknowable_user_is_empty_rather_than_an_error(self) -> None:
        """A greeting is not worth failing a request over."""
        from unittest.mock import patch

        from visionforge.gui import server as server_mod

        with patch.object(
            server_mod.getpass, "getuser", side_effect=OSError("no account")
        ):
            assert server_mod._current_user() == ""
