"""index.html must never be cached, or a new build serves the old app.

Vite fingerprints asset filenames, so a bundle is safe to cache forever. The one
file whose name never changes is index.html, and it is the file that names the
current bundle — so caching it is how a browser keeps running yesterday's app
against today's server. That has been mistaken for a broken build twice.
"""

from __future__ import annotations

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
