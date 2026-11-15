"""Security headers on every response, and the API docs only on request (SEC-2).

Without a Content-Security-Policy, a script that reaches the page (a dataset's
class name rendered somewhere unescaped, a compromised dependency) runs with the
page's full power and can send what it reads anywhere. Without
``frame-ancestors``/``X-Frame-Options`` another site can frame the GUI and trick
a click. The SPA's ``index.html`` also told Google about every visit through two
``preconnect`` links to fonts it never loads from there.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visionforge.gui.security import (
    SpaContentSecurityPolicy,
    content_security_policy,
    inline_script_hashes,
)
from visionforge.gui.server import STATIC_DIR, app

REPO = Path(__file__).resolve().parents[2]
SOURCE_INDEX = REPO / "frontend" / "index.html"
BUILT_INDEX = STATIC_DIR / "index.html"

HEADERS = {
    "x-frame-options": "DENY",
    "x-content-type-options": "nosniff",
    # Not ``no-referrer``: under it the Fetch standard sends ``Origin: null`` on
    # the SPA's own POSTs, which the Host/Origin guard (SEC-1) refuses.
    "referrer-policy": "same-origin",
}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _directives(csp: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for part in csp.split(";"):
        tokens = part.split()
        if tokens:
            out[tokens[0]] = tokens[1:]
    return out


def _first_asset() -> str:
    assets = sorted((STATIC_DIR / "assets").glob("index-*.js"))
    assert assets, "the versioned SPA bundle is missing"
    return f"/assets/{assets[0].name}"


def _sha256(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return "'sha256-" + base64.b64encode(digest).decode("ascii") + "'"


class TestHeadersOnEveryResponse:
    @pytest.mark.parametrize(
        "path", ["/", "/runs/some-deep-link", "/api/health", "/favicon.svg"]
    )
    def test_spa_api_and_static_files(self, client: TestClient, path: str) -> None:
        response = client.get(path)
        assert response.status_code == 200
        for name, value in HEADERS.items():
            assert response.headers.get(name) == value, name
        assert response.headers.get("content-security-policy")

    def test_bundle_asset(self, client: TestClient) -> None:
        response = client.get(_first_asset())
        assert response.status_code == 200
        for name, value in HEADERS.items():
            assert response.headers.get(name) == value, name
        assert response.headers.get("content-security-policy")

    def test_a_refused_request_carries_them_too(self, client: TestClient) -> None:
        response = client.get("/api/health", headers={"Host": "evil.example"})
        assert response.status_code == 400
        assert response.headers.get("x-frame-options") == "DENY"
        assert response.headers.get("content-security-policy")


class TestSpaPolicy:
    def test_only_this_server_and_no_inline_code(self, client: TestClient) -> None:
        csp = client.get("/").headers["content-security-policy"]
        directives = _directives(csp)
        assert directives["default-src"] == ["'self'"]
        assert directives["style-src"] == ["'self'"]
        assert directives["connect-src"] == ["'self'"]
        assert directives["frame-ancestors"] == ["'none'"]
        assert directives["object-src"] == ["'none'"]
        assert directives["base-uri"] == ["'self'"]
        assert directives["form-action"] == ["'self'"]
        # Dataset crops arrive as data: URIs; downloads are built as blob: URLs.
        assert directives["img-src"] == ["'self'", "data:", "blob:"]
        assert "'unsafe-inline'" not in csp
        assert "'unsafe-eval'" not in csp
        assert "http:" not in csp
        assert "https:" not in csp

    def test_the_theme_script_is_allowed_by_its_hash(self, client: TestClient) -> None:
        html = BUILT_INDEX.read_text(encoding="utf-8")
        inline = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
        assert len(inline) == 1, "index.html paints the theme with one inline script"
        csp = client.get("/").headers["content-security-policy"]
        assert _directives(csp)["script-src"] == ["'self'", _sha256(inline[0])]


class TestInlineScriptHashes:
    def test_hashes_only_scripts_without_src(self) -> None:
        html = (
            '<script type="module" src="/assets/index.js"></script>'
            "<script>a()</script>"
            '<SCRIPT type="text/javascript">\n b()\n</SCRIPT>'
        )
        assert inline_script_hashes(html) == [_sha256("a()"), _sha256("\n b()\n")]

    def test_line_endings_as_the_browser_reads_them(self) -> None:
        # The HTML parser turns CRLF into LF before the script is hashed, and a
        # Windows checkout may write index.html with CRLF.
        assert inline_script_hashes("<script>\r\na()\r\n</script>") == [
            _sha256("\na()\n")
        ]

    def test_policy_without_inline_scripts(self) -> None:
        directives = _directives(content_security_policy([]))
        assert directives["script-src"] == ["'self'"]

    def test_follows_a_rebuilt_index(self, tmp_path: Path) -> None:
        index = tmp_path / "index.html"
        policy = SpaContentSecurityPolicy(index)
        assert _directives(policy())["script-src"] == ["'self'"]
        index.write_text("<script>x()</script>", encoding="utf-8")
        assert _directives(policy())["script-src"] == ["'self'", _sha256("x()")]
        index.write_text("<script>y(1)</script>", encoding="utf-8")
        os.utime(index, ns=(1, 1))
        assert _directives(policy())["script-src"] == ["'self'", _sha256("y(1)")]


class TestNoThirdPartyHosts:
    @pytest.mark.parametrize(
        "index", [SOURCE_INDEX, BUILT_INDEX], ids=["source", "built"]
    )
    def test_index_html_names_no_other_host(self, index: Path) -> None:
        html = index.read_text(encoding="utf-8")
        assert "googleapis" not in html
        assert "gstatic" not in html
        assert re.findall(r"""(?:src|href)\s*=\s*["']?(?:https?:)?//""", html) == []


class TestApiDocs:
    @pytest.mark.parametrize("path", ["/docs", "/redoc", "/docs/oauth2-redirect"])
    def test_off_by_default(self, client: TestClient, path: str) -> None:
        response = client.get(path)
        assert response.status_code == 404
        assert 'id="root"' not in response.text

    def test_openapi_document_stays(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "/api/health" in response.json()["paths"]

    def test_on_with_visionforge_docs(self, tmp_path: Path) -> None:
        script = textwrap.dedent("""
            from fastapi.testclient import TestClient
            from visionforge.gui import security
            from visionforge.gui.server import app

            security.DEFAULT_POLICY = security.HostPolicy(
                extra=frozenset({"testserver"})
            )
            client = TestClient(app)
            for path in ("/docs", "/redoc"):
                response = client.get(path)
                assert response.status_code == 200, (path, response.status_code)
                assert "cdn.jsdelivr.net" in response.text, path
                csp = response.headers["content-security-policy"]
                assert "https://cdn.jsdelivr.net" in csp, csp
                assert "frame-ancestors 'none'" in csp, csp
            spa = client.get("/").headers["content-security-policy"]
            assert "cdn.jsdelivr.net" not in spa, spa
            assert client.get("/openapi.json").status_code == 200
        """)
        env = {**os.environ, "VISIONFORGE_DOCS": "1", "PYTHONPATH": str(REPO / "src")}
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
