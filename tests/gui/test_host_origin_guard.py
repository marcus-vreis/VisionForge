"""Host and Origin guard: DNS rebinding and cross-site requests (SEC-1).

A page of another site reaches the local server through the user's own
browser. Rebinding makes the browser send the attacker's name in ``Host``; a
cross-site form sends the attacker's page in ``Origin``. Both are refused here
before any route runs.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from visionforge.gui.security import HostOriginGuard, HostPolicy, normalize_host

LAB_NAMES = ["labbox", "labbox.lab.example", "192.168.1.20", "fe80::1"]


@pytest.fixture
def app() -> Iterator[Any]:
    """The real app under the loopback-only policy; its state restored after.

    ``start_server`` also records ``bind_host``, which other routes read, so
    both are put back as they were.
    """
    from visionforge.gui.server import app

    names = ("host_policy", "bind_host")
    saved = {
        name: getattr(app.state, name) for name in names if hasattr(app.state, name)
    }
    app.state.host_policy = HostPolicy()
    yield app
    for name in names:
        if name in saved:
            setattr(app.state, name, saved[name])
        elif hasattr(app.state, name):
            delattr(app.state, name)


def _client(app: Any, host: str) -> TestClient:
    """A client whose requests carry ``Host: host``.

    Set as a header rather than through ``base_url``: TestClient cannot parse a
    bracketed IPv6 address there.
    """
    return TestClient(app, base_url="http://127.0.0.1:8000", headers={"Host": host})


class TestLoopbackServer:
    @pytest.mark.parametrize(
        "host",
        ["evil.example:8000", "evil.example", "localhost.evil.example:8000"],
    )
    def test_a_foreign_host_is_refused(self, app: Any, host: str) -> None:
        resp = _client(app, host).get("/api/health")

        assert resp.status_code == 400
        assert "version" not in resp.json()

    def test_the_spa_is_refused_under_a_foreign_host_too(self, app: Any) -> None:
        resp = _client(app, "evil.example:8000").get("/")

        assert resp.status_code == 400

    @pytest.mark.parametrize(
        "host",
        [
            "localhost:8000",
            "127.0.0.1:8000",
            "[::1]:8000",
            "localhost",
            "127.0.0.1",
            "[::1]",
            "LOCALHOST:8000",
        ],
    )
    def test_loopback_hosts_are_served(self, app: Any, host: str) -> None:
        resp = _client(app, host).get("/api/health")

        assert resp.status_code == 200
        assert "version" in resp.json()

    def test_the_machines_name_is_not_accepted_on_loopback(self, app: Any) -> None:
        app.state.host_policy = HostPolicy.for_bind(
            "127.0.0.1", names_of_machine=LAB_NAMES
        )

        assert _client(app, "labbox:8000").get("/api/health").status_code == 400

    def test_the_message_follows_the_page_language(self, app: Any) -> None:
        client = _client(app, "evil.example:8000")

        pt = client.get("/api/health")
        en = client.get("/api/health", headers={"X-VF-Lang": "en"})

        assert pt.json()["detail"].startswith("Pedido recusado")
        assert en.json()["detail"].startswith("Request refused")
        assert pt.json()["detail"] != en.json()["detail"]


class TestForeignOrigin:
    @pytest.mark.parametrize(
        "origin",
        [
            "http://evil.example",
            "https://evil.example:8000",
            "http://127.0.0.1.evil.example:8000",
            "http://192.168.1.20:8000",
            "null",
            "not a url",
        ],
    )
    def test_post_and_delete_with_a_foreign_origin_are_403(
        self, app: Any, origin: str
    ) -> None:
        client = _client(app, "127.0.0.1:8000")
        headers = {"Origin": origin}

        post = client.post("/api/experiment/run", json={}, headers=headers)
        delete = client.delete("/api/queue/no_such_run", headers=headers)

        assert (post.status_code, delete.status_code) == (403, 403)
        assert post.json()["detail"] == (
            "Pedido recusado: ele veio de uma página que não é do VisionForge."
        )

    def test_the_origin_message_follows_the_page_language(self, app: Any) -> None:
        resp = _client(app, "127.0.0.1:8000").delete(
            "/api/queue/no_such_run",
            headers={"Origin": "http://evil.example", "X-VF-Lang": "en"},
        )

        assert resp.status_code == 403
        assert resp.json()["detail"] == (
            "Request refused: it came from a page that is not VisionForge's."
        )

    @pytest.mark.parametrize(
        "origin",
        [
            "http://localhost:5173",
            "http://127.0.0.1:8000",
            "http://[::1]:8000",
        ],
    )
    def test_a_loopback_origin_reaches_the_route(self, app: Any, origin: str) -> None:
        client = _client(app, "127.0.0.1:8000")
        headers = {"Origin": origin}

        post = client.post("/api/experiment/run", json={}, headers=headers)
        delete = client.delete("/api/queue/no_such_run", headers=headers)

        assert (post.status_code, delete.status_code) == (422, 404)

    def test_no_origin_reaches_the_route(self, app: Any) -> None:
        resp = _client(app, "127.0.0.1:8000").delete("/api/queue/no_such_run")

        assert resp.status_code == 404

    def test_a_read_with_a_foreign_origin_is_served(self, app: Any) -> None:
        """A GET changes nothing; CORS already keeps the answer from the page."""
        resp = _client(app, "127.0.0.1:8000").get(
            "/api/health", headers={"Origin": "http://evil.example"}
        )

        assert resp.status_code == 200


class TestLabServer:
    """``--host 0.0.0.0`` (ADR-114): the machine's own names and addresses work."""

    @pytest.fixture
    def lab(self, app: Any) -> Any:
        app.state.host_policy = HostPolicy.for_bind(
            "0.0.0.0", ["vf.lab.example"], names_of_machine=LAB_NAMES
        )
        return app

    @pytest.mark.parametrize(
        "host",
        [
            "labbox:8000",
            "LabBox.Lab.Example:8000",
            "192.168.1.20:8000",
            "[fe80::1]:8000",
            "vf.lab.example",
            "localhost:8000",
        ],
    )
    def test_the_declared_names_are_served(self, lab: Any, host: str) -> None:
        assert _client(lab, host).get("/api/health").status_code == 200

    @pytest.mark.parametrize("host", ["evil.example:8000", "0.0.0.0:8000"])
    def test_any_other_host_is_refused(self, lab: Any, host: str) -> None:
        assert _client(lab, host).get("/api/health").status_code == 400

    def test_the_pages_own_origin_reaches_the_route(self, lab: Any) -> None:
        resp = _client(lab, "labbox:8000").delete(
            "/api/queue/no_such_run", headers={"Origin": "http://labbox:8000"}
        )

        assert resp.status_code == 404

    def test_a_foreign_origin_is_403(self, lab: Any) -> None:
        resp = _client(lab, "labbox:8000").delete(
            "/api/queue/no_such_run", headers={"Origin": "http://evil.example"}
        )

        assert resp.status_code == 403

    def test_a_specific_bind_address_is_accepted(self, app: Any) -> None:
        app.state.host_policy = HostPolicy.for_bind("10.1.2.3", names_of_machine=[])

        assert _client(app, "10.1.2.3:8000").get("/api/health").status_code == 200
        assert _client(app, "10.1.2.4:8000").get("/api/health").status_code == 400


class TestWebSocket:
    """No route upgrades today; a future one must not skip the guard."""

    @staticmethod
    def _app() -> Any:
        async def echo(ws: WebSocket) -> None:
            await ws.accept()
            await ws.send_text("ok")
            await ws.close()

        return Starlette(
            routes=[WebSocketRoute("/ws", echo)],
            middleware=[],
        )

    def test_a_foreign_origin_handshake_is_closed(self) -> None:
        guarded = HostOriginGuard(self._app())
        client = TestClient(guarded, base_url="http://127.0.0.1:8000")

        def handshake() -> None:
            with client.websocket_connect(
                "/ws", headers={"Origin": "http://evil.example"}
            ) as ws:
                ws.receive_text()

        with pytest.raises(WebSocketDisconnect):
            handshake()

    def test_a_loopback_handshake_is_served(self) -> None:
        guarded = HostOriginGuard(self._app())
        client = TestClient(guarded, base_url="http://127.0.0.1:8000")

        with client.websocket_connect(
            "/ws", headers={"Origin": "http://127.0.0.1:8000"}
        ) as ws:
            assert ws.receive_text() == "ok"


class TestNormalizeHost:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("localhost:8000", "localhost"),
            ("LOCALHOST", "localhost"),
            ("[::1]:8000", "::1"),
            ("[::1]", "::1"),
            ("::1", "::1"),
            ("127.0.0.1:8000", "127.0.0.1"),
            ("labbox.", "labbox"),
        ],
    )
    def test_spellings(self, value: str, expected: str) -> None:
        assert normalize_host(value) == expected

    @pytest.mark.parametrize(
        "value", ["", None, "[::1", "[::1]x", "localhost:http", "[nope]:80", ":80"]
    )
    def test_malformed_is_none(self, value: str | None) -> None:
        assert normalize_host(value) is None


class TestStartServer:
    def test_installs_the_policy_and_opens_a_reachable_page(self, app: Any) -> None:
        import visionforge.gui.server as server_mod

        with (
            patch("uvicorn.Server") as fake_server,
            patch("webbrowser.open") as opened,
            patch("visionforge.gui.security.machine_names", return_value={"labbox"}),
        ):
            server_mod.start_server(
                host="0.0.0.0", port=8123, allowed_hosts=["vf.lab.example"]
            )

        policy = app.state.host_policy
        assert {"labbox", "vf.lab.example"} <= policy.extra
        assert "0.0.0.0" not in policy.extra
        # 0.0.0.0 is not a page a browser can open, nor a Host the guard accepts.
        opened.assert_called_once_with("http://127.0.0.1:8123")
        fake_server.return_value.run.assert_called_once()

    def test_cli_passes_every_allowed_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from visionforge.__main__ import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "visionforge",
                "gui",
                "--host",
                "0.0.0.0",
                "--allowed-host",
                "a.example",
                "--allowed-host",
                "b.example",
                "--no-browser",
            ],
        )
        with patch("visionforge.gui.server.start_server") as start:
            main()

        assert start.call_args.kwargs["allowed_hosts"] == ["a.example", "b.example"]
        assert start.call_args.kwargs["host"] == "0.0.0.0"
