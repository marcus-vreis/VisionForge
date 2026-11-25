"""Server mode behind a login proxy on the same machine (ADR-213 amendment).

uvicorn trusts ``X-Forwarded-For`` from 127.0.0.1 by default and replaces the
client address with it. The server-mode gate recognises the proxy by the
address that connected, so with the proxy on loopback every request arrived
as the browser's address and was refused (403).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from visionforge.gui.server_mode import ServerMode


@pytest.fixture
def server_app() -> Iterator[Any]:
    from visionforge.gui.server import app

    names = ("host_policy", "bind_host", "server_mode")
    saved = {n: getattr(app.state, n) for n in names if hasattr(app.state, n)}
    yield app
    for name in names:
        if name in saved:
            setattr(app.state, name, saved[name])
        elif hasattr(app.state, name):
            delattr(app.state, name)


def _served_app(mode: ServerMode) -> Any:
    """The ASGI app uvicorn would serve, as ``start_server`` configures it."""
    import visionforge.gui.server as server_mod

    with patch("uvicorn.Server") as fake_server, patch("webbrowser.open"):
        server_mod.start_server(host="127.0.0.1", port=8123, server_mode=mode)
    config = fake_server.call_args.args[0]
    config.load()
    return config.loaded_app


async def _get(app: Any, headers: dict[str, str]) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 50123))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1:8123"
    ) as client:
        return await client.get("/api/session", headers=headers)


def test_a_proxy_on_loopback_that_forwards_the_browser_address_is_answered(
    server_app: Any,
) -> None:
    mode = ServerMode.from_options(enabled=True, trusted_proxies=["127.0.0.1"])
    response = asyncio.run(
        _get(
            _served_app(mode),
            {"X-Forwarded-For": "203.0.113.9", "X-Forwarded-User": "alice"},
        )
    )
    assert response.status_code == 200, response.text
    assert response.json()["user"] == "alice"


def test_a_forwarded_address_does_not_make_a_stranger_the_proxy(
    server_app: Any,
) -> None:
    """The proxy is the peer that connected, never what a header claims."""
    mode = ServerMode.from_options(enabled=True, trusted_proxies=["10.0.0.5"])
    response = asyncio.run(
        _get(
            _served_app(mode),
            {"X-Forwarded-For": "10.0.0.5", "X-Forwarded-User": "alice"},
        )
    )
    assert response.status_code == 403
