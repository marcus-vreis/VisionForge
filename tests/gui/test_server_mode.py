"""Server mode: only the login proxy is answered, and it says who is asking.

With ``--server`` the user comes from the proxy's header and nothing the
browser sends can change it; a route that is not ready for several people is
404 (ADR-212, ADR-213). With the mode off nothing changes.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from visionforge.gui.api import routes
from visionforge.gui.api.profiles import is_valid_slug
from visionforge.gui.server_mode import (
    OWNER_READY,
    ROUTE_CLASSES,
    ServerMode,
    ServerModeError,
    answers,
    normalize_user,
    route_key,
    user_slug,
)

PROXY = "10.0.0.5"
USER = {"X-Forwarded-User": "ana@lab.org"}


def test_off_by_default_and_the_proxy_options_need_the_flag() -> None:
    assert ServerMode.from_options(enabled=False) == ServerMode()
    with pytest.raises(ServerModeError, match="need --server"):
        ServerMode.from_options(enabled=False, trusted_proxies=["10.0.0.5"])
    with pytest.raises(ServerModeError, match="need --server"):
        ServerMode.from_options(enabled=False, user_header="Remote-User")


def test_server_mode_refuses_to_start_without_a_trusted_proxy() -> None:
    with pytest.raises(ServerModeError, match="--trusted-proxy"):
        ServerMode.from_options(enabled=True)
    with pytest.raises(ServerModeError, match="not an IP address"):
        ServerMode.from_options(enabled=True, trusted_proxies=["proxy.lab"])


def test_the_user_header_cannot_be_the_profile_header() -> None:
    with pytest.raises(ServerModeError, match="--user-header"):
        ServerMode.from_options(
            enabled=True, trusted_proxies=[PROXY], user_header="x-vf-profile"
        )


def test_only_the_proxy_addresses_are_trusted() -> None:
    mode = ServerMode.from_options(
        enabled=True, trusted_proxies=["10.0.0.5", "172.18.0.0/16", "::1"]
    )
    assert mode.trusts("10.0.0.5")
    assert mode.trusts("172.18.3.9")
    assert mode.trusts("::1")
    assert mode.trusts("::ffff:10.0.0.5")
    assert not mode.trusts("10.0.0.6")
    assert not mode.trusts("127.0.0.1")
    assert not mode.trusts("testclient")
    assert not mode.trusts(None)


def test_the_account_name_has_one_spelling() -> None:
    assert normalize_user("  Ana@Lab.org ") == "ana@lab.org"
    assert normalize_user(None) is None
    assert normalize_user("   ") is None
    assert normalize_user("ana lab") is None
    assert normalize_user("ana\x00") is None
    assert normalize_user("a" * 255) is None


def test_two_accounts_never_share_a_folder() -> None:
    dotted, dashed = user_slug("a.b@lab.org"), user_slug("a-b@lab.org")
    assert dotted != dashed
    assert dotted.startswith("a-b-lab-org-")
    assert user_slug("a.b@lab.org") == dotted
    long = user_slug("x" * 200 + "@lab.org")
    for slug in (dotted, dashed, long, user_slug("日本")):
        assert is_valid_slug(slug)


def test_every_route_of_the_app_has_a_class() -> None:
    from visionforge.gui.server import app

    # The OpenAPI document lists every route however the routers are split;
    # ``app.routes`` hides an included router's children on newer FastAPI.
    served = {
        f"{method.upper()} {path}"
        for path, operations in app.openapi()["paths"].items()
        for method in operations
    }
    # OpenAPI drops the ``:path`` converter.
    classed = {key.replace(":path}", "}") for key in ROUTE_CLASSES}
    assert served - classed == set(), "routes with no server-mode class"
    assert classed - served == set(), "classes of routes that are gone"


def test_a_request_finds_its_route_in_the_table() -> None:
    assert route_key("GET", "/api/runs") == "GET /api/runs"
    assert route_key("HEAD", "/api/health") == "GET /api/health"
    assert route_key("DELETE", "/api/runs/20261118_1") == "DELETE /api/runs/{run_id}"
    assert route_key("GET", "/api/detection/schema") == "GET /api/detection/schema"
    assert route_key("GET", "/api/custom/mine/schema") == "GET /api/custom/{key}/schema"
    assert (
        route_key("GET", "/api/artifacts/outputs/a/b.png")
        == "GET /api/artifacts/{file_path:path}"
    )
    assert route_key("GET", "/assets/index.js") == "GET /{path:path}"
    assert route_key("GET", "/") == "GET /{path:path}"
    assert route_key("POST", "/api/explorer/scan") == "POST /api/explorer/scan"
    assert route_key("OPTIONS", "/api/runs") is None
    assert route_key("POST", "/api/nothing") is None
    assert route_key("GET", "/api/nothing") is None
    assert route_key("GET", "/openapi.json") is None


def test_only_owner_routes_are_marked_as_read() -> None:
    assert {ROUTE_CLASSES[key] for key in OWNER_READY} == {"owner"}


def test_a_route_the_table_does_not_know_is_refused() -> None:
    assert not answers("GET /api/brand-new")
    assert answers("GET /api/health")
    assert answers("GET /api/runs")
    assert not answers("GET /api/queue")
    assert not answers("POST /api/dataset/stats")
    assert not answers("POST /api/dataset/pick")


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """The real app in server mode, its profiles under ``tmp_path``."""
    from visionforge.gui.server import app

    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    app.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY]
    )
    yield app
    if had:
        app.state.server_mode = saved
    else:
        del app.state.server_mode


def _client(app: Any, host: str = PROXY) -> TestClient:
    return TestClient(app, base_url="http://localhost", client=(host, 50000))


def test_a_local_install_reports_no_session() -> None:
    from visionforge.gui.server import app

    body = TestClient(app, base_url="http://localhost").get("/api/session").json()
    assert body == {"server_mode": False, "user": None, "profile": None}


def test_a_client_that_is_not_the_proxy_is_refused(served: Any) -> None:
    response = _client(served, "10.0.0.6").get("/api/session", headers=USER)
    assert response.status_code == 403
    assert "proxy" in response.json()["detail"]


def test_a_request_without_a_user_is_refused(served: Any) -> None:
    response = _client(served).get("/api/session")
    assert response.status_code == 401
    assert "X-Forwarded-User" in response.json()["detail"]
    assert (
        _client(served).get("/", headers={"X-Forwarded-User": " "}).status_code == 401
    )


def test_the_proxy_health_probe_needs_no_user(served: Any) -> None:
    assert _client(served).get("/api/health").status_code == 200


def test_the_user_the_proxy_names_gets_their_own_profile(
    served: Any, tmp_path: Path
) -> None:
    body = _client(served).get("/api/session", headers=USER).json()
    slug = user_slug("ana@lab.org")
    assert body == {"server_mode": True, "user": "ana@lab.org", "profile": slug}
    assert (tmp_path / "profiles" / slug / "models").is_dir()
    assert _client(served).get("/api/runs", headers=USER).json() == []


def test_the_browser_cannot_pick_another_profile(served: Any, tmp_path: Path) -> None:
    client = _client(served)
    client.get("/api/session", headers={"X-Forwarded-User": "bia@lab.org"})
    theirs = user_slug("bia@lab.org")
    run = tmp_path / "profiles" / theirs / "models" / "exp" / "20261118_120000_000001"
    run.mkdir(parents=True)

    forged = {**USER, "X-VF-Profile": theirs}
    assert client.get("/api/session", headers=forged).json()["profile"] == user_slug(
        "ana@lab.org"
    )
    assert client.get(f"/api/runs/{run.name}", headers=forged).status_code == 404
    assert client.delete(f"/api/runs/{run.name}", headers=forged).status_code == 404
    assert run.is_dir()


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/dataset/pick"),
        ("POST", "/api/update/apply"),
        ("POST", "/api/profiles"),
        ("GET", "/api/queue"),
        ("GET", "/api/experiment/events"),
        ("GET", "/api/artifacts/outputs/models/x.png"),
        ("POST", "/api/dataset/stats"),
        ("POST", "/api/experiment/run"),
        ("GET", "/api/colab/status"),
        ("GET", "/api/brand-new"),
        ("OPTIONS", "/api/runs"),
        ("GET", "/openapi.json"),
    ],
)
def test_a_route_not_ready_for_several_people_is_404(
    served: Any, method: str, path: str
) -> None:
    response = _client(served).request(method, path, headers=USER)
    assert response.status_code == 404
    assert "servidor" in response.json()["detail"]


def _run_cli(argv: list[str]) -> dict[str, Any]:
    from visionforge.__main__ import main

    seen: dict[str, Any] = {}
    with (
        patch.object(sys, "argv", ["visionforge", *argv]),
        patch("visionforge.gui.server.start_server", lambda **kw: seen.update(kw)),
    ):
        main()
    return seen


def test_the_cli_refuses_server_mode_without_a_proxy(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as stop:
        _run_cli(["gui", "--server"])
    assert stop.value.code == 2
    assert "--trusted-proxy" in capsys.readouterr().err


def test_the_cli_hands_the_server_options_to_the_server() -> None:
    seen = _run_cli(
        [
            "gui",
            "--server",
            "--trusted-proxy",
            "10.0.0.5",
            "--user-header",
            "Remote-User",
        ]
    )
    mode = seen["server_mode"]
    assert (
        mode.enabled and mode.trusts("10.0.0.5") and mode.user_header == "Remote-User"
    )
    assert not _run_cli(["gui"])["server_mode"].enabled
