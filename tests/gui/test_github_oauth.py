"""Connecting a profile to GitHub through the operator's OAuth App (ADR-250).

The real app answers; GitHub is replaced at the HTTP layer (``urlopen``), so no
request leaves the machine. Both a local install and server mode are covered.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from email.message import Message
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from loguru import logger

from visionforge.gui.api import github_oauth, profile_tokens, routes
from visionforge.gui.server_mode import ServerMode

CLIENT_ID = "Iv1.client-id-for-tests"
CLIENT_SECRET = "client-secret-never-logged-0123456789"
CODE = "github-code-never-logged-42"
TOKEN = "gho_token-never-logged-abcdef"
PROXY = "10.0.0.5"
ANA = {"X-Forwarded-User": "ana@lab.org"}
BIA = {"X-Forwarded-User": "bia@lab.org"}


class FakeGitHub:
    """``urlopen`` that answers as GitHub's token endpoint and records the calls."""

    def __init__(self) -> None:
        self.calls: list[urllib.request.Request] = []
        self.answer: Any = {
            "access_token": TOKEN,
            "token_type": "bearer",
            "scope": "repo",
        }
        self.raise_error: Exception | None = None

    def __call__(self, request: urllib.request.Request, timeout: float = 0) -> Any:
        self.calls.append(request)
        if self.raise_error is not None:
            raise self.raise_error
        return io.BytesIO(json.dumps(self.answer).encode())

    def form(self, index: int = -1) -> dict[str, str]:
        data = self.calls[index].data
        assert isinstance(data, bytes)
        return dict(urllib.parse.parse_qsl(data.decode()))


@pytest.fixture
def github(monkeypatch: pytest.MonkeyPatch) -> FakeGitHub:
    fake = FakeGitHub()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


@pytest.fixture
def logs() -> Iterator[list[str]]:
    """Everything loguru writes while the test runs, at every level."""
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), level="TRACE", format="{message}")
    yield lines
    logger.remove(sink)


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(github_oauth.CLIENT_ID_ENV, CLIENT_ID)
    monkeypatch.setenv(github_oauth.CLIENT_SECRET_ENV, CLIENT_SECRET)
    monkeypatch.delenv(github_oauth.PUBLIC_URL_ENV, raising=False)


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("VISIONFORGE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(profile_tokens, "config_dir", lambda: tmp_path / "config")
    monkeypatch.delenv(github_oauth.CLIENT_ID_ENV, raising=False)
    monkeypatch.delenv(github_oauth.CLIENT_SECRET_ENV, raising=False)
    github_oauth.STATES.clear()
    yield
    github_oauth.STATES.clear()


@pytest.fixture
def local() -> TestClient:
    from visionforge.gui.server import app

    return TestClient(app, base_url="http://localhost", follow_redirects=False)


@pytest.fixture
def server() -> Iterator[TestClient]:
    from visionforge.gui.server import app

    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    app.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY]
    )
    yield TestClient(
        app, base_url="http://localhost", client=(PROXY, 50000), follow_redirects=False
    )
    if had:
        app.state.server_mode = saved
    else:
        del app.state.server_mode


def _query(url: str) -> dict[str, str]:
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))


def _start(client: TestClient, headers: dict[str, str] | None = None) -> dict[str, str]:
    response = client.post("/api/github/oauth/start", headers=headers or {})
    assert response.status_code == 200, response.text
    url = response.json()["url"]
    assert url.startswith(github_oauth.AUTHORIZE_URL + "?")
    return _query(url)


def _callback(
    client: TestClient, params: dict[str, str], headers: dict[str, str] | None = None
) -> Any:
    return client.get(
        "/api/github/oauth/callback", params=params, headers=headers or {}
    )


def _token_of(header: str | None) -> str | None:
    return profile_tokens.read_token("github", routes.current_profile(header))


def _server_token(client: TestClient, who: dict[str, str]) -> str | None:
    slug = client.get("/api/session", headers=who).json()["profile"]
    return _token_of(slug)


# --- Off without the environment -------------------------------------------------


def test_without_the_environment_the_feature_is_absent(
    local: TestClient, github: FakeGitHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert local.get("/api/session").json()["github_oauth"] is False
    assert local.post("/api/github/oauth/start").status_code == 404
    gone = _callback(local, {"state": "x", "code": CODE})
    assert gone.status_code == 404
    assert "token do GitHub" in gone.json()["detail"]
    # Half a configuration is no configuration.
    monkeypatch.setenv(github_oauth.CLIENT_ID_ENV, CLIENT_ID)
    assert local.post("/api/github/oauth/start").status_code == 404
    assert github.calls == []


def test_without_the_environment_a_server_hides_it_too(
    server: TestClient,
) -> None:
    assert server.get("/api/session", headers=ANA).json()["github_oauth"] is False
    assert server.post("/api/github/oauth/start", headers=ANA).status_code == 404


def test_an_unusable_public_url_turns_the_feature_off(
    local: TestClient, configured: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(github_oauth.PUBLIC_URL_ENV, "lab.org/visionforge")
    assert local.get("/api/session").json()["github_oauth"] is False
    monkeypatch.setenv(github_oauth.PUBLIC_URL_ENV, "https://lab.org/?x=1")
    assert local.post("/api/github/oauth/start").status_code == 404


# --- The start ------------------------------------------------------------------


def test_the_start_asks_github_for_the_repo_scope_with_pkce(
    local: TestClient, configured: None
) -> None:
    assert local.get("/api/session").json()["github_oauth"] is True
    query = _start(local)
    assert query["client_id"] == CLIENT_ID
    assert query["scope"] == "repo"
    assert query["code_challenge_method"] == "S256"
    assert len(query["state"]) >= 40
    assert len(query["code_challenge"]) == 43
    # No public address: GitHub uses the callback registered in the app.
    assert "redirect_uri" not in query
    assert CLIENT_SECRET not in json.dumps(query)
    assert _start(local)["state"] != query["state"]


def test_the_public_url_builds_the_callback(
    local: TestClient,
    configured: None,
    github: FakeGitHub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(github_oauth.PUBLIC_URL_ENV, "https://vf.lab.org/")
    query = _start(local)
    assert query["redirect_uri"] == "https://vf.lab.org/api/github/oauth/callback"
    done = _callback(local, {"state": query["state"], "code": CODE})
    assert done.status_code == 303
    assert done.headers["location"] == "https://vf.lab.org/?github_oauth=connected"
    assert github.form()["redirect_uri"] == query["redirect_uri"]


def test_a_start_from_another_site_is_refused(
    local: TestClient, configured: None
) -> None:
    response = local.post(
        "/api/github/oauth/start", headers={"Origin": "https://evil.example"}
    )
    assert response.status_code == 403


# --- The callback on a local install --------------------------------------------


def test_the_callback_saves_the_token_for_the_profile_that_started(
    local: TestClient, configured: None, github: FakeGitHub
) -> None:
    assert local.post("/api/profiles", json={"name": "Ana"}).status_code == 201
    query = _start(local, {"X-VF-Profile": "ana"})
    # GitHub sends the browser back: a navigation, without the profile header.
    done = _callback(local, {"state": query["state"], "code": CODE})

    assert done.status_code == 303
    assert done.headers["location"] == "/?github_oauth=connected"
    assert done.headers["cache-control"] == "no-store"
    assert _token_of("ana") == TOKEN
    assert _token_of(None) is None
    assert local.get("/api/github/token", headers={"X-VF-Profile": "ana"}).json()[
        "configured"
    ]

    (call,) = github.calls
    assert call.full_url == github_oauth.ACCESS_TOKEN_URL
    assert call.get_method() == "POST"
    assert call.get_header("Accept") == "application/json"
    form = github.form()
    assert form["client_id"] == CLIENT_ID
    assert form["client_secret"] == CLIENT_SECRET
    assert form["code"] == CODE
    verifier = form["code_verifier"]
    digest = hashlib.sha256(verifier.encode()).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    assert expected == query["code_challenge"]


@pytest.mark.parametrize(
    "params",
    [{}, {"code": CODE}, {"state": "made-up", "code": CODE}],
    ids=["nothing", "no-state", "unknown-state"],
)
def test_a_callback_without_a_state_this_server_issued_is_refused(
    local: TestClient, configured: None, github: FakeGitHub, params: dict[str, str]
) -> None:
    _start(local)
    response = _callback(local, params)
    assert response.status_code == 400
    assert "Conectar com GitHub" in response.json()["detail"]
    assert github.calls == []
    assert _token_of(None) is None


def test_a_state_is_spent_on_first_use(
    local: TestClient, configured: None, github: FakeGitHub
) -> None:
    state = _start(local)["state"]
    assert _callback(local, {"state": state, "code": CODE}).status_code == 303
    profile_tokens.token_file("github", routes.current_profile(None)).unlink()
    again = _callback(local, {"state": state, "code": CODE})
    assert again.status_code == 400
    assert len(github.calls) == 1
    assert _token_of(None) is None


def test_a_state_expires_after_ten_minutes(
    local: TestClient,
    configured: None,
    github: FakeGitHub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"t": 1000.0}
    monkeypatch.setattr(
        github_oauth, "STATES", github_oauth.OAuthStates(lambda: now["t"])
    )
    late = _start(local)["state"]
    on_time = _start(local)["state"]
    now["t"] += github_oauth.STATE_TTL_S - 1
    assert _callback(local, {"state": on_time, "code": CODE}).status_code == 303
    now["t"] += 1
    assert _callback(local, {"state": late, "code": CODE}).status_code == 400
    assert len(github.calls) == 1


def test_a_profile_keeps_only_its_latest_requests_open(
    local: TestClient, configured: None, github: FakeGitHub
) -> None:
    states = [_start(local)["state"] for _ in range(github_oauth.MAX_PENDING + 1)]
    assert _callback(local, {"state": states[0], "code": CODE}).status_code == 400
    assert _callback(local, {"state": states[-1], "code": CODE}).status_code == 303


@pytest.mark.parametrize(
    "answer",
    [
        {"error": "bad_verification_code", "error_description": "The code is wrong."},
        {"access_token": "has a space"},
        {"access_token": ""},
        ["not", "an", "object"],
    ],
    ids=["github-error", "spaces", "empty", "not-an-object"],
)
def test_no_usable_token_goes_back_with_a_failure(
    local: TestClient, configured: None, github: FakeGitHub, answer: Any
) -> None:
    github.answer = answer
    done = _callback(local, {"state": _start(local)["state"], "code": CODE})
    assert done.status_code == 303
    assert done.headers["location"] == "/?github_oauth=failed"
    assert _token_of(None) is None


def test_github_unreachable_goes_back_with_a_failure(
    local: TestClient, configured: None, github: FakeGitHub
) -> None:
    github.raise_error = urllib.error.URLError("no route")
    done = _callback(local, {"state": _start(local)["state"], "code": CODE})
    assert done.headers["location"] == "/?github_oauth=failed"
    assert _token_of(None) is None


def test_a_refusal_on_github_exchanges_nothing(
    local: TestClient, configured: None, github: FakeGitHub
) -> None:
    state = _start(local)["state"]
    done = _callback(local, {"state": state, "error": "access_denied"})
    assert done.headers["location"] == "/?github_oauth=failed"
    assert github.calls == []
    # The refusal spent the request.
    assert _callback(local, {"state": state, "code": CODE}).status_code == 400


def test_secrets_never_reach_the_logs_or_the_address(
    local: TestClient, configured: None, github: FakeGitHub, logs: list[str]
) -> None:
    query = _start(local)
    done = _callback(local, {"state": query["state"], "code": CODE})
    verifier = github.form()["code_verifier"]
    github.answer = {
        "error": "bad_verification_code",
        "error_description": CLIENT_SECRET,
    }
    failed = _callback(local, {"state": _start(local)["state"], "code": CODE})
    github.raise_error = urllib.error.HTTPError(
        github_oauth.ACCESS_TOKEN_URL, 500, TOKEN, Message(), None
    )
    _callback(local, {"state": _start(local)["state"], "code": CODE})

    text = "\n".join(logs)
    assert "saved" in text and "bad_verification_code" in text and "HTTPError" in text
    for secret in (CLIENT_SECRET, CODE, TOKEN, query["state"], verifier):
        assert secret not in text
        assert secret not in done.headers["location"]
        assert secret not in failed.headers["location"]
    assert repr(github_oauth.oauth_app()).count(CLIENT_SECRET) == 0


# --- Server mode ------------------------------------------------------------------


def test_on_a_server_the_token_is_saved_for_the_signed_in_person_only(
    server: TestClient, configured: None, github: FakeGitHub
) -> None:
    assert server.get("/api/session", headers=ANA).json()["github_oauth"] is True
    query = _start(server, ANA)
    done = _callback(server, {"state": query["state"], "code": CODE}, ANA)
    assert done.status_code == 303, done.text
    assert done.headers["location"] == "/?github_oauth=connected"
    assert _server_token(server, ANA) == TOKEN
    assert _server_token(server, BIA) is None
    assert server.get("/api/github/token", headers=ANA).json()["configured"]
    assert not server.get("/api/github/token", headers=BIA).json()["configured"]
    # Disconnecting is the token's own DELETE.
    assert server.delete("/api/github/token", headers=ANA).status_code == 200
    assert _server_token(server, ANA) is None


def test_on_a_server_another_persons_state_is_refused(
    server: TestClient, configured: None, github: FakeGitHub
) -> None:
    anas = _start(server, ANA)["state"]
    stolen = _callback(server, {"state": anas, "code": CODE}, BIA)
    assert stolen.status_code == 400
    assert github.calls == []
    assert _server_token(server, ANA) is None
    assert _server_token(server, BIA) is None
    # The attempt spent it: ana starts again.
    assert _callback(server, {"state": anas, "code": CODE}, ANA).status_code == 400


def test_on_a_server_the_callback_needs_the_login(
    server: TestClient, configured: None, github: FakeGitHub
) -> None:
    state = _start(server, ANA)["state"]
    assert _callback(server, {"state": state, "code": CODE}).status_code == 401
    assert github.calls == []
