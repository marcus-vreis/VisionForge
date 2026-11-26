"""The per-profile tokens encrypted at rest with the operator's key (ADR-253)."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from loguru import logger

from visionforge.gui.api import profile_tokens, routes
from visionforge.gui.api.job_process import child_env
from visionforge.gui.api.profile_tokens import read_token, token_file, write_token
from visionforge.gui.api.profiles import Profile
from visionforge.gui.api.token_cipher import (
    KEY_ENV,
    SEALED_PREFIX,
    TokenKeyError,
    token_cipher,
)
from visionforge.gui.server_mode import ServerMode

fernet = pytest.importorskip("cryptography.fernet")

TOKEN = "example-private-token"
PROXY = "10.0.0.5"
ANA = {"X-Forwarded-User": "ana@lab.org"}


@pytest.fixture(autouse=True)
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty config folder and no key, whatever the machine has set."""
    monkeypatch.delenv(KEY_ENV, raising=False)
    monkeypatch.setattr(profile_tokens, "config_dir", lambda: tmp_path / "config")
    return tmp_path / "config"


@pytest.fixture
def ana(tmp_path: Path) -> Profile:
    return Profile(slug="ana", name="Ana", models_dir=tmp_path / "ana")


@pytest.fixture
def bia(tmp_path: Path) -> Profile:
    return Profile(slug="bia", name="Bia", models_dir=tmp_path / "bia")


@pytest.fixture
def warnings() -> Iterator[list[str]]:
    seen: list[str] = []
    sink = logger.add(lambda message: seen.append(str(message)), level="INFO")
    yield seen
    logger.remove(sink)


def _new_key() -> str:
    return str(fernet.Fernet.generate_key().decode())


def _body(profile: Profile, service: str = "hub") -> str:
    # Typed here: the isolated mypy hook sees ``visionforge`` as Any.
    path: Path = token_file(service, profile)  # type: ignore[arg-type]
    return path.read_text(encoding="utf-8")


def test_without_a_key_the_file_is_the_token_as_before(ana: Profile) -> None:
    assert token_cipher() is None
    write_token("hub", ana, TOKEN)
    assert _body(ana) == TOKEN
    assert read_token("hub", ana) == TOKEN


def test_with_a_key_the_file_holds_ciphertext_and_reads_back(
    ana: Profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_ENV, _new_key())
    write_token("hub", ana, TOKEN)
    body = _body(ana)
    assert body.startswith(SEALED_PREFIX)
    assert TOKEN not in body
    assert read_token("hub", ana) == TOKEN


def test_a_plain_token_is_sealed_the_first_time_it_is_read_with_a_key(
    ana: Profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_token("hub", ana, TOKEN)
    monkeypatch.setenv(KEY_ENV, _new_key())
    assert read_token("hub", ana) == TOKEN
    assert _body(ana).startswith(SEALED_PREFIX)
    assert read_token("hub", ana) == TOKEN


def test_a_sealed_token_is_not_read_without_the_key(
    ana: Profile, monkeypatch: pytest.MonkeyPatch, warnings: list[str]
) -> None:
    monkeypatch.setenv(KEY_ENV, _new_key())
    write_token("hub", ana, TOKEN)
    monkeypatch.delenv(KEY_ENV)
    assert read_token("hub", ana) is None
    assert any(KEY_ENV in line and "not set" in line for line in warnings)
    assert not any(TOKEN in line for line in warnings)


def test_another_key_does_not_open_it(
    ana: Profile, monkeypatch: pytest.MonkeyPatch, warnings: list[str]
) -> None:
    monkeypatch.setenv(KEY_ENV, _new_key())
    write_token("hub", ana, TOKEN)
    sealed = _body(ana)
    monkeypatch.setenv(KEY_ENV, _new_key())
    assert read_token("hub", ana) is None
    assert _body(ana) == sealed
    assert any("No key" in line for line in warnings)


def test_rotation_opens_with_the_old_key_and_reseals_with_the_new(
    ana: Profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, new = _new_key(), _new_key()
    monkeypatch.setenv(KEY_ENV, old)
    write_token("hub", ana, TOKEN)
    monkeypatch.setenv(KEY_ENV, f"{new}, {old}")
    assert read_token("hub", ana) == TOKEN
    # The old key can be retired: the new one alone opens the file now.
    monkeypatch.setenv(KEY_ENV, new)
    assert read_token("hub", ana) == TOKEN
    monkeypatch.setenv(KEY_ENV, old)
    assert read_token("hub", ana) is None


def test_a_sealed_file_moved_to_another_profile_or_service_does_not_open(
    ana: Profile, bia: Profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_ENV, _new_key())
    write_token("hub", ana, TOKEN)
    source = token_file("hub", ana)
    for service, profile in (("hub", bia), ("github", ana)):
        target = token_file(service, profile)  # type: ignore[arg-type]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert read_token(service, profile) is None  # type: ignore[arg-type]
    assert read_token("hub", ana) == TOKEN


def test_a_tampered_file_does_not_open(
    ana: Profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_ENV, _new_key())
    write_token("hub", ana, TOKEN)
    token_file("hub", ana).write_text(SEALED_PREFIX + "not-fernet", encoding="utf-8")
    assert read_token("hub", ana) is None


def test_a_value_that_is_not_a_key_is_refused_without_being_echoed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KEY_ENV, f"{_new_key()},not-a-key-at-all")
    with pytest.raises(TokenKeyError) as refused:
        token_cipher()
    assert "not-a-key-at-all" not in str(refused.value)
    assert "of the 2 given" in str(refused.value)


def test_a_key_without_the_package_says_what_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KEY_ENV, _new_key())
    # ``None`` in ``sys.modules`` makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "cryptography.fernet", None)
    with pytest.raises(TokenKeyError, match=r"visionforge-studio\[server\]"):
        token_cipher()


def test_the_cli_refuses_a_bad_key_before_starting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from visionforge.__main__ import main

    monkeypatch.setenv(KEY_ENV, "not-a-key-at-all")
    started: list[Any] = []
    with (
        patch.object(sys, "argv", ["visionforge", "gui", "--no-browser"]),
        patch("visionforge.gui.server.start_server", lambda **kw: started.append(kw)),
        pytest.raises(SystemExit) as stop,
    ):
        main()
    assert stop.value.code == 2
    error = capsys.readouterr().err
    assert KEY_ENV in error
    assert "not-a-key-at-all" not in error
    assert not started


def test_the_child_of_a_job_does_not_receive_the_key(tmp_path: Path) -> None:
    env = child_env({KEY_ENV: _new_key(), "PATH": "/usr/bin"}, tmp_path)
    assert KEY_ENV not in env
    assert env["PATH"] == "/usr/bin"


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


def _start(mode: ServerMode | None) -> None:
    import visionforge.gui.server as server_mod

    with patch("uvicorn.Server"), patch("webbrowser.open"):
        server_mod.start_server(host="127.0.0.1", port=8123, server_mode=mode)


def test_a_server_without_a_key_warns_that_tokens_are_plain_text(
    server_app: Any, warnings: list[str]
) -> None:
    _start(ServerMode.from_options(enabled=True, trusted_proxies=[PROXY]))
    assert any("plain text" in line and KEY_ENV in line for line in warnings)
    assert not any("are encrypted" in line for line in warnings)


def test_a_server_with_a_key_says_tokens_are_encrypted(
    server_app: Any, warnings: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_ENV, _new_key())
    _start(ServerMode.from_options(enabled=True, trusted_proxies=[PROXY]))
    assert any("are encrypted" in line for line in warnings)
    assert not any("plain text" in line for line in warnings)


def test_the_local_tool_without_a_key_says_nothing_about_it(
    server_app: Any, warnings: list[str]
) -> None:
    _start(None)
    assert not any(KEY_ENV in line for line in warnings)


@pytest.fixture
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server_app: Any
) -> TestClient:
    """The real app in server mode, as the login proxy reaches it."""
    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    server_app.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY]
    )
    return TestClient(server_app, base_url="http://localhost", client=(PROXY, 50000))


def test_a_token_saved_through_the_api_is_sealed_on_disk(
    client: TestClient, config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_ENV, _new_key())
    saved = client.put("/api/hub/token", json={"token": TOKEN}, headers=ANA)
    assert saved.json() == {"configured": True, "shared": True}
    (stored,) = [p for p in config.rglob("*") if p.is_file()]
    assert stored.read_text(encoding="utf-8").startswith(SEALED_PREFIX)
    assert TOKEN.encode() not in stored.read_bytes()
    assert client.get("/api/hub/token", headers=ANA).json()["configured"]


def test_the_panel_says_no_token_when_the_key_is_gone(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_ENV, _new_key())
    client.put("/api/hub/token", json={"token": TOKEN}, headers=ANA)
    monkeypatch.delenv(KEY_ENV)
    assert client.get("/api/hub/token", headers=ANA).json() == {
        "configured": False,
        "shared": True,
    }
    # Saving again replaces the file nobody can open.
    again = client.put("/api/hub/token", json={"token": TOKEN}, headers=ANA)
    assert again.json()["configured"]
