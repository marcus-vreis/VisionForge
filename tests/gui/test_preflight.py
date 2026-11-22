"""``visionforge gui --check``: each check passes and fails, and nothing starts."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

import visionforge.__main__ as cli
import visionforge.gui.server as server
from visionforge.gui.preflight import run_gui_check


@pytest.fixture
def spa(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A SPA build in a temp folder that the server module points at."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "index-abc123.js").write_text("//", encoding="utf-8")
    (tmp_path / "index.html").write_text(
        '<script src="/assets/index-abc123.js"></script>', encoding="utf-8"
    )
    monkeypatch.setattr(server, "STATIC_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
    return port


def test_every_check_passes(
    spa: Path, free_port: int, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_gui_check("127.0.0.1", free_port) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert all(line.startswith("[OK] ") for line in lines)
    assert "index-abc123.js" in lines[1]


def test_a_missing_build_fails(
    spa: Path, free_port: int, capsys: pytest.CaptureFixture[str]
) -> None:
    (spa / "index.html").unlink()
    assert run_gui_check("127.0.0.1", free_port) == 1
    out = capsys.readouterr().out
    assert "[FAIL] no SPA build" in out
    assert "[OK] port" in out  # the other checks still report


def test_an_index_without_a_bundle_fails(
    spa: Path, free_port: int, capsys: pytest.CaptureFixture[str]
) -> None:
    (spa / "index.html").write_text("<html></html>", encoding="utf-8")
    assert run_gui_check("127.0.0.1", free_port) == 1
    assert "does not reference an index-*.js bundle" in capsys.readouterr().out


def test_a_bundle_missing_on_disk_fails(
    spa: Path, free_port: int, capsys: pytest.CaptureFixture[str]
) -> None:
    (spa / "assets" / "index-abc123.js").unlink()
    assert run_gui_check("127.0.0.1", free_port) == 1
    assert "assets/index-abc123.js, which is missing" in capsys.readouterr().out


def test_a_wrong_health_version_fails(
    spa: Path,
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def health() -> dict[str, str]:
        return {"version": "0.0.0-stale", "spa_bundle": ""}

    monkeypatch.setattr(server, "health", health)
    assert run_gui_check("127.0.0.1", free_port) == 1
    assert "[FAIL] /api/health reports version '0.0.0-stale'" in capsys.readouterr().out


def test_a_raising_health_fails(
    spa: Path,
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def health() -> dict[str, str]:
        raise RuntimeError("boom")

    monkeypatch.setattr(server, "health", health)
    assert run_gui_check("127.0.0.1", free_port) == 1
    assert "[FAIL] /api/health failed: RuntimeError: boom" in capsys.readouterr().out


def test_a_busy_port_fails(spa: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert run_gui_check("127.0.0.1", port) == 1
    out = capsys.readouterr().out
    assert f"[FAIL] port {port} on 127.0.0.1 is already in use" in out


def test_a_wildcard_host_checks_loopback(
    spa: Path, free_port: int, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_gui_check("0.0.0.0", free_port) == 0  # noqa: S104
    assert f"port {free_port} on 0.0.0.0 is free" in capsys.readouterr().out


def test_the_flag_exits_with_the_result_and_never_starts_the_server(
    spa: Path,
    free_port: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import webbrowser

    import visionforge.utils.logger as vf_logger

    started: list[object] = []
    opened: list[object] = []
    monkeypatch.setattr(cli, "_use_utf8_console", lambda: None)
    monkeypatch.setattr(vf_logger, "setup_logger", lambda: None)
    monkeypatch.setattr(server, "start_server", lambda **kw: started.append(kw))
    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: opened.append(a))
    monkeypatch.setattr(
        sys, "argv", ["visionforge", "gui", "--check", "--port", str(free_port)]
    )

    with pytest.raises(SystemExit) as ok:
        cli.main()
    assert ok.value.code == 0

    (spa / "index.html").unlink()
    with pytest.raises(SystemExit) as bad:
        cli.main()
    assert bad.value.code == 1
    assert started == []
    assert opened == []
    assert "[FAIL]" in capsys.readouterr().out
