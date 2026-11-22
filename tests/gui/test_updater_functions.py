"""Edge paths of the stand-alone updater (ADR-120): waits, install, relaunch, cleanup.

Complements ``test_update.py``, which covers the happy paths and the retry rules.
Nothing here installs anything or starts a GUI: subprocess calls are faked or run
``sys.executable`` on a one-liner.
"""

from __future__ import annotations

import ctypes
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from visionforge.gui import updater


def _args(tmp_path: Path, **over: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 8000,
        "pids": [],
        "target": "0.15.0",
        "install": ["new"],
        "restore": None,
        "relaunch": ["gui"],
        "cwd": str(tmp_path),
        "log": str(tmp_path / "logs" / "update.log"),
        "status": str(tmp_path / "out" / "status.json"),
    }
    args.update(over)
    return args


def _status(tmp_path: Path) -> dict[str, Any]:
    status: dict[str, Any] = json.loads(
        (tmp_path / "out" / "status.json").read_text(encoding="utf-8")
    )
    return status


# ── port ──────────────────────────────────────────────────────────────────────


def test_port_in_use_sees_a_listener_and_a_free_port() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)  # probes are never accepted: keep the backlog from filling
        port = listener.getsockname()[1]
        assert updater.port_in_use("127.0.0.1", port)
        # the old server never lets go: the wait gives up instead of hanging
        assert not updater.wait_for_port_free("127.0.0.1", port, timeout=0.2, poll=0.01)
    assert not updater.port_in_use("127.0.0.1", port)


# ── processes ─────────────────────────────────────────────────────────────────


def test_process_alive_posix_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    outcomes: dict[int, BaseException | None] = {
        1: None,
        2: ProcessLookupError(),
        3: PermissionError(),
    }

    def fake_kill(pid: int, sig: int) -> None:
        assert sig == 0  # a probe, never a real signal
        exc = outcomes[pid]
        if exc:
            raise exc

    monkeypatch.setattr(updater.os, "kill", fake_kill)
    assert updater.process_alive(1) is True
    assert updater.process_alive(2) is False
    # exists but belongs to someone else: still alive
    assert updater.process_alive(3) is True


class _FakeKernel32:
    def __init__(self, handle: int, wait_result: int) -> None:
        self.handle = handle
        self.wait_result = wait_result
        self.closed: list[int] = []

    def OpenProcess(self, access: int, inherit: bool, pid: int) -> int:  # noqa: N802
        return self.handle

    def WaitForSingleObject(self, handle: int, ms: int) -> int:  # noqa: N802
        return self.wait_result

    def CloseHandle(self, handle: int) -> None:  # noqa: N802
        self.closed.append(handle)


@pytest.mark.parametrize(
    ("handle", "wait_result", "alive", "closed"),
    [
        (0, 0, False, []),  # no such process: no handle to close
        (7, 0x00000102, True, [7]),  # WAIT_TIMEOUT: still running
        (7, 0, False, [7]),  # signalled: ended
    ],
)
def test_process_alive_windows_branch(
    monkeypatch: pytest.MonkeyPatch,
    handle: int,
    wait_result: int,
    alive: bool,
    closed: list[int],
) -> None:
    kernel = _FakeKernel32(handle, wait_result)

    class _Windll:
        kernel32 = kernel

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", _Windll, raising=False)
    assert updater.process_alive(123) is alive
    assert kernel.closed == closed


def test_wait_for_processes_gone_times_out_while_one_lives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(updater, "process_alive", lambda pid: pid == 2)
    assert not updater.wait_for_processes_gone([1, 2], timeout=0.1, poll=0.01)
    assert updater.wait_for_processes_gone([1, 3], timeout=0.1, poll=0.01)
    assert updater.wait_for_processes_gone([], timeout=0.1, poll=0.01)


# ── install ───────────────────────────────────────────────────────────────────


def test_run_install_appends_output_with_a_header(tmp_path: Path) -> None:
    log = tmp_path / "deep" / "update.log"
    cmd = [sys.executable, "-c", "print('hello from install'); raise SystemExit(3)"]
    assert updater.run_install(cmd, log) == 3
    assert updater.run_install(cmd, log) == 3  # appends, never truncates
    text = log.read_text(encoding="utf-8")
    assert text.splitlines().count("hello from install") == 2
    assert text.count("=== ") == 2
    assert "-c" in text.splitlines()[1]


def test_run_install_that_cannot_start_returns_minus_one(tmp_path: Path) -> None:
    log = tmp_path / "update.log"
    code = updater.run_install([str(tmp_path / "no-such-binary")], log)
    assert code == -1
    assert "install did not run:" in log.read_text(encoding="utf-8")


def test_run_install_timeout_returns_minus_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(updater.subprocess, "run", fake_run)
    log = tmp_path / "update.log"
    assert updater.run_install(["x"], log) == -1
    assert "install did not run:" in log.read_text(encoding="utf-8")


def test_run_install_has_no_creationflags_off_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(updater.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "platform", "linux")
    assert updater.run_install(["x"], tmp_path / "update.log") == 0
    assert seen["creationflags"] == 0
    assert seen["check"] is False


def test_a_lock_that_never_clears_stops_after_the_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    naps: list[float] = []

    def locked(command: list[str], log: Path) -> int:
        calls.append(1)
        with log.open("a", encoding="utf-8") as fh:
            fh.write("PermissionError: [WinError 32] in use\n")
        return 1

    monkeypatch.setattr(updater, "run_install", locked)
    monkeypatch.setattr(updater.time, "sleep", naps.append)
    code = updater.install_with_retries(["x"], tmp_path / "update.log", 3, 7.0)
    assert code == 1
    assert len(calls) == 4  # first try + 3 retries
    assert naps == [7.0, 7.0, 7.0]


def test_tail_of_a_missing_log_is_empty_and_of_a_long_one_is_the_end(
    tmp_path: Path,
) -> None:
    assert updater._tail(tmp_path / "missing.log") == ""
    log = tmp_path / "update.log"
    log.write_text("a" * 5000 + "END", encoding="utf-8")
    tail = updater._tail(log, size=10)
    assert tail == "a" * 7 + "END"


# ── relaunch ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_relaunch_detaches_and_silences_the_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    seen: dict[str, Any] = {}

    def fake_popen(command: list[str], **kwargs: Any) -> None:
        seen["command"] = command
        seen.update(kwargs)

    monkeypatch.setattr(updater.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(updater.subprocess, "DETACHED_PROCESS", 0x8, raising=False)
    monkeypatch.setattr(
        updater.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False
    )
    updater.relaunch(["visionforge", "gui"], str(tmp_path))

    assert seen["command"] == ["visionforge", "gui"]
    assert seen["cwd"] == str(tmp_path)
    assert seen["env"]["VISIONFORGE_NO_BROWSER"] == "1"
    assert seen["stdin"] is seen["stdout"] is seen["stderr"] is subprocess.DEVNULL
    assert seen["close_fds"] is True
    if platform == "win32":
        assert seen["creationflags"] == 0x8 | 0x200
        assert "start_new_session" not in seen
    else:
        assert seen["start_new_session"] is True
        assert "creationflags" not in seen


# ── main ──────────────────────────────────────────────────────────────────────


def test_main_gives_up_when_the_server_processes_never_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran: list[str] = []
    waited: list[list[int]] = []

    def stuck(pids: list[int]) -> bool:
        waited.append(pids)
        return False

    monkeypatch.setattr(updater, "wait_for_processes_gone", stuck)

    def port_free(host: str, port: int) -> bool:
        ran.append("port")
        return True

    def install(command: list[str], log: Path) -> int:
        ran.append("install")
        return 0

    monkeypatch.setattr(updater, "wait_for_port_free", port_free)
    monkeypatch.setattr(updater, "install_with_retries", install)
    monkeypatch.setattr(updater, "relaunch", lambda c, cwd: ran.append("relaunch"))

    code = updater.main(_args(tmp_path, pids=["11", 22], restore=["old"]))

    assert code == -2
    assert waited == [[11, 22]]  # pids from JSON may be strings
    assert ran == []  # not even the port wait, let alone an install over a live server
    status = _status(tmp_path)
    assert status["ok"] is False
    assert status["port_freed"] is False
    assert status["restored"] is None
    assert status["target"] == "0.15.0"


def test_main_failed_install_without_restore_still_relaunches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relaunched: list[tuple[list[str], str]] = []
    monkeypatch.setattr(updater, "wait_for_port_free", lambda h, p: True)
    monkeypatch.setattr(updater, "install_with_retries", lambda c, log: 5)
    monkeypatch.setattr(updater, "relaunch", lambda c, cwd: relaunched.append((c, cwd)))

    assert updater.main(_args(tmp_path)) == 5

    status = _status(tmp_path)
    assert (status["ok"], status["returncode"], status["restored"]) == (False, 5, None)
    assert status["log"] == str(tmp_path / "logs" / "update.log")
    assert relaunched == [(["gui"], str(tmp_path))]


def test_main_records_a_restore_that_failed_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    codes = {"new": 1, "old": 2}
    monkeypatch.setattr(updater, "wait_for_port_free", lambda h, p: True)
    monkeypatch.setattr(updater, "install_with_retries", lambda c, log: codes[c[0]])
    monkeypatch.setattr(updater, "relaunch", lambda c, cwd: None)

    assert updater.main(_args(tmp_path, restore=["old"])) == 1
    assert _status(tmp_path)["restored"] is False


# ── own-copy cleanup ──────────────────────────────────────────────────────────


def test_remove_own_copy_deletes_only_the_temporary_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temp = tmp_path / "vf-update-abc"
    temp.mkdir()
    (temp / "updater.py").write_text("x", encoding="utf-8")
    monkeypatch.setattr(updater, "__file__", str(temp / "updater.py"))
    updater._remove_own_copy()
    assert not temp.exists()


def test_remove_own_copy_leaves_the_installed_package_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = tmp_path / "gui"
    pkg.mkdir()
    (pkg / "updater.py").write_text("x", encoding="utf-8")
    monkeypatch.setattr(updater, "__file__", str(pkg / "updater.py"))
    updater._remove_own_copy()
    assert (pkg / "updater.py").is_file()
