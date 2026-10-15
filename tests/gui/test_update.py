"""Updating the install from the page (ADR-120).

No test reaches PyPI, installs anything or stops the server: the lookup, the
installer probe, the spawn and the shutdown are replaced. What is checked is the
decision the page receives (available, can_apply, why not) and the refusals of
``POST /api/update/apply``.
"""

from __future__ import annotations

import io
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from visionforge.gui.api import updates


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("VISIONFORGE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(updates.DISABLE_ENV, raising=False)
    monkeypatch.setattr(updates, "__version__", "0.14.0")
    updates.clear_cache()
    yield
    updates.clear_cache()


def _pypi(version: str) -> Any:
    """An ``urlopen`` that answers like PyPI's JSON API."""

    def opener(url: str, timeout: float) -> Any:
        assert url == updates.PYPI_URL
        return io.BytesIO(json.dumps({"info": {"version": version}}).encode())

    return opener


class TestVersions:
    @pytest.mark.parametrize(
        ("text", "parsed"),
        [
            ("0.14.0", (0, 14, 0)),
            ("1.2.3", (1, 2, 3)),
            ("0.15.0rc1", None),
            ("0.0.0+unknown", None),
            ("", None),
            (None, None),
        ],
    )
    def test_only_plain_releases_parse(self, text: str | None, parsed: Any) -> None:
        assert updates.parse_version(text) == parsed

    def test_newer_compares_numerically(self) -> None:
        assert updates.is_newer("0.15.0", "0.14.0")
        assert updates.is_newer("0.14.10", "0.14.9")
        assert not updates.is_newer("0.14.0", "0.14.0")
        assert not updates.is_newer("0.13.9", "0.14.0")
        assert not updates.is_newer(None, "0.14.0")
        assert not updates.is_newer("0.15.0", "0.0.0+unknown")

    def test_fetch_reads_pypi(self) -> None:
        assert updates.fetch_latest(opener=_pypi("0.15.0")) == "0.15.0"

    def test_fetch_is_silent_offline(self) -> None:
        def offline(url: str, timeout: float) -> Any:
            raise OSError("no network")

        assert updates.fetch_latest(opener=offline) is None

    def test_fetch_ignores_a_body_it_cannot_read(self) -> None:
        def odd(url: str, timeout: float) -> Any:
            return io.BytesIO(b"<html>")

        assert updates.fetch_latest(opener=odd) is None

    def test_the_answer_is_cached_for_hours(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []

        def fetch() -> str:
            calls.append(1)
            return "0.15.0"

        monkeypatch.setattr(updates, "fetch_latest", fetch)
        clock = iter([0.0, 60.0, 7 * 3600.0])

        for _ in range(3):
            updates.latest_version(now=lambda: next(clock))

        assert len(calls) == 2  # the first, then again once six hours passed


class TestInstaller:
    def test_pip_of_this_interpreter_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(updates.importlib.util, "find_spec", lambda name: object())

        assert updates.installer_command("0.15.0") == [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "visionforge-studio==0.15.0",
        ]

    def test_uv_when_there_is_no_pip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(updates.importlib.util, "find_spec", lambda name: None)
        monkeypatch.setattr(updates.shutil, "which", lambda name: "C:/bin/uv.exe")

        assert updates.installer_command("0.15.0") == [
            "C:/bin/uv.exe",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--upgrade",
            "visionforge-studio==0.15.0",
        ]

    def test_none_without_either(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(updates.importlib.util, "find_spec", lambda name: None)
        monkeypatch.setattr(updates.shutil, "which", lambda name: None)

        assert updates.installer_command("0.15.0") is None


def _status(
    monkeypatch: pytest.MonkeyPatch,
    *,
    latest: str | None,
    source: bool = False,
    installer: bool = True,
    local: bool = True,
    busy: bool = False,
) -> dict[str, Any]:
    monkeypatch.setattr(updates, "latest_version", lambda: latest)
    monkeypatch.setattr(updates, "installed_from_source", lambda: source)
    monkeypatch.setattr(
        updates,
        "installer_command",
        lambda v: ["pip", "install", v] if installer else None,
    )
    status: dict[str, Any] = updates.update_status(client_is_local=local, busy=busy)
    return status


class TestStatus:
    def test_a_newer_release_can_be_applied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = _status(monkeypatch, latest="0.15.0")

        assert s["available"] and s["can_apply"]
        assert s["blocked"] is None
        assert s["command"] == "pip install --upgrade visionforge-studio==0.15.0"

    def test_nothing_newer_or_offline_offers_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for latest in ("0.14.0", None):
            s = _status(monkeypatch, latest=latest)
            assert not s["available"] and not s["can_apply"]
            assert s["blocked"] is None and s["command"] is None

    @pytest.mark.parametrize(
        ("kwargs", "reason"),
        [
            ({"source": True}, "source"),
            ({"installer": False}, "no_installer"),
            ({"local": False}, "remote"),
            ({"busy": True}, "busy"),
        ],
    )
    def test_why_the_button_cannot_install(
        self, monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, bool], reason: str
    ) -> None:
        s = _status(monkeypatch, latest="0.15.0", **kwargs)

        assert s["available"] and not s["can_apply"]
        assert s["blocked"] == reason

    def test_a_checkout_is_told_to_git_pull(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert (
            _status(monkeypatch, latest="0.15.0", source=True)["command"] == "git pull"
        )

    def test_the_check_can_be_turned_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(updates.DISABLE_ENV, "1")
        monkeypatch.setattr(
            updates, "latest_version", lambda: pytest.fail("PyPI asked")
        )

        s = updates.update_status(client_is_local=True, busy=False)

        assert s["check_disabled"] and not s["available"]

    def test_the_last_attempt_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = updates.update_dir()
        folder.mkdir(parents=True)
        (folder / "status.json").write_text(
            json.dumps({"target": "0.15.0", "ok": False, "log": "x"}), encoding="utf-8"
        )

        s = _status(monkeypatch, latest="0.15.0")

        assert s["last_attempt"]["ok"] is False


@pytest.fixture
def routes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The app with the lookup fixed at 0.15.0 and the spawn and shutdown recorded."""
    import visionforge.gui.server as server_mod
    from visionforge.gui.server import app

    monkeypatch.setattr(updates, "latest_version", lambda: "0.15.0")
    monkeypatch.setattr(updates, "installed_from_source", lambda: False)
    monkeypatch.setattr(updates, "installer_command", lambda v: ["pip", "install", v])
    calls: dict[str, Any] = {"spawned": [], "shutdown": 0}
    monkeypatch.setattr(
        updates,
        "spawn_updater",
        lambda **kw: calls["spawned"].append(kw) or Path("log"),
    )
    monkeypatch.setattr(
        server_mod,
        "request_shutdown",
        lambda: calls.__setitem__("shutdown", calls["shutdown"] + 1),
    )
    calls["app"] = app
    return calls


def _client(app: Any, host: str = "127.0.0.1") -> TestClient:
    return TestClient(app, client=(host, 50000))


class TestRoutes:
    def test_get_reports_the_update(self, routes: dict[str, Any]) -> None:
        body = _client(routes["app"]).get("/api/update").json()

        assert body["available"] and body["can_apply"]
        assert body["latest"] == "0.15.0"

    def test_get_tells_a_remote_client_it_cannot_apply(
        self, routes: dict[str, Any]
    ) -> None:
        body = _client(routes["app"], "192.168.1.20").get("/api/update").json()

        assert body["available"] and body["blocked"] == "remote"

    def test_apply_spawns_the_updater_and_stops_the_server(
        self, routes: dict[str, Any]
    ) -> None:
        resp = _client(routes["app"]).post("/api/update/apply")

        assert resp.status_code == 200
        assert resp.json() == {"status": "restarting", "target": "0.15.0"}
        assert routes["spawned"][0]["version"] == "0.15.0"
        assert routes["shutdown"] == 1

    def test_apply_refuses_a_remote_client(self, routes: dict[str, Any]) -> None:
        resp = _client(routes["app"], "10.0.0.5").post("/api/update/apply")

        assert resp.status_code == 403
        assert routes["spawned"] == [] and routes["shutdown"] == 0

    def test_apply_refuses_a_page_of_another_site(self, routes: dict[str, Any]) -> None:
        resp = _client(routes["app"]).post(
            "/api/update/apply", headers={"Origin": "https://evil.example"}
        )

        assert resp.status_code == 403
        assert routes["spawned"] == []

    def test_apply_refuses_while_a_job_runs(
        self, routes: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import visionforge.gui.api.routes as routes_mod

        monkeypatch.setattr(routes_mod._RUN_QUEUE, "is_busy", lambda: True)

        resp = _client(routes["app"]).post(
            "/api/update/apply", headers={"X-VF-Lang": "en"}
        )

        assert resp.status_code == 409
        assert "queue" in resp.json()["detail"]
        assert routes["spawned"] == []

    def test_apply_with_nothing_to_install_is_a_409(
        self, routes: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(updates, "latest_version", lambda: "0.14.0")

        assert _client(routes["app"]).post("/api/update/apply").status_code == 409


class TestUpdater:
    """The standalone script, driven with harmless commands instead of an install."""

    def test_it_records_the_outcome_and_relaunches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from visionforge.gui import updater

        relaunched: list[Any] = []
        monkeypatch.setattr(updater, "wait_for_port_free", lambda host, port: True)
        monkeypatch.setattr(
            updater, "relaunch", lambda cmd, cwd: relaunched.append((cmd, cwd))
        )
        args = {
            "host": "127.0.0.1",
            "port": 8000,
            "target": "0.15.0",
            "install": [sys.executable, "-c", "print('installed')"],
            "relaunch": ["visionforge", "gui"],
            "cwd": str(tmp_path),
            "log": str(tmp_path / "update.log"),
            "status": str(tmp_path / "status.json"),
        }

        code = updater.main(args)

        status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
        assert code == 0 and status["ok"] and status["target"] == "0.15.0"
        assert "installed" in (tmp_path / "update.log").read_text(encoding="utf-8")
        assert relaunched == [(["visionforge", "gui"], str(tmp_path))]

    def test_a_failed_install_is_recorded_and_the_old_gui_reopens(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from visionforge.gui import updater

        relaunched: list[Any] = []
        monkeypatch.setattr(updater, "wait_for_port_free", lambda host, port: True)
        monkeypatch.setattr(
            updater, "relaunch", lambda cmd, cwd: relaunched.append(cmd)
        )
        args = {
            "host": "127.0.0.1",
            "port": 8000,
            "target": "0.15.0",
            "install": [sys.executable, "-c", "import sys; sys.exit(3)"],
            "relaunch": ["gui"],
            "cwd": str(tmp_path),
            "log": str(tmp_path / "update.log"),
            "status": str(tmp_path / "status.json"),
        }

        assert updater.main(args) == 3
        status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
        assert status["ok"] is False and status["returncode"] == 3
        assert relaunched == [["gui"]]

    def test_a_server_that_never_lets_go_is_not_replaced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from visionforge.gui import updater

        ran: list[Any] = []
        monkeypatch.setattr(updater, "wait_for_port_free", lambda host, port: False)

        def install(cmd: list[str], log: Path) -> int:
            ran.append(cmd)
            return 0

        monkeypatch.setattr(updater, "run_install", install)
        monkeypatch.setattr(
            updater, "relaunch", lambda cmd, cwd: ran.append("relaunch")
        )
        args = {
            "host": "127.0.0.1",
            "port": 8000,
            "target": "0.15.0",
            "install": ["x"],
            "relaunch": ["gui"],
            "cwd": str(tmp_path),
            "log": str(tmp_path / "update.log"),
            "status": str(tmp_path / "status.json"),
        }

        assert updater.main(args) == -2
        assert (
            ran == []
        )  # neither installed over a live server nor started a second one

    def test_a_free_port_is_seen_as_free(self) -> None:
        import socket

        from visionforge.gui import updater

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        # Nothing listens on a port just released.
        assert updater.wait_for_port_free("127.0.0.1", port, timeout=1.0)


class TestUpdaterSafety:
    """What the first real run on Windows taught (ADR-120): the server's port frees
    before its processes end, and pip, finding visionforge.exe locked, removed the
    package without putting it back."""

    def _args(
        self, tmp_path: Path, install: list[str], restore: list[str] | None
    ) -> dict[str, Any]:
        return {
            "host": "127.0.0.1",
            "port": 8000,
            "pids": [],
            "target": "0.15.0",
            "install": install,
            "restore": restore,
            "relaunch": ["gui"],
            "cwd": str(tmp_path),
            "log": str(tmp_path / "update.log"),
            "status": str(tmp_path / "status.json"),
        }

    def test_this_process_is_alive_and_a_finished_one_is_not(self) -> None:
        import os
        import subprocess

        from visionforge.gui import updater

        done = subprocess.run([sys.executable, "-c", "pass"], check=True)
        assert done.returncode == 0
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()

        assert updater.process_alive(os.getpid())
        assert not updater.process_alive(child.pid)
        assert updater.wait_for_processes_gone([child.pid], timeout=1.0)

    def test_an_install_that_finds_a_file_locked_is_retried(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from visionforge.gui import updater

        log = tmp_path / "update.log"
        codes = iter([1, 1, 0])
        attempts: list[int] = []

        def fake_install(command: list[str], path: Path) -> int:
            attempts.append(1)
            code = next(codes)
            if code:
                path.write_text("ERROR: [WinError 32] in use\n", encoding="utf-8")
            return code

        monkeypatch.setattr(updater, "run_install", fake_install)

        assert updater.install_with_retries(["x"], log, pause=0) == 0
        assert len(attempts) == 3

    def test_any_other_failure_is_not_retried(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from visionforge.gui import updater

        attempts: list[int] = []

        def fake_install(command: list[str], path: Path) -> int:
            attempts.append(1)
            path.write_text("ERROR: No matching distribution\n", encoding="utf-8")
            return 1

        monkeypatch.setattr(updater, "run_install", fake_install)

        assert (
            updater.install_with_retries(["x"], tmp_path / "update.log", pause=0) == 1
        )
        assert len(attempts) == 1

    def test_a_failed_install_restores_the_previous_version(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from visionforge.gui import updater

        ran: list[list[str]] = []
        monkeypatch.setattr(updater, "wait_for_port_free", lambda host, port: True)
        monkeypatch.setattr(
            updater, "relaunch", lambda cmd, cwd: ran.append(["relaunch"])
        )

        def install(cmd: list[str], log: Path) -> int:
            ran.append(cmd)
            return 1 if cmd == ["new"] else 0

        monkeypatch.setattr(updater, "install_with_retries", install)

        code = updater.main(self._args(tmp_path, ["new"], ["old"]))

        status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
        assert code == 1 and status["ok"] is False and status["restored"] is True
        assert ran == [["new"], ["old"], ["relaunch"]]

    def test_a_successful_install_restores_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from visionforge.gui import updater

        ran: list[list[str]] = []
        monkeypatch.setattr(updater, "wait_for_port_free", lambda host, port: True)
        monkeypatch.setattr(updater, "relaunch", lambda cmd, cwd: None)

        def install(cmd: list[str], log: Path) -> int:
            ran.append(cmd)
            return 0

        monkeypatch.setattr(updater, "install_with_retries", install)

        updater.main(self._args(tmp_path, ["new"], ["old"]))

        status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
        assert ran == [["new"]] and status["restored"] is None

    def test_the_server_names_its_launcher_only_when_started_through_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        monkeypatch.setattr(sys, "argv", [r"C:\venv\Scripts\visionforge.exe", "gui"])
        assert updates.server_pids() == [os.getpid(), os.getppid()]

        monkeypatch.setattr(
            sys, "argv", [r"C:\repo\src\visionforge\__main__.py", "gui"]
        )
        assert updates.server_pids() == [os.getpid()]
