"""Opening a run's folder from the History: where it may run and what it opens.

The window appears on the *server's* desktop, so the route has to refuse a client
that is not the same machine, and the page learns that from ``can_reveal`` on the
run detail. The folder comes from ``_find_run_dir`` only -- never from a path the
client sent. The opener itself is replaced everywhere here: no test opens a real
file manager.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from visionforge.gui.api import folder_opener
from visionforge.gui.api.folder_opener import FolderOpenError

_RUN_ID = "20260522_100000_000000"


@pytest.fixture
def app_and_routes():  # type: ignore[return]
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    return app, routes_mod


def _write_run(models: Path, run_id: str = _RUN_ID) -> Path:
    run_dir = models / "exp1" / run_id
    run_dir.mkdir(parents=True)
    data = {
        "id": f"exp1_{run_id}",
        "experiment": "exp1",
        "timestamp": "2026-05-22T10:00:00",
        "status": "completed",
        "run_dir": str(run_dir.resolve()),
        "config": {},
        "metrics": {},
        "history": [],
        "artifacts": {},
        "tests": [],
    }
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
    return run_dir


class _Opened:
    """Stands in for ``open_folder`` and remembers what it was asked to open."""

    def __init__(self, error: Exception | None = None) -> None:
        self.paths: list[Path] = []
        self._error = error

    def __call__(self, path: Path) -> None:
        self.paths.append(path)
        if self._error is not None:
            raise self._error


def _client(app: Any, host: str) -> TestClient:
    return TestClient(app, raise_server_exceptions=True, client=(host, 50000))


class TestRevealRoute:
    def test_opens_the_folder_of_that_run(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        run_dir = _write_run(tmp_path)
        opened = _Opened()
        with (
            patch.object(routes_mod, "_MODELS_DIR", tmp_path),
            patch.object(routes_mod, "open_folder", opened),
        ):
            resp = _client(app, "127.0.0.1").post(f"/api/runs/{run_dir.name}/reveal")
        assert resp.status_code == 200
        assert [p.resolve() for p in opened.paths] == [run_dir.resolve()]
        assert resp.json()["run_id"] == run_dir.name
        assert Path(resp.json()["run_dir"]) == run_dir.resolve()

    def test_finds_the_run_by_its_recorded_id_too(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        run_dir = _write_run(tmp_path)
        opened = _Opened()
        with (
            patch.object(routes_mod, "_MODELS_DIR", tmp_path),
            patch.object(routes_mod, "open_folder", opened),
        ):
            resp = _client(app, "127.0.0.1").post(f"/api/runs/exp1_{_RUN_ID}/reveal")
        assert resp.status_code == 200
        assert [p.resolve() for p in opened.paths] == [run_dir.resolve()]

    @pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
    def test_loopback_clients_are_served(
        self, app_and_routes: tuple, tmp_path: Path, host: str
    ) -> None:
        app, routes_mod = app_and_routes
        run_dir = _write_run(tmp_path)
        opened = _Opened()
        with (
            patch.object(routes_mod, "_MODELS_DIR", tmp_path),
            patch.object(routes_mod, "open_folder", opened),
        ):
            resp = _client(app, host).post(f"/api/runs/{run_dir.name}/reveal")
        assert resp.status_code == 200
        assert len(opened.paths) == 1

    def test_unknown_run_is_404_and_opens_nothing(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        opened = _Opened()
        with (
            patch.object(routes_mod, "_MODELS_DIR", tmp_path),
            patch.object(routes_mod, "open_folder", opened),
        ):
            resp = _client(app, "127.0.0.1").post("/api/runs/does_not_exist/reveal")
        assert resp.status_code == 404
        assert opened.paths == []

    @pytest.mark.parametrize("host", ["192.168.1.20", "10.0.0.5", "testclient"])
    def test_a_client_on_another_machine_is_403(
        self, app_and_routes: tuple, tmp_path: Path, host: str
    ) -> None:
        app, routes_mod = app_and_routes
        run_dir = _write_run(tmp_path)
        opened = _Opened()
        with (
            patch.object(routes_mod, "_MODELS_DIR", tmp_path),
            patch.object(routes_mod, "open_folder", opened),
        ):
            resp = _client(app, host).post(f"/api/runs/{run_dir.name}/reveal")
        assert resp.status_code == 403
        assert opened.paths == []

    def test_a_remote_client_learns_nothing_about_which_runs_exist(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        """403 before the lookup, so 403 and 404 do not tell run ids apart."""
        app, routes_mod = app_and_routes
        _write_run(tmp_path)
        with (
            patch.object(routes_mod, "_MODELS_DIR", tmp_path),
            patch.object(routes_mod, "open_folder", _Opened()),
        ):
            resp = _client(app, "192.168.1.20").post("/api/runs/does_not_exist/reveal")
        assert resp.status_code == 403

    def test_a_path_sent_by_the_client_is_never_used(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        run_dir = _write_run(tmp_path)
        elsewhere = tmp_path / "somewhere_else"
        elsewhere.mkdir()
        opened = _Opened()
        with (
            patch.object(routes_mod, "_MODELS_DIR", tmp_path),
            patch.object(routes_mod, "open_folder", opened),
        ):
            resp = _client(app, "127.0.0.1").post(
                f"/api/runs/{run_dir.name}/reveal",
                params={"path": str(elsewhere), "run_dir": str(elsewhere)},
                json={"path": str(elsewhere), "run_dir": str(elsewhere)},
            )
        assert resp.status_code == 200
        assert [p.resolve() for p in opened.paths] == [run_dir.resolve()]

    @pytest.mark.parametrize("run_id", ["..", "../outside", "outside", "exp1"])
    def test_an_id_that_is_not_a_run_resolves_to_no_folder(
        self, app_and_routes: tuple, tmp_path: Path, run_id: str
    ) -> None:
        """Only a run's own folder name or recorded id selects a folder."""
        _, routes_mod = app_and_routes
        models = tmp_path / "models"
        models.mkdir()
        _write_run(models)
        (tmp_path / "outside").mkdir()
        with patch.object(routes_mod, "_MODELS_DIR", models):
            assert routes_mod._find_run_dir(run_id) is None

    def test_a_file_manager_that_cannot_start_is_a_500_with_the_reason(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        run_dir = _write_run(tmp_path)
        opened = _Opened(error=FolderOpenError("xdg-open not found"))
        with (
            patch.object(routes_mod, "_MODELS_DIR", tmp_path),
            patch.object(routes_mod, "open_folder", opened),
        ):
            resp = _client(app, "127.0.0.1").post(f"/api/runs/{run_dir.name}/reveal")
        assert resp.status_code == 500
        assert "xdg-open not found" in resp.json()["detail"]


class TestCanRevealFlag:
    """The run detail tells the page whether to offer the button."""

    def _detail(
        self, app_and_routes: tuple, tmp_path: Path, host: str
    ) -> dict[str, Any]:
        app, routes_mod = app_and_routes
        run_dir = _write_run(tmp_path)
        with (
            patch.object(routes_mod, "_MODELS_DIR", tmp_path),
            patch.object(folder_opener, "opener_available", lambda: True),
        ):
            resp = _client(app, host).get(f"/api/runs/{run_dir.name}")
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        return body

    def test_true_for_the_machine_the_server_runs_on(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        body = self._detail(app_and_routes, tmp_path, "127.0.0.1")
        assert body["can_reveal"] is True

    def test_false_for_any_other_client(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        body = self._detail(app_and_routes, tmp_path, "192.168.1.20")
        assert body["can_reveal"] is False

    def test_the_absolute_folder_is_in_the_payload(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        body = self._detail(app_and_routes, tmp_path, "192.168.1.20")
        assert Path(body["run_dir"]).is_absolute()

    def test_false_when_the_machine_has_no_way_to_open_a_folder(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        app, routes_mod = app_and_routes
        run_dir = _write_run(tmp_path)
        with (
            patch.object(routes_mod, "_MODELS_DIR", tmp_path),
            patch.object(folder_opener, "opener_available", lambda: False),
        ):
            resp = _client(app, "127.0.0.1").get(f"/api/runs/{run_dir.name}")
        assert resp.json()["can_reveal"] is False


class TestIsLoopbackHost:
    @pytest.mark.parametrize(
        "host",
        ["127.0.0.1", "127.0.0.2", "::1", "localhost", "LOCALHOST", "::ffff:127.0.0.1"],
    )
    def test_this_machine(self, host: str) -> None:
        assert folder_opener.is_loopback_host(host) is True

    @pytest.mark.parametrize(
        "host",
        [
            None,
            "",
            "testclient",
            "192.168.1.20",
            "10.0.0.5",
            "0.0.0.0",
            "8.8.8.8",
            "::ffff:192.168.1.20",
            "2001:db8::1",
            "127.0.0.1.evil.example",
        ],
    )
    def test_anything_else(self, host: str | None) -> None:
        assert folder_opener.is_loopback_host(host) is False


class TestOpenFolderPlatforms:
    """Each platform gets its own command, as an argument list and no shell."""

    @staticmethod
    def _spawn_recorder() -> tuple[list[tuple[Any, dict]], types.SimpleNamespace]:
        calls: list[tuple[Any, dict]] = []

        def popen(args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))

        fake = types.SimpleNamespace(Popen=popen, DEVNULL=-3)
        return calls, fake

    def test_windows_uses_startfile(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        started: list[str] = []
        monkeypatch.setattr(
            folder_opener, "sys", types.SimpleNamespace(platform="win32")
        )
        monkeypatch.setattr(
            folder_opener.os, "startfile", started.append, raising=False
        )
        calls, fake = self._spawn_recorder()
        monkeypatch.setattr(folder_opener, "subprocess", fake)
        folder_opener.open_folder(tmp_path)
        assert started == [str(tmp_path.resolve())]
        assert calls == []

    def test_macos_uses_open(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            folder_opener, "sys", types.SimpleNamespace(platform="darwin")
        )
        calls, fake = self._spawn_recorder()
        monkeypatch.setattr(folder_opener, "subprocess", fake)
        folder_opener.open_folder(tmp_path)
        assert [c[0] for c in calls] == [["open", str(tmp_path.resolve())]]
        assert not calls[0][1].get("shell")

    @pytest.mark.parametrize("platform", ["linux", "freebsd14"])
    def test_linux_and_the_rest_use_xdg_open(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str
    ) -> None:
        monkeypatch.setattr(
            folder_opener, "sys", types.SimpleNamespace(platform=platform)
        )
        calls, fake = self._spawn_recorder()
        monkeypatch.setattr(folder_opener, "subprocess", fake)
        folder_opener.open_folder(tmp_path)
        assert [c[0] for c in calls] == [["xdg-open", str(tmp_path.resolve())]]
        assert not calls[0][1].get("shell")

    def test_a_dash_leading_name_cannot_pass_for_an_option(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            folder_opener, "sys", types.SimpleNamespace(platform="linux")
        )
        calls, fake = self._spawn_recorder()
        monkeypatch.setattr(folder_opener, "subprocess", fake)
        folder_opener.open_folder(Path("-n"))
        target = calls[0][0][1]
        assert not target.startswith("-")
        assert Path(target).is_absolute()

    def test_a_missing_command_becomes_folder_open_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            folder_opener, "sys", types.SimpleNamespace(platform="linux")
        )

        def missing(*_args: Any, **_kwargs: Any) -> None:
            raise FileNotFoundError("xdg-open")

        monkeypatch.setattr(
            folder_opener,
            "subprocess",
            types.SimpleNamespace(Popen=missing, DEVNULL=-3),
        )
        with pytest.raises(FolderOpenError):
            folder_opener.open_folder(tmp_path)


class TestOpenerAvailable:
    def test_windows_and_macos_always_have_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(folder_opener.shutil, "which", lambda _name: None)
        for platform in ("win32", "darwin"):
            monkeypatch.setattr(
                folder_opener, "sys", types.SimpleNamespace(platform=platform)
            )
            assert folder_opener.opener_available() is True

    def test_linux_needs_xdg_open_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            folder_opener, "sys", types.SimpleNamespace(platform="linux")
        )
        monkeypatch.setattr(folder_opener.shutil, "which", lambda _name: None)
        assert folder_opener.opener_available() is False
        monkeypatch.setattr(
            folder_opener.shutil, "which", lambda _name: "/usr/bin/xdg-open"
        )
        assert folder_opener.opener_available() is True

    def test_can_reveal_needs_both_the_machine_and_an_opener(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(folder_opener, "opener_available", lambda: True)
        assert folder_opener.can_reveal("127.0.0.1") is True
        assert folder_opener.can_reveal("192.168.1.20") is False
        assert folder_opener.can_reveal(None) is False
        monkeypatch.setattr(folder_opener, "opener_available", lambda: False)
        assert folder_opener.can_reveal("127.0.0.1") is False
