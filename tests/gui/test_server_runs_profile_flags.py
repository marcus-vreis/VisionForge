"""A run folder another job still uses, and the flags server mode reports.

Two findings of the server-mode review (phase 0, ADR-212): deleting a run that a
resumed job or an export is still using, and the ``shared``/``can_reveal`` flags
when the login proxy sits on the same machine.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import visionforge.gui.api.execution as execution_mod
import visionforge.gui.api.runs_routes as runs_routes_mod
from visionforge.gui.api import folder_opener, profile_tokens, routes
from visionforge.gui.api.run_queue import QueuedJob
from visionforge.gui.server_mode import ServerMode
from visionforge.utils.messages import tr

RUN_ID = "20261124_100000_000000"


def _run_dir(models: Path) -> Path:
    run_dir = models / "exp" / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{}", encoding="utf-8")
    return run_dir


async def _never() -> None:  # pragma: no cover - a queued job is never started
    raise AssertionError("the queued job must not start")


@pytest.fixture
def local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from visionforge.gui.server import app

    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    return TestClient(app, raise_server_exceptions=False)


class TestDeleteARunInUse:
    """Finding 5: the History id is the folder name, not the id of the job."""

    def test_a_resumed_job_waiting_for_the_folder_keeps_it(
        self, local: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = _run_dir(tmp_path / "models")
        # A resume gets a new job id and writes into the old folder (resume_dir).
        job = QueuedJob(
            run_id="exp_20261124_110000_000000",
            label="exp (retomando)",
            task="classification",
            strategy="simple",
            start=_never,
            resume_dir=run_dir,
        )
        monkeypatch.setattr(execution_mod._RUN_QUEUE, "_pending", deque([job]))

        response = local.delete(f"/api/runs/{RUN_ID}")

        assert response.status_code == 409
        assert response.json()["detail"] == tr("run.delete_busy")
        assert (run_dir / "run.json").is_file()

    def test_a_folder_being_exported_is_kept(
        self, local: TestClient, tmp_path: Path
    ) -> None:
        run_dir = _run_dir(tmp_path / "models")
        assert routes._EVIDENCE_EXPORTS.reserve(run_dir)
        try:
            response = local.delete(f"/api/runs/{RUN_ID}")
        finally:
            routes._EVIDENCE_EXPORTS.release(run_dir)

        assert response.status_code == 409
        assert response.json()["detail"] == tr("run.delete_busy")
        assert (run_dir / "run.json").is_file()

    def test_the_delete_holds_the_export_off_while_it_runs(
        self, local: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = _run_dir(tmp_path / "models")
        seen: dict[str, bool] = {}

        def rmtree(path: Path) -> None:
            seen["export_refused"] = not routes._EVIDENCE_EXPORTS.reserve(run_dir)

        monkeypatch.setattr(runs_routes_mod.shutil, "rmtree", rmtree)

        assert local.delete(f"/api/runs/{RUN_ID}").status_code == 200
        assert seen == {"export_refused": True}
        # Released afterwards: the folder can be exported again.
        assert routes._EVIDENCE_EXPORTS.reserve(run_dir)
        routes._EVIDENCE_EXPORTS.release(run_dir)

    def test_a_folder_the_os_will_not_remove_is_a_409_not_a_500(
        self, local: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir = _run_dir(tmp_path / "models")

        def rmtree(path: Path) -> None:
            raise PermissionError(13, "in use by another process", str(path))

        monkeypatch.setattr(runs_routes_mod.shutil, "rmtree", rmtree)

        response = local.delete(f"/api/runs/{RUN_ID}")

        assert response.status_code == 409
        assert response.json()["detail"] == tr(
            "run.delete_failed", error="in use by another process"
        )
        assert routes._EVIDENCE_EXPORTS.reserve(run_dir)
        routes._EVIDENCE_EXPORTS.release(run_dir)


PROXY_ON_THIS_MACHINE = "127.0.0.1"


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Server mode behind a login proxy on this machine (the ADR-212 topology)."""
    from visionforge.gui.server import app

    monkeypatch.setattr(routes, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(routes, "_MODELS_DIR", tmp_path / "models")
    # The bind host is loopback too (``--host 127.0.0.1``).
    monkeypatch.setattr(profile_tokens, "is_loopback_host", lambda _: True)
    monkeypatch.setattr(folder_opener, "opener_available", lambda: True)
    had = hasattr(app.state, "server_mode")
    saved = getattr(app.state, "server_mode", None)
    app.state.server_mode = ServerMode.from_options(
        enabled=True, trusted_proxies=[PROXY_ON_THIS_MACHINE]
    )
    yield TestClient(
        app, base_url="http://localhost", client=(PROXY_ON_THIS_MACHINE, 50000)
    )
    if had:
        app.state.server_mode = saved
    else:
        del app.state.server_mode


USER = {"X-Forwarded-User": "ana@lab.org"}


class TestServerModeFlagsBehindALocalProxy:
    """Finding 10: a loopback proxy is not the person at the keyboard."""

    @pytest.mark.parametrize("service", ["hub", "wandb", "zenodo", "github", "comet"])
    def test_a_token_is_reported_as_shared(
        self, served: TestClient, service: str
    ) -> None:
        response = served.get(f"/api/{service}/token", headers=USER)
        assert response.status_code == 200
        assert response.json()["shared"] is True

    def test_the_run_folder_cannot_be_revealed(
        self, served: TestClient, tmp_path: Path
    ) -> None:
        session = served.get("/api/session", headers=USER).json()
        models = tmp_path / "profiles" / session["profile"] / "models"
        _run_dir_with_history(models)

        response = served.get(f"/api/runs/{RUN_ID}", headers=USER)

        assert response.status_code == 200
        assert response.json()["can_reveal"] is False


def _run_dir_with_history(models: Path) -> Path:
    run_dir = models / "exp" / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        '{"experiment": "exp", "timestamp": "2026-11-24T10:00:00",'
        ' "status": "completed", "config": {"name": "exp", "task": "binary"}}',
        encoding="utf-8",
    )
    return run_dir
