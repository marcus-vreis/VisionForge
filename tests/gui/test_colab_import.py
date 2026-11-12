"""Bringing a Colab run back and importing it (ADR-163, phase 3), against a fake CLI."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

import visionforge.gui.api.colab_run_routes as colab_run_mod
import visionforge.gui.api.execution as execution_mod
from visionforge.core.experiment_bundle import export_experiment_bundle
from visionforge.gui.api import colab_jobs as cj
from visionforge.gui.api import colab_remote_runner as runner
from visionforge.gui.api.run_events import RunEventLog
from visionforge.gui.api.run_history import load_runs

from .conftest import job_run, release_queue, start_job
from .test_colab_jobs import PAYLOAD, FakeCli, event_line, run

REMOTE = "/content/vf/run-bundle.zip"


def _bundle_bytes(tmp_path: Path) -> bytes:
    """A real bundle of a small run, as the VM's export writes it."""
    source = tmp_path / "vm" / "exp" / "20261105_120000_000001"
    source.mkdir(parents=True)
    (source / "best_model.pth").write_bytes(b"weights")
    record = {
        "experiment": "exp",
        "timestamp": "2026-11-05T12:01:00",
        "status": "completed",
        "device_used": "cuda",
        "environment": {"python": "3.12.12", "torch": "2.9.0", "gpu": "Tesla T4"},
        "run_dir": str(source),
        "config": {"name": "exp", "schema_version": 1},
        "metrics": {"best_val_loss": 0.4, "best_epoch": 2, "total_epochs": 2},
        "history": [],
        "artifacts": {"model": str(source / "best_model.pth"), "graphics": []},
    }
    (source / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return export_experiment_bundle(source, tmp_path / "vm.zip").read_bytes()


def _ok_lines(bundle: dict[str, Any] | None) -> list[str]:
    return [
        event_line({"event": "epoch_end", "epoch": 1}),
        event_line(
            {"event": "job_end", "status": "ok", "report": {}, "bundle": bundle}
        ),
    ]


# ── the download, before the VM stops ────────────────────────────────────────


def test_the_bundle_is_downloaded_before_the_vm_stops(tmp_path: Path) -> None:
    data = _bundle_bytes(tmp_path)
    cli = FakeCli(_ok_lines({"path": REMOTE, "size": len(data)}))
    cli.download_bytes = data

    result, events = run(cli)

    assert result.outcome == "completed"
    assert result.bundle is not None
    assert result.bundle.read_bytes() == data
    commands = cli.commands()
    assert commands.index("download") < commands.index("stop")
    download = cli.calls[commands.index("download")]
    assert download[:4] == ["download", "-s", download[2], REMOTE]
    assert {"event": "remote", "stage": "download", "session": download[2]} in events


def test_a_failed_download_fails_the_job_and_still_stops_the_vm(
    tmp_path: Path,
) -> None:
    cli = FakeCli(_ok_lines({"path": REMOTE, "size": 100}))

    result, _ = run(cli)

    assert result.outcome == "failed"
    assert result.bundle is None
    assert "404: file not found" in (result.error or "")
    assert cli.commands()[-1] == "stop"
    assert list((tmp_path / "colab").rglob("*.zip")) == []


def test_a_download_of_the_wrong_size_is_discarded(tmp_path: Path) -> None:
    cli = FakeCli(_ok_lines({"path": REMOTE, "size": 100}))
    cli.download_bytes = b"truncated"

    result, _ = run(cli)

    assert result.outcome == "failed"
    assert result.bundle is None
    assert cli.commands()[-1] == "stop"
    assert list((tmp_path / "colab").rglob("*.zip")) == []


def test_a_bundle_over_the_limit_is_never_downloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cj, "MAX_BUNDLE_BYTES", 10)
    cli = FakeCli(_ok_lines({"path": REMOTE, "size": 11}))

    result, _ = run(cli)

    assert result.outcome == "failed"
    assert "download" not in cli.commands()
    assert cli.commands()[-1] == "stop"


@pytest.mark.parametrize(
    "path", ["/etc/passwd", "/content/vf/../../etc/passwd", None, "relative.zip"]
)
def test_a_bundle_path_outside_the_job_folder_is_refused(path: Any) -> None:
    cli = FakeCli(_ok_lines({"path": path, "size": 1}))

    result, _ = run(cli)

    assert result.outcome == "failed"
    assert "download" not in cli.commands()


def test_a_job_without_a_bundle_completes_without_downloading() -> None:
    cli = FakeCli(_ok_lines(None))

    result, _ = run(cli)

    assert result.outcome == "completed"
    assert result.bundle is None
    assert "download" not in cli.commands()


# ── the import, in the executor ──────────────────────────────────────────────


def _execute(
    monkeypatch: pytest.MonkeyPatch, result: cj.RemoteResult, models: Path
) -> list[dict[str, Any]]:

    class Jobs:
        def run_job(
            self, slug, accelerator, payload, on_event, cancelled, consent=None
        ):  # type: ignore[no-untyped-def]
            on_event({"event": "remote", "stage": "download"})
            return result

    monkeypatch.setattr(colab_run_mod, "_COLAB_JOBS", Jobs())
    log = RunEventLog("r1")

    async def go() -> list[dict[str, Any]]:
        start_job(execution_mod, "r1", log=log)
        await colab_run_mod._execute_colab("r1", "default", "T4", PAYLOAD, models)
        return [e async for e in log.read()]

    return asyncio.run(go())


def test_the_executor_imports_the_run_and_marks_it_as_colab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "job.zip"
    local.write_bytes(_bundle_bytes(tmp_path))
    models = tmp_path / "profile" / "models"
    try:
        _execute(
            monkeypatch,
            cj.RemoteResult("completed", report={"train": {}}, bundle=local),
            models,
        )
        current = job_run(execution_mod, "r1")
        assert current is not None
        assert current["status"] == "completed"
        assert current["error"] is None
        run_dir = Path(current["run_dir"])
        assert run_dir == models.resolve() / "exp" / "20261105_120000_000001"
        data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert data["origin"]["kind"] == "colab"
        assert data["origin"]["accelerator"] == "T4"
        assert data["device_used"] == "cuda"
        assert [s.origin for s in load_runs(models)] == ["colab"]
        assert not local.exists()
    finally:
        release_queue(execution_mod)


def test_a_bundle_that_does_not_import_fails_the_job_and_writes_no_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "job.zip"
    local.write_bytes(b"not a zip")
    models = tmp_path / "profile" / "models"
    try:
        events = _execute(
            monkeypatch, cj.RemoteResult("completed", report={}, bundle=local), models
        )
        current = job_run(execution_mod, "r1")
        assert current is not None
        assert current["status"] == "failed"
        assert current["run_dir"] is None
        assert "invalid experiment bundle" in current["error"]
        assert not models.exists()
        assert not local.exists()
        # The stream still closes after the events that did arrive.
        assert events == [{"event": "remote", "stage": "download"}]
    finally:
        release_queue(execution_mod)


def test_an_imported_run_opens_in_the_runs_routes_like_any_other(
    tmp_path: Path,
) -> None:
    from fastapi.testclient import TestClient

    import visionforge.gui.api.routes as routes_mod
    from visionforge.core.experiment_bundle import import_experiment_bundle
    from visionforge.gui.server import app

    bundle = tmp_path / "job.zip"
    bundle.write_bytes(_bundle_bytes(tmp_path))
    models = tmp_path / "models"
    run_dir = import_experiment_bundle(bundle, models, origin={"kind": "colab"})

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(routes_mod, "_MODELS_DIR", models)
        client = TestClient(app, raise_server_exceptions=True)
        listed = client.get("/api/runs").json()
        detail = client.get(f"/api/runs/{run_dir.name}")

    assert [(r["run_id"], r["origin"]) for r in listed] == [(run_dir.name, "colab")]
    assert detail.status_code == 200
    body = detail.json()
    assert body["device_used"] == "cuda"
    assert body["environment"]["gpu"] == "Tesla T4"
    assert body["run_dir"] == str(run_dir)


# ── the runner side ──────────────────────────────────────────────────────────


def test_the_runner_packs_the_trained_run_as_a_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "REMOTE_ROOT", tmp_path / "vf")
    (tmp_path / "vf").mkdir()
    source = tmp_path / "vf" / "outputs" / "models" / "exp" / "20261105_120000_1"
    source.mkdir(parents=True)
    (source / "best_model.pth").write_bytes(b"weights")
    (source / "run.json").write_text(
        json.dumps({"artifacts": {"model": str(source / "best_model.pth")}}),
        encoding="utf-8",
    )

    packed = runner.pack_run({"train": {"run_dir": str(source)}})

    assert packed is not None
    bundle = Path(packed["path"])
    assert bundle == tmp_path / "vf" / runner.BUNDLE_NAME
    assert packed["size"] == bundle.stat().st_size


def test_the_runner_has_nothing_to_pack_without_a_run_folder() -> None:
    assert runner.pack_run({"eval": {"accuracy": 0.9}}) is None
    assert runner.pack_run(None) is None
