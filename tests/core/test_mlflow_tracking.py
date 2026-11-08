"""Failure-boundary tests and an executable real-SQLite adapter integration check."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if __name__ == "__main__" and not TYPE_CHECKING:
    import importlib.util

    # The isolated extra-only environment deliberately has no torch installation.
    source = (
        Path(__file__).resolve().parents[2] / "src/visionforge/core/mlflow_tracking.py"
    )
    spec = importlib.util.spec_from_file_location("vf_mlflow_spike", source)
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    MlflowTracker = adapter.MlflowTracker
else:
    from visionforge.core.mlflow_tracking import MlflowTracker


def _record(run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(exist_ok=True)
    record = {
        "config": {"training": {"epochs": 2, "token": "must-not-export"}},
        "history": [{"epoch": 1, "train_loss": 0.9}, {"epoch": 2, "train_loss": 0.7}],
        "metrics": {"accuracy": 0.875},
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return record


def test_missing_extra_degrades_and_keeps_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    record = _record(run_dir)
    monkeypatch.setitem(sys.modules, "mlflow", None)
    tracker = MlflowTracker(tmp_path / "store", "default", run_dir)
    tracker.start(record["config"])
    tracker.epoch(1, {"loss/train": 0.9})
    tracker.finish("FINISHED")
    sidecar = json.loads((run_dir / "tracking.json").read_text())
    assert sidecar["state"] == "disabled"
    assert sidecar["error"] == "ModuleNotFoundError"
    assert sidecar["status"] == "FINISHED"
    assert json.loads((run_dir / "run.json").read_text()) == record


def test_client_error_is_sanitised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    _record(run_dir)

    def fail(**kwargs: Any) -> None:
        raise RuntimeError("password=do-not-leak")

    monkeypatch.setitem(sys.modules, "mlflow", types.SimpleNamespace(MlflowClient=fail))
    MlflowTracker(tmp_path / "store", "default", run_dir).start({})
    text = (run_dir / "tracking.json").read_text()
    assert "do-not-leak" not in text
    assert json.loads(text)["error"] == "RuntimeError"


@pytest.mark.parametrize("status", ["FAILED", "KILLED"])
def test_repeated_missing_extra_retry_preserves_status_and_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    run_dir = tmp_path / "run"
    record = _record(run_dir)
    store = tmp_path / "store"
    monkeypatch.setitem(sys.modules, "mlflow", None)
    tracker = MlflowTracker(store, "default", run_dir)
    tracker.start(record["config"])
    tracker.finish(status)
    saved = json.loads((run_dir / "tracking.json").read_text())
    # An acknowledged external run must survive a later missing SDK too.
    saved["run_id"] = "previously-created-run"
    (run_dir / "tracking.json").write_text(json.dumps(saved))
    for _ in range(2):
        MlflowTracker(store, "default", run_dir).retry(record)
        actual = json.loads((run_dir / "tracking.json").read_text())
        assert actual["state"] == "disabled"
        assert actual["error"] == "ModuleNotFoundError"
        assert actual["status"] == status
        assert actual["run_id"] == saved["run_id"]
        assert actual["identity"] == saved["identity"]


def test_cleanup_failure_is_observable(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _record(run_dir)

    class FailingCleanup:
        def log_artifact(self, *args: Any) -> None:
            pass

        def set_terminated(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("private cleanup diagnostic")

    tracker = MlflowTracker(tmp_path / "store", "default", run_dir)
    tracker._client = FailingCleanup()
    tracker._run_id = "verified-run"
    tracker.finish("KILLED")
    saved = json.loads((run_dir / "tracking.json").read_text())
    assert saved["state"] == "failed"
    assert saved["error"] == "RuntimeError"
    assert saved["run_id"] == "verified-run"
    assert saved["status"] == "KILLED"


def test_full_disk_status_write_cannot_fail_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    _record(run_dir)
    monkeypatch.setitem(sys.modules, "mlflow", None)

    def fail(self: MlflowTracker) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(MlflowTracker, "_save", fail)
    tracker = MlflowTracker(tmp_path / "store", "default", run_dir)
    tracker.start({})
    tracker.finish("FAILED")
    assert (run_dir / "run.json").is_file()


def test_retry_without_sidecar_does_not_import_mlflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    record = _record(run_dir)
    monkeypatch.setitem(sys.modules, "mlflow", None)
    MlflowTracker(tmp_path / "store", "default", run_dir).retry(record)
    state = json.loads((run_dir / "tracking.json").read_text())
    assert state["state"] == "failed"
    assert state["error"] == "FileNotFoundError"
    assert state["run_id"] is None


def test_parameters_reject_ambiguity_and_exclude_sensitive_keys() -> None:
    assert MlflowTracker._parameters(
        {"training": {"epochs": 2, "api_token": "secret"}, "dataset_path": "C:/private"}
    ) == {"training.epochs": "2"}
    with pytest.raises(ValueError, match="Ambiguous"):
        MlflowTracker._parameters({"training.epochs": 2})
    assert MlflowTracker._parameters({"long": "x" * 251}) == {}


@pytest.mark.parametrize("profile", ["../other", "", "A", "a" * 41])
def test_invalid_profile_is_best_effort(tmp_path: Path, profile: str) -> None:
    run_dir = tmp_path / "run"
    _record(run_dir)
    MlflowTracker(tmp_path / "store", profile, run_dir).start({})
    assert json.loads((run_dir / "tracking.json").read_text())["error"] == "ValueError"


def _real_checks() -> None:
    """Run directly in the isolated spike interpreter with the real MLflow extra."""
    import tempfile
    from unittest.mock import patch

    from mlflow import MlflowClient

    with tempfile.TemporaryDirectory(prefix="vf118-adapter-") as temporary:
        root = Path(temporary)
        run_dir = root / "run"
        record = _record(run_dir)
        store = root / "store"
        tracker = MlflowTracker(store, "default", run_dir)
        tracker.start(record["config"])
        tracker.epoch(1, {"train_loss": 0.9, "missing": None, "nan": float("nan")})
        tracker.finish("FINISHED")
        sidecar = json.loads((run_dir / "tracking.json").read_text())
        assert sidecar["state"] == "finished", sidecar
        run_id = sidecar["run_id"]
        client = MlflowClient(
            tracking_uri=f"sqlite:///{(store / 'mlflow.db').as_posix()}"
        )
        assert client.get_metric_history(run_id, "loss/train")[0].value == 0.9
        assert client.get_run(run_id).data.params == {"training.epochs": "2"}
        tracker.retry(record)
        tracker.retry(record)
        rows = client.get_metric_history(run_id, "loss/train")
        assert [(row.step, row.value) for row in rows] == [(1, 0.9), (2, 0.7)]
        assert client.get_run(run_id).info.status == "FINISHED"
        assert json.loads((run_dir / "tracking.json").read_text())["run_id"] == run_id
        assert len(client.search_runs([client.get_run(run_id).info.experiment_id])) == 1
        downloaded = Path(client.download_artifacts(run_id, "run.json", str(root)))
        assert json.loads(downloaded.read_text()) == record
        for status in ("KILLED", "FAILED"):
            tracker.finish(status)
            tracker.retry(record)
            assert client.get_run(run_id).info.status == status

        record["history"].append({"epoch": 3, "train_loss": 0.6})
        (run_dir / "run.json").write_text(json.dumps(record))
        with patch.object(
            MlflowClient, "log_batch", side_effect=RuntimeError("private")
        ):
            MlflowTracker(store, "default", run_dir).retry(record)
        failed = json.loads((run_dir / "tracking.json").read_text())
        assert failed["state"] == "failed"
        assert failed["run_id"] == run_id
        assert failed["status"] == "FAILED"
        with patch.dict(sys.modules, {"mlflow": None}):
            MlflowTracker(store, "default", run_dir).retry(record)
        disabled = json.loads((run_dir / "tracking.json").read_text())
        assert disabled["run_id"] == run_id
        assert disabled["status"] == "FAILED"
        assert disabled["state"] == "disabled"
        MlflowTracker(store, "default", run_dir).retry(record)
        restored = json.loads((run_dir / "tracking.json").read_text())
        assert restored["state"] == "finished"
        assert restored["status"] == "FAILED"
        assert restored["run_id"] == run_id
        assert len(client.get_metric_history(run_id, "loss/train")) == 3

        # Remote create succeeded but the local acknowledgement did not survive.
        (run_dir / "tracking.json").unlink()
        recovered = MlflowTracker(store, "default", run_dir)
        recovered.start(record["config"])
        assert json.loads((run_dir / "tracking.json").read_text())["run_id"] == run_id
        recovered.epoch(1, {"loss/train": 0.1})
        assert (
            json.loads((run_dir / "tracking.json").read_text())["error"] == "ValueError"
        )
        recovered.finish("FAILED")
        assert client.get_run(run_id).info.status == "FAILED"

        # A forged external ID must not touch another profile's external run.
        other_dir = root / "other"
        _record(other_dir)
        other = MlflowTracker(store, "other", other_dir)
        other.start({})
        other.finish("FINISHED")
        other_state = json.loads((other_dir / "tracking.json").read_text())
        state = json.loads((run_dir / "tracking.json").read_text())
        state["run_id"] = other_state["run_id"]
        (run_dir / "tracking.json").write_text(json.dumps(state))
        forged = MlflowTracker(store, "default", run_dir)
        forged.start({})
        forged.finish("FAILED")
        assert client.get_run(other_state["run_id"]).info.status == "FINISHED"
        assert json.loads((run_dir / "tracking.json").read_text())["state"] == "failed"
        # MLflow retains SQLAlchemy pooled connections until process shutdown.
        client._tracking_client.store.engine.dispose()
        print(
            "PASS: real SQLite adapter replay, statuses, recovery, artifact and identity checks"
        )


if __name__ == "__main__":
    _real_checks()
