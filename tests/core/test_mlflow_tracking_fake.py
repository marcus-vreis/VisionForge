"""MlflowTracker against a named in-memory fake of the MLflow client.

CI does not install the optional ``mlflow`` extra, so these tests exercise the
export logic (experiment and run identity, parameters, epoch metrics, final
artifact, retry) through a small fake with the same method names. The real
SQLite store is covered by ``python tests/core/test_mlflow_tracking.py`` in an
environment that has the extra.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from visionforge.core.mlflow_tracking import MlflowTracker

from ..fake_mlflow import FakeMlflowClient, FakeRun


def _run_dir(tmp_path: Path, name: str = "run") -> tuple[Path, dict[str, Any]]:
    run_dir = tmp_path / name
    run_dir.mkdir()
    record: dict[str, Any] = {
        "config": {
            "training": {"epochs": 2, "learning_rate": 0.001},
            "data": {"base_dir": "C:/private"},
        },
        "history": [
            {"epoch": 1, "train_loss": 0.9, "val_accuracy": 0.5},
            {"epoch": 2, "train_loss": 0.7, "val_accuracy": 0.75},
        ],
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return run_dir, record


def _sidecar(run_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((run_dir / "tracking.json").read_text())
    return data


def _train(tracker: MlflowTracker, record: dict[str, Any], status: str) -> None:
    tracker.start(record["config"])
    for row in record["history"]:
        tracker.epoch(row["epoch"], {k: v for k, v in row.items() if k != "epoch"})
    tracker.finish(status)


def test_training_exports_parameters_metrics_and_record(
    tmp_path: Path, fake_mlflow: type[FakeMlflowClient]
) -> None:
    run_dir, record = _run_dir(tmp_path)
    _train(MlflowTracker(tmp_path / "store", "ana", run_dir), record, "FINISHED")

    assert list(fake_mlflow.experiments) == ["visionforge-ana"]
    (run,) = fake_mlflow.runs.values()
    assert run.status == "FINISHED"
    assert run.tags["vf.profile"] == "ana"
    assert run.params == {"training.epochs": "2", "training.learning_rate": "0.001"}
    assert [(m.step, m.value) for m in run.metrics["loss/train"]] == [
        (1, 0.9),
        (2, 0.7),
    ]
    assert [(m.step, m.value) for m in run.metrics["accuracy/val"]] == [
        (1, 0.5),
        (2, 0.75),
    ]
    assert run.artifacts == ["run.json"]
    saved = _sidecar(run_dir)
    assert saved["state"] == "finished"
    assert saved["status"] == "FINISHED"
    assert saved["run_id"] == run.run_id
    assert saved["error"] is None


def test_non_finite_and_non_numeric_values_are_skipped(
    tmp_path: Path, fake_mlflow: type[FakeMlflowClient]
) -> None:
    run_dir, record = _run_dir(tmp_path)
    tracker = MlflowTracker(tmp_path / "store", "ana", run_dir)
    tracker.start(record["config"])
    tracker.epoch(
        1, {"train_loss": float("nan"), "note": "x", "flag": True, "lr": 0.01}
    )
    (run,) = fake_mlflow.runs.values()
    assert list(run.metrics) == ["lr"]


def test_same_epoch_twice_is_logged_once(
    tmp_path: Path, fake_mlflow: type[FakeMlflowClient]
) -> None:
    run_dir, record = _run_dir(tmp_path)
    tracker = MlflowTracker(tmp_path / "store", "ana", run_dir)
    tracker.start(record["config"])
    tracker.epoch(1, {"train_loss": 0.9})
    tracker.epoch(1, {"train_loss": 0.9})
    (run,) = fake_mlflow.runs.values()
    assert len(run.metrics["loss/train"]) == 1


def test_changed_value_for_a_logged_epoch_fails_the_export(
    tmp_path: Path, fake_mlflow: type[FakeMlflowClient]
) -> None:
    run_dir, record = _run_dir(tmp_path)
    tracker = MlflowTracker(tmp_path / "store", "ana", run_dir)
    tracker.start(record["config"])
    tracker.epoch(1, {"train_loss": 0.9})
    tracker.epoch(1, {"train_loss": 0.5})
    tracker.finish("FINISHED")
    saved = _sidecar(run_dir)
    assert saved["state"] == "failed"
    assert saved["error"] == "ValueError"
    (run,) = fake_mlflow.runs.values()
    assert run.status == "FINISHED"
    assert run.artifacts == []


def test_retry_reuses_the_saved_run_without_duplicates(
    tmp_path: Path, fake_mlflow: type[FakeMlflowClient]
) -> None:
    run_dir, record = _run_dir(tmp_path)
    store = tmp_path / "store"
    _train(MlflowTracker(store, "ana", run_dir), record, "FINISHED")
    MlflowTracker(store, "ana", run_dir).retry(record)

    (run,) = fake_mlflow.runs.values()
    assert len(run.metrics["loss/train"]) == 2
    assert run.artifacts == ["run.json", "run.json"]
    assert _sidecar(run_dir)["state"] == "finished"


def test_retry_after_a_lost_acknowledgement_finds_the_run_by_identity(
    tmp_path: Path, fake_mlflow: type[FakeMlflowClient]
) -> None:
    run_dir, record = _run_dir(tmp_path)
    store = tmp_path / "store"
    _train(MlflowTracker(store, "ana", run_dir), record, "KILLED")
    saved = _sidecar(run_dir)
    saved["run_id"] = None
    (run_dir / "tracking.json").write_text(json.dumps(saved))

    MlflowTracker(store, "ana", run_dir).retry(record)

    assert len(fake_mlflow.runs) == 1
    (run,) = fake_mlflow.runs.values()
    assert run.status == "KILLED"
    assert _sidecar(run_dir)["run_id"] == run.run_id


def test_saved_run_of_another_profile_is_refused(
    tmp_path: Path, fake_mlflow: type[FakeMlflowClient]
) -> None:
    store = tmp_path / "store"
    other_dir, other_record = _run_dir(tmp_path, "other")
    _train(MlflowTracker(store, "bia", other_dir), other_record, "FINISHED")
    run_dir, record = _run_dir(tmp_path)
    _train(MlflowTracker(store, "ana", run_dir), record, "FINISHED")
    saved = _sidecar(run_dir)
    saved["run_id"] = _sidecar(other_dir)["run_id"]
    (run_dir / "tracking.json").write_text(json.dumps(saved))

    MlflowTracker(store, "ana", run_dir).retry(record)

    assert _sidecar(run_dir)["state"] == "failed"
    other_run = fake_mlflow.runs[_sidecar(other_dir)["run_id"]]
    assert other_run.artifacts == ["run.json"]


def test_two_external_runs_with_one_identity_are_ambiguous(
    tmp_path: Path, fake_mlflow: type[FakeMlflowClient]
) -> None:
    run_dir, record = _run_dir(tmp_path)
    tracker = MlflowTracker(tmp_path / "store", "ana", run_dir)
    tracker.start(record["config"])
    (run,) = fake_mlflow.runs.values()
    fake_mlflow.runs["dup"] = FakeRun("dup", run.experiment_id, dict(run.tags))
    (run_dir / "tracking.json").unlink()

    MlflowTracker(tmp_path / "store", "ana", run_dir).start(record["config"])

    assert _sidecar(run_dir)["error"] == "ValueError"


def test_invalid_epoch_and_status_are_refused(
    tmp_path: Path, fake_mlflow: type[FakeMlflowClient]
) -> None:
    run_dir, record = _run_dir(tmp_path)
    tracker = MlflowTracker(tmp_path / "store", "ana", run_dir)
    tracker.start(record["config"])
    tracker.epoch(-1, {"train_loss": 0.1})
    assert _sidecar(run_dir)["error"] == "ValueError"

    other_dir, _ = _run_dir(tmp_path, "other")
    other = MlflowTracker(tmp_path / "store", "ana", other_dir)
    other.start({})
    other.finish("DONE")
    assert _sidecar(other_dir)["error"] == "ValueError"
