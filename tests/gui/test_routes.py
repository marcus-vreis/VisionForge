"""Tests for GET /api/runs endpoint and _load_runs helper."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from visionforge.gui.api.routes import _load_runs
from visionforge.gui.api.schemas import RunSummary

# ── fixtures ──────────────────────────────────────────────────────────────────

_TS_EARLY = "2026-01-01T10:00:00"
_TS_LATE = "2026-06-01T10:00:00"


def _make_run_json(
    tmp_path: Path,
    *,
    folder: str = "20260101_100000_000000",
    experiment: str = "exp1",
    model_name: str = "resnet18",
    task: str = "binary",
    status: str = "completed",
    timestamp: str = _TS_EARLY,
    total_epochs: int = 5,
    best_val_loss: float = 0.25,
    test_accuracy: float | None = 0.92,
    test_f1: float | None = 0.91,
) -> Path:
    """Write a synthetic run.json into tmp_path/experiment/folder/ and return the dir."""
    run_dir = tmp_path / experiment / folder
    run_dir.mkdir(parents=True)

    metrics: dict[str, object] = {
        "best_val_loss": best_val_loss,
        "best_epoch": 3,
        "total_epochs": total_epochs,
    }
    if test_accuracy is not None:
        metrics["test_accuracy"] = test_accuracy
    if test_f1 is not None:
        metrics["test_f1"] = test_f1

    data = {
        "id": f"{experiment}_{folder}",
        "experiment": experiment,
        "timestamp": timestamp,
        "status": status,
        "config": {
            "model": {"name": model_name},
            "task": task,
        },
        "metrics": metrics,
        "history": [],
        "artifacts": {},
    }
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
    return run_dir


# ── tests ─────────────────────────────────────────────────────────────────────


class TestLoadRunsEmpty:
    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        """_load_runs on a missing directory must return an empty list."""
        result = _load_runs(tmp_path / "does_not_exist")
        assert result == []

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        """_load_runs on an existing but empty directory must return []."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        assert _load_runs(models_dir) == []


class TestLoadRunsSingleRun:
    def test_single_completed_run(self, tmp_path: Path) -> None:
        """A single valid run.json must be parsed into a RunSummary."""
        _make_run_json(tmp_path, folder="20260101_100000_000000")
        results = _load_runs(tmp_path)

        assert len(results) == 1
        s = results[0]
        assert isinstance(s, RunSummary)
        assert s.run_id == "20260101_100000_000000"
        assert s.experiment_name == "exp1"
        assert s.model_arch == "resnet18"
        assert s.task == "binary"
        assert s.status == "completed"
        assert s.epochs_completed == 5
        assert s.started_at == datetime.fromisoformat(_TS_EARLY)

    def test_completed_run_has_finished_at(self, tmp_path: Path) -> None:
        """A completed run must have finished_at equal to started_at."""
        _make_run_json(tmp_path, status="completed")
        s = _load_runs(tmp_path)[0]
        assert s.finished_at == s.started_at

    def test_running_status_has_no_finished_at(self, tmp_path: Path) -> None:
        """A run.json with status 'running' must yield finished_at=None."""
        _make_run_json(tmp_path, status="running")
        s = _load_runs(tmp_path)[0]
        assert s.finished_at is None

    def test_failed_status_has_no_finished_at(self, tmp_path: Path) -> None:
        """A run.json with status 'failed' must yield finished_at=None."""
        _make_run_json(tmp_path, status="failed")
        s = _load_runs(tmp_path)[0]
        assert s.finished_at is None


class TestLoadRunsFinalMetrics:
    def test_all_three_metrics_present(self, tmp_path: Path) -> None:
        """final_metrics must expose accuracy, f1, and val_loss when all are in run.json."""
        _make_run_json(
            tmp_path,
            best_val_loss=0.3,
            test_accuracy=0.85,
            test_f1=0.84,
        )
        s = _load_runs(tmp_path)[0]
        assert s.final_metrics == pytest.approx(
            {"accuracy": 0.85, "f1": 0.84, "val_loss": 0.3}
        )

    def test_only_val_loss_when_no_test_metrics(self, tmp_path: Path) -> None:
        """final_metrics must only include present keys — no KeyError for absent ones."""
        _make_run_json(
            tmp_path,
            best_val_loss=0.5,
            test_accuracy=None,
            test_f1=None,
        )
        s = _load_runs(tmp_path)[0]
        assert s.final_metrics == pytest.approx({"val_loss": 0.5})

    def test_metric_values_are_floats(self, tmp_path: Path) -> None:
        """final_metrics values must be float, not int."""
        _make_run_json(tmp_path, best_val_loss=1, test_accuracy=1, test_f1=0)
        s = _load_runs(tmp_path)[0]
        for v in s.final_metrics.values():
            assert isinstance(v, float)


class TestLoadRunsOrdering:
    def test_sorted_by_started_at_descending(self, tmp_path: Path) -> None:
        """_load_runs must return runs sorted newest-first."""
        _make_run_json(
            tmp_path,
            folder="20260101_100000_000000",
            timestamp=_TS_EARLY,
        )
        _make_run_json(
            tmp_path,
            folder="20260601_100000_000000",
            timestamp=_TS_LATE,
        )
        results = _load_runs(tmp_path)
        assert len(results) == 2
        assert results[0].started_at > results[1].started_at

    def test_order_is_stable_for_equal_timestamps(self, tmp_path: Path) -> None:
        """Two runs with the same timestamp must still both be returned."""
        _make_run_json(tmp_path, folder="run_a", timestamp=_TS_EARLY, experiment="exp1")
        _make_run_json(tmp_path, folder="run_b", timestamp=_TS_EARLY, experiment="exp2")
        results = _load_runs(tmp_path)
        assert len(results) == 2


class TestLoadRunsSkipBadEntries:
    def test_skips_invalid_json(self, tmp_path: Path) -> None:
        """A directory with malformed JSON must be silently skipped."""
        bad_dir = tmp_path / "exp_bad" / "20260101_000000_000000"
        bad_dir.mkdir(parents=True)
        (bad_dir / "run.json").write_text("{not valid json", encoding="utf-8")

        _make_run_json(tmp_path, folder="20260101_100000_000000", experiment="exp_good")
        results = _load_runs(tmp_path)

        assert len(results) == 1
        assert results[0].experiment_name == "exp_good"

    def test_skips_missing_required_field(self, tmp_path: Path) -> None:
        """A run.json missing a required field (e.g. 'experiment') must be skipped."""
        bad_dir = tmp_path / "exp_missing" / "20260101_000000_000001"
        bad_dir.mkdir(parents=True)
        # Missing 'experiment' key
        (bad_dir / "run.json").write_text(
            json.dumps(
                {
                    "id": "x",
                    "timestamp": _TS_EARLY,
                    "status": "completed",
                    "config": {"model": {"name": "resnet18"}, "task": "binary"},
                    "metrics": {"total_epochs": 1, "best_val_loss": 0.1},
                }
            ),
            encoding="utf-8",
        )

        _make_run_json(tmp_path, folder="20260101_100000_000000", experiment="exp_ok")
        results = _load_runs(tmp_path)

        assert len(results) == 1
        assert results[0].experiment_name == "exp_ok"

    def test_skips_missing_total_epochs(self, tmp_path: Path) -> None:
        """A run.json without metrics.total_epochs must be skipped."""
        bad_dir = tmp_path / "exp_no_epochs" / "20260101_000000_000002"
        bad_dir.mkdir(parents=True)
        (bad_dir / "run.json").write_text(
            json.dumps(
                {
                    "id": "x",
                    "experiment": "exp_no_epochs",
                    "timestamp": _TS_EARLY,
                    "status": "completed",
                    "config": {"model": {"name": "resnet18"}, "task": "binary"},
                    "metrics": {"best_val_loss": 0.1},
                }
            ),
            encoding="utf-8",
        )

        results = _load_runs(tmp_path)
        assert results == []

    def test_returns_empty_when_all_bad(self, tmp_path: Path) -> None:
        """_load_runs must return [] when every run.json is unparsable."""
        bad_dir = tmp_path / "exp_bad" / "20260101_000000_000000"
        bad_dir.mkdir(parents=True)
        (bad_dir / "run.json").write_text("", encoding="utf-8")
        assert _load_runs(tmp_path) == []
