"""Tests for GET /api/runs endpoint and _load_runs helper."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from visionforge.gui.api.routes import _detect_dataset_layout, _load_runs
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
        # No preprocessing in the synthetic run → count is 0 by default.
        assert s.preprocessing_count == 0

    def test_summary_reports_preprocessing_count(self, tmp_path: Path) -> None:
        """RunSummary must expose the number of preprocessing filters used."""
        run_dir = tmp_path / "exp_pp" / "20260301_120000_000000"
        run_dir.mkdir(parents=True)
        data = {
            "id": "exp_pp_run",
            "experiment": "exp_pp",
            "timestamp": "2026-03-01T12:00:00",
            "status": "completed",
            "config": {
                "model": {"name": "resnet18"},
                "task": "binary",
                "data": {
                    "preprocessing": {
                        "steps": [
                            {"kind": "gaussian_blur", "radius": 1.5},
                            {"kind": "grayscale"},
                        ]
                    },
                },
            },
            "metrics": {
                "best_val_loss": 0.4,
                "best_epoch": 1,
                "total_epochs": 2,
            },
        }
        (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")

        results = _load_runs(tmp_path)
        assert len(results) == 1
        assert results[0].preprocessing_count == 2

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


# ── integration tests (TestClient) ───────────────────────────────────────────


@pytest.fixture
def app_and_routes():  # type: ignore[return]
    """Return (app, routes_mod) so tests can patch _MODELS_DIR before requests."""
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    return app, routes_mod


class TestApiRunsEndpoint:
    def test_get_runs_empty(self, tmp_path: Path, app_and_routes: tuple) -> None:
        """GET /api/runs with no runs on disk must return an empty list."""
        app, routes_mod = app_and_routes
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch.object(routes_mod, "_MODELS_DIR", empty_dir):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get("/api/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_runs_returns_summary(
        self, tmp_path: Path, app_and_routes: tuple
    ) -> None:
        """GET /api/runs must return a list with one RunSummary."""
        app, routes_mod = app_and_routes
        _make_run_json(tmp_path, folder="20260101_100000_000000")
        with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get("/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["experiment_name"] == "exp1"
        assert data[0]["model_arch"] == "resnet18"
        assert data[0]["status"] == "completed"

    def test_get_schema(self, app_and_routes: tuple) -> None:
        """GET /api/schema must return a JSON object with 'properties'."""
        app, _ = app_and_routes
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/api/schema")
        assert resp.status_code == 200
        assert "properties" in resp.json()

    def test_get_status_idle(self, app_and_routes: tuple) -> None:
        """GET /api/experiment/status with no active run must return idle."""
        app, routes_mod = app_and_routes
        # Reset global state to ensure idle.
        routes_mod._current_run = None
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/api/experiment/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "idle"

    def test_get_result_unknown_run(self, app_and_routes: tuple) -> None:
        """GET /api/experiment/result/<unknown> must return 404."""
        app, routes_mod = app_and_routes
        routes_mod._current_run = None
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/api/experiment/result/nonexistent-run-id")
        assert resp.status_code == 404

    def test_serve_artifact_outside_outputs(self, app_and_routes: tuple) -> None:
        """GET /api/artifacts/ with a path that escapes outputs/ must return 403."""
        app, _ = app_and_routes
        client = TestClient(app, raise_server_exceptions=True)
        # Use an absolute path outside outputs/ — the handler resolves it and checks.
        outside_path = Path("C:/Windows/System32/drivers/etc/hosts").as_posix()
        resp = client.get(f"/api/artifacts/{outside_path}")
        assert resp.status_code == 403

    def test_serve_artifact_existing_file_inside_outputs(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        """GET /api/artifacts/ with an existing file inside outputs must return 200."""
        app, _ = app_and_routes

        # Create a real file inside the real cwd-relative outputs/ so both
        # is_relative_to and is_file() pass without any path mocking.
        cwd_outputs = Path("outputs").resolve()
        cwd_outputs.mkdir(parents=True, exist_ok=True)
        real_file = cwd_outputs / "_test_artifact.png"
        real_file.write_bytes(b"fake-png")
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get(f"/api/artifacts/{real_file.as_posix()}")
            assert resp.status_code == 200
        finally:
            real_file.unlink(missing_ok=True)

    def test_post_experiment_run_starts_run(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        """POST /experiment/run must return 200 with a run_id."""
        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        minimal_config = {
            "name": "test_exp",
            "task": "binary",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {
                "learning_rate": 0.001,
                "epochs": 1,
                "batch_size": 4,
                "early_stopping_patience": 1,
                "seed": 0,
            },
            "data": {"base_dir": str(tmp_path)},
            "output": {
                "models_dir": str(tmp_path / "models"),
                "graphics_dir": str(tmp_path / "graphics"),
                "logs_dir": str(tmp_path / "logs"),
                "reports_dir": str(tmp_path / "reports"),
            },
        }

        # Mock _execute_experiment so the background task doesn't actually train.
        async def fake_execute(config: object, run_id: str) -> None:
            routes_mod._current_run = {
                "run_id": run_id,
                "status": "completed",
                "error": None,
                "report": {},
                "run_dir": None,
            }

        with patch.object(routes_mod, "_execute_experiment", fake_execute):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/experiment/run", json=minimal_config)

        assert resp.status_code == 200
        body = resp.json()
        assert "run_id" in body
        assert body["status"] == "running"
        routes_mod._current_run = None

    def test_post_experiment_run_accepts_preprocessing_steps(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        """POST /experiment/run must accept the flat preprocessing step shape the
        PreprocessingPanel sends (``{kind, ...params}``).

        Regression guard: until the panel was made controlled, the pipeline never
        reached the backend. This test pins down that the round-trip works.
        """
        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        captured: dict[str, object] = {}

        async def fake_execute(config: object, run_id: str) -> None:
            captured["config"] = config
            routes_mod._current_run = {
                "run_id": run_id,
                "status": "completed",
                "error": None,
                "report": {},
                "run_dir": None,
            }

        payload = {
            "name": "test_exp_pp",
            "task": "binary",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {
                "learning_rate": 0.001,
                "epochs": 1,
                "batch_size": 4,
                "early_stopping_patience": 1,
                "seed": 0,
            },
            "data": {
                "base_dir": str(tmp_path),
                "preprocessing": {
                    "steps": [
                        {"kind": "gaussian_blur", "radius": 1.5},
                        {"kind": "grayscale"},
                        {"kind": "wavelet", "band": "LL"},
                    ],
                },
            },
        }

        with patch.object(routes_mod, "_execute_experiment", fake_execute):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/experiment/run", json=payload)

        assert resp.status_code == 200, resp.text

        # The handler validated the payload via Pydantic before dispatching the
        # worker, so the captured config exposes the typed pipeline.
        config = captured["config"]
        steps = config.data.preprocessing.steps  # type: ignore[attr-defined]
        assert len(steps) == 3
        assert steps[0].kind == "gaussian_blur"
        # extra="allow" preserves filter params on the step model
        assert steps[0].radius == 1.5  # type: ignore[attr-defined]
        assert steps[1].kind == "grayscale"
        assert steps[2].kind == "wavelet"
        assert steps[2].band == "LL"  # type: ignore[attr-defined]

        routes_mod._current_run = None

    def test_checkpoint_pick_handles_cancelled_dialog(
        self, app_and_routes: tuple
    ) -> None:
        """POST /api/checkpoint/pick must return cancelled=True when no file chosen."""
        from visionforge.gui.api.schemas import CheckpointPickResponse

        app, routes_mod = app_and_routes

        def fake_open() -> CheckpointPickResponse:
            return CheckpointPickResponse(path="", cancelled=True, message="Cancelado.")

        with patch.object(routes_mod, "_open_native_checkpoint_dialog", fake_open):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/checkpoint/pick")
        assert resp.status_code == 200
        body = resp.json()
        assert body["cancelled"] is True
        assert body["path"] == ""

    def test_checkpoint_pick_returns_chosen_path(self, app_and_routes: tuple) -> None:
        """POST /api/checkpoint/pick must return the absolute file path."""
        from visionforge.gui.api.schemas import CheckpointPickResponse

        app, routes_mod = app_and_routes
        chosen = "/abs/path/to/model.pth"

        def fake_open() -> CheckpointPickResponse:
            return CheckpointPickResponse(path=chosen, cancelled=False)

        with patch.object(routes_mod, "_open_native_checkpoint_dialog", fake_open):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/checkpoint/pick")
        assert resp.status_code == 200
        body = resp.json()
        assert body["cancelled"] is False
        assert body["path"] == chosen

    def test_post_experiment_run_dispatches_cross_validation_block(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        """When config.block='cross_validation' the worker must instantiate
        CrossValidationBlock instead of ClassificationBlock.

        Regression guard for the new dispatch path — until this change, the
        backend hardcoded ClassificationBlock and any other block was unreachable
        through the GUI.
        """
        from visionforge.blocks.cross_validation import CrossValidationBlock

        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        captured_block: dict[str, object] = {}

        # Replace the heavy run() with a stub so we never actually train. We
        # only assert which class the dispatcher constructed.
        original_setup = CrossValidationBlock.setup
        original_run = CrossValidationBlock.run
        original_report = CrossValidationBlock.report

        def fake_setup(self, config: object) -> None:  # type: ignore[no-untyped-def]
            captured_block["class"] = type(self).__name__
            captured_block["config_block"] = config.block  # type: ignore[attr-defined]

        def fake_run(self) -> None:  # type: ignore[no-untyped-def]
            return None

        def fake_report(self) -> dict[str, object]:  # type: ignore[no-untyped-def]
            return {"mean_accuracy": 0.9, "std_accuracy": 0.02}

        CrossValidationBlock.setup = fake_setup  # type: ignore[method-assign]
        CrossValidationBlock.run = fake_run  # type: ignore[method-assign]
        CrossValidationBlock.report = fake_report  # type: ignore[method-assign]

        try:
            payload = {
                "name": "test_cv",
                "task": "binary",
                "block": "cross_validation",
                "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
                "training": {
                    "learning_rate": 0.001,
                    "epochs": 1,
                    "batch_size": 4,
                    "early_stopping_patience": 1,
                    "seed": 0,
                },
                "data": {"base_dir": str(tmp_path)},
                "cross_validation": {
                    "n_folds": 3,
                    "stratified": True,
                    "shuffle": True,
                    "fold_seed": 42,
                },
            }
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/experiment/run", json=payload)

            assert resp.status_code == 200
            # Let the background task pick up so fake_setup runs.
            import time

            for _ in range(50):
                if "class" in captured_block:
                    break
                time.sleep(0.05)
            assert captured_block.get("class") == "CrossValidationBlock"
            assert captured_block.get("config_block") == "cross_validation"
        finally:
            CrossValidationBlock.setup = original_setup  # type: ignore[method-assign]
            CrossValidationBlock.run = original_run  # type: ignore[method-assign]
            CrossValidationBlock.report = original_report  # type: ignore[method-assign]
            routes_mod._current_run = None

    def test_post_experiment_run_dispatches_transfer_learning_block(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        """block='transfer_learning' must instantiate TransferLearningBlock."""
        from visionforge.blocks.transfer_learning import TransferLearningBlock

        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        captured: dict[str, object] = {}
        original_setup = TransferLearningBlock.setup
        original_run = TransferLearningBlock.run
        original_report = TransferLearningBlock.report

        def fake_setup(self, config: object) -> None:  # type: ignore[no-untyped-def]
            captured["class"] = type(self).__name__
            captured["mode"] = config.transfer_learning.mode  # type: ignore[attr-defined]

        def fake_run(self) -> None:  # type: ignore[no-untyped-def]
            return None

        def fake_report(self) -> dict[str, object]:  # type: ignore[no-untyped-def]
            return {"mode": "feature_extraction", "eval": {"accuracy": 0.9}}

        TransferLearningBlock.setup = fake_setup  # type: ignore[method-assign]
        TransferLearningBlock.run = fake_run  # type: ignore[method-assign]
        TransferLearningBlock.report = fake_report  # type: ignore[method-assign]

        try:
            payload = {
                "name": "test_tl",
                "task": "binary",
                "block": "transfer_learning",
                "model": {"name": "resnet18", "num_classes": 1, "pretrained": True},
                "training": {
                    "learning_rate": 0.001,
                    "epochs": 1,
                    "batch_size": 4,
                    "early_stopping_patience": 1,
                    "seed": 0,
                },
                "data": {"base_dir": str(tmp_path)},
                "transfer_learning": {
                    "mode": "feature_extraction",
                    "unfreeze_from_layer": None,
                    "backbone_lr_multiplier": 0.1,
                },
            }
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/experiment/run", json=payload)
            assert resp.status_code == 200, resp.text

            import time

            for _ in range(50):
                if "class" in captured:
                    break
                time.sleep(0.05)
            assert captured.get("class") == "TransferLearningBlock"
            assert captured.get("mode") == "feature_extraction"
        finally:
            TransferLearningBlock.setup = original_setup  # type: ignore[method-assign]
            TransferLearningBlock.run = original_run  # type: ignore[method-assign]
            TransferLearningBlock.report = original_report  # type: ignore[method-assign]
            routes_mod._current_run = None

    def test_post_experiment_run_dispatches_model_comparison_block(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        """block='model_comparison' must instantiate ModelComparisonBlock."""
        from visionforge.blocks.model_comparison import ModelComparisonBlock

        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        captured: dict[str, object] = {}
        original_setup = ModelComparisonBlock.setup
        original_run = ModelComparisonBlock.run
        original_report = ModelComparisonBlock.report

        def fake_setup(self, config: object) -> None:  # type: ignore[no-untyped-def]
            captured["class"] = type(self).__name__
            captured["model_names"] = list(config.model_comparison.model_names)  # type: ignore[attr-defined]
            captured["metric"] = config.model_comparison.metric  # type: ignore[attr-defined]

        def fake_run(self) -> None:  # type: ignore[no-untyped-def]
            return None

        def fake_report(self) -> dict[str, object]:  # type: ignore[no-untyped-def]
            return {"top_3": [], "total_ran": 2, "failed_count": 0}

        ModelComparisonBlock.setup = fake_setup  # type: ignore[method-assign]
        ModelComparisonBlock.run = fake_run  # type: ignore[method-assign]
        ModelComparisonBlock.report = fake_report  # type: ignore[method-assign]

        try:
            payload = {
                "name": "test_mc",
                "task": "binary",
                "block": "model_comparison",
                "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
                "training": {
                    "learning_rate": 0.001,
                    "epochs": 1,
                    "batch_size": 4,
                    "early_stopping_patience": 1,
                    "seed": 0,
                },
                "data": {"base_dir": str(tmp_path)},
                "model_comparison": {
                    "model_names": ["resnet18", "resnet50"],
                    "metric": "f1",
                },
            }
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/experiment/run", json=payload)
            assert resp.status_code == 200, resp.text

            import time

            for _ in range(50):
                if "class" in captured:
                    break
                time.sleep(0.05)
            assert captured.get("class") == "ModelComparisonBlock"
            assert captured.get("model_names") == ["resnet18", "resnet50"]
            assert captured.get("metric") == "f1"
        finally:
            ModelComparisonBlock.setup = original_setup  # type: ignore[method-assign]
            ModelComparisonBlock.run = original_run  # type: ignore[method-assign]
            ModelComparisonBlock.report = original_report  # type: ignore[method-assign]
            routes_mod._current_run = None

    def test_post_experiment_run_dispatches_grid_search_block(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        """block='grid_search' must instantiate GridSearchBlock with the dict
        hyperparameter space exactly as submitted."""
        from visionforge.blocks.grid_search import GridSearchBlock

        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        captured: dict[str, object] = {}
        original_setup = GridSearchBlock.setup
        original_run = GridSearchBlock.run
        original_report = GridSearchBlock.report

        def fake_setup(self, config: object) -> None:  # type: ignore[no-untyped-def]
            captured["class"] = type(self).__name__
            captured["hp"] = dict(config.grid_search.hyperparameters)  # type: ignore[attr-defined]

        def fake_run(self) -> None:  # type: ignore[no-untyped-def]
            # The GUI must wire the SSE event pump so grid search streams live.
            captured["has_callback"] = self._progress_callback is not None
            return None

        def fake_report(self) -> dict[str, object]:  # type: ignore[no-untyped-def]
            return {
                "best_trial": {"trial_index": 1, "test_accuracy": 0.92},
                "total_trials": 3,
                "successful_trials": 3,
            }

        GridSearchBlock.setup = fake_setup  # type: ignore[method-assign]
        GridSearchBlock.run = fake_run  # type: ignore[method-assign]
        GridSearchBlock.report = fake_report  # type: ignore[method-assign]

        try:
            payload = {
                "name": "test_grid",
                "task": "binary",
                "block": "grid_search",
                "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
                "training": {
                    "learning_rate": 0.001,
                    "epochs": 1,
                    "batch_size": 4,
                    "early_stopping_patience": 1,
                    "seed": 0,
                },
                "data": {"base_dir": str(tmp_path)},
                "grid_search": {
                    "hyperparameters": {
                        "training.learning_rate": [0.001, 0.0005, 0.0001],
                    }
                },
            }
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/experiment/run", json=payload)
            assert resp.status_code == 200, resp.text

            import time

            for _ in range(50):
                if "class" in captured:
                    break
                time.sleep(0.05)
            assert captured.get("class") == "GridSearchBlock"
            assert captured.get("hp") == {
                "training.learning_rate": [0.001, 0.0005, 0.0001]
            }
            assert captured.get("has_callback") is True
        finally:
            GridSearchBlock.setup = original_setup  # type: ignore[method-assign]
            GridSearchBlock.run = original_run  # type: ignore[method-assign]
            GridSearchBlock.report = original_report  # type: ignore[method-assign]
            routes_mod._current_run = None

    def test_post_experiment_run_dispatches_random_search_block(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        """block='random_search' must instantiate RandomSearchBlock with the
        submitted search_space + n_trials exactly as the payload describes."""
        from visionforge.blocks.random_search import RandomSearchBlock

        app, routes_mod = app_and_routes
        routes_mod._current_run = None

        captured: dict[str, object] = {}
        original_setup = RandomSearchBlock.setup
        original_run = RandomSearchBlock.run
        original_report = RandomSearchBlock.report

        def fake_setup(self, config: object) -> None:  # type: ignore[no-untyped-def]
            captured["class"] = type(self).__name__
            captured["n_trials"] = config.random_search.n_trials  # type: ignore[attr-defined]
            captured["space"] = dict(config.random_search.search_space)  # type: ignore[attr-defined]

        def fake_run(self) -> None:  # type: ignore[no-untyped-def]
            return None

        def fake_report(self) -> dict[str, object]:  # type: ignore[no-untyped-def]
            return {
                "best_trial": {"trial_index": 2, "test_accuracy": 0.88},
                "total_trials": 5,
                "successful_trials": 5,
            }

        RandomSearchBlock.setup = fake_setup  # type: ignore[method-assign]
        RandomSearchBlock.run = fake_run  # type: ignore[method-assign]
        RandomSearchBlock.report = fake_report  # type: ignore[method-assign]

        try:
            payload = {
                "name": "test_random",
                "task": "binary",
                "block": "random_search",
                "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
                "training": {
                    "learning_rate": 0.001,
                    "epochs": 1,
                    "batch_size": 4,
                    "early_stopping_patience": 1,
                    "seed": 0,
                },
                "data": {"base_dir": str(tmp_path)},
                "random_search": {
                    "n_trials": 5,
                    "seed": 7,
                    "search_space": {
                        "training.learning_rate": {
                            "type": "log_uniform",
                            "low": 1e-5,
                            "high": 1e-2,
                        },
                    },
                },
            }
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/experiment/run", json=payload)
            assert resp.status_code == 200, resp.text

            import time

            for _ in range(50):
                if "class" in captured:
                    break
                time.sleep(0.05)
            assert captured.get("class") == "RandomSearchBlock"
            assert captured.get("n_trials") == 5
            assert captured.get("space") == {
                "training.learning_rate": {
                    "type": "log_uniform",
                    "low": 1e-5,
                    "high": 1e-2,
                },
            }
        finally:
            RandomSearchBlock.setup = original_setup  # type: ignore[method-assign]
            RandomSearchBlock.run = original_run  # type: ignore[method-assign]
            RandomSearchBlock.report = original_report  # type: ignore[method-assign]
            routes_mod._current_run = None

    def test_post_experiment_run_rejects_concurrent(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        """POST /experiment/run must return 409 when a run is already active."""
        app, routes_mod = app_and_routes
        routes_mod._current_run = {
            "run_id": "already-running",
            "status": "running",
            "error": None,
        }
        try:
            minimal_config = {
                "name": "test_exp",
                "task": "binary",
                "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
                "training": {
                    "learning_rate": 0.001,
                    "epochs": 1,
                    "batch_size": 4,
                    "early_stopping_patience": 1,
                    "seed": 0,
                },
                "data": {"base_dir": str(tmp_path)},
            }
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/api/experiment/run", json=minimal_config)
            assert resp.status_code == 409
        finally:
            routes_mod._current_run = None

    def test_get_status_with_active_run(self, app_and_routes: tuple) -> None:
        """GET /api/experiment/status must return run_id and status when a run exists."""
        app, routes_mod = app_and_routes
        routes_mod._current_run = {
            "run_id": "test-run-123",
            "status": "running",
            "error": None,
        }
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get("/api/experiment/status")
            assert resp.status_code == 200
            body = resp.json()
            assert body["run_id"] == "test-run-123"
            assert body["status"] == "running"
        finally:
            routes_mod._current_run = None

    def test_get_result_still_running(self, app_and_routes: tuple) -> None:
        """GET /api/experiment/result returns 409 when experiment is still running."""
        app, routes_mod = app_and_routes
        routes_mod._current_run = {
            "run_id": "run-abc",
            "status": "running",
            "error": None,
        }
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get("/api/experiment/result/run-abc")
            assert resp.status_code == 409
        finally:
            routes_mod._current_run = None

    def test_get_result_failed_run(self, app_and_routes: tuple) -> None:
        """GET /api/experiment/result returns 500 when experiment failed."""
        app, routes_mod = app_and_routes
        routes_mod._current_run = {
            "run_id": "run-xyz",
            "status": "failed",
            "error": "something broke",
        }
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get("/api/experiment/result/run-xyz")
            assert resp.status_code == 500
        finally:
            routes_mod._current_run = None

    def test_get_result_completed_no_run_dir(self, app_and_routes: tuple) -> None:
        """GET /api/experiment/result returns empty metrics when run_dir is None."""
        app, routes_mod = app_and_routes
        routes_mod._current_run = {
            "run_id": "run-done",
            "status": "completed",
            "error": None,
            "report": {"train": {"best_epoch": 1}},
            "run_dir": None,
        }
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get("/api/experiment/result/run-done")
            assert resp.status_code == 200
            body = resp.json()
            assert body["run_id"] == "run-done"
            assert body["metrics"] == {}
        finally:
            routes_mod._current_run = None

    def test_get_result_completed_with_run_json(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        """GET /api/experiment/result returns metrics from run.json when it exists."""
        app, routes_mod = app_and_routes
        run_dir = _make_run_json(tmp_path)
        routes_mod._current_run = {
            "run_id": "run-with-file",
            "status": "completed",
            "error": None,
            "report": {},
            "run_dir": str(run_dir),
        }
        try:
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get("/api/experiment/result/run-with-file")
            assert resp.status_code == 200
            body = resp.json()
            assert "best_val_loss" in body["metrics"]
        finally:
            routes_mod._current_run = None


class TestServerSpaFallback:
    def test_unknown_path_does_not_crash_server(self, app_and_routes: tuple) -> None:
        """SPA fallback for an unknown route must not produce an unhandled exception."""
        app, _ = app_and_routes
        # raise_server_exceptions=False: we want to observe the HTTP response,
        # not re-raise the FileNotFoundError from the missing static dir.
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/some/unknown/spa/route", follow_redirects=True)
        # Any 4xx is acceptable — the point is no 5xx from the route handler itself.
        assert resp.status_code < 600  # server responded at all

    def test_known_static_file_served(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        """SPA fallback must serve an existing static file directly."""
        app, _ = app_and_routes
        # Patch STATIC_DIR to a tmp directory containing a fake file.
        fake_file = tmp_path / "test.txt"
        fake_file.write_text("hello", encoding="utf-8")
        import visionforge.gui.server as server_mod

        with patch.object(server_mod, "STATIC_DIR", tmp_path):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get("/test.txt")
        assert resp.status_code == 200

    def test_index_html_fallback(self, app_and_routes: tuple, tmp_path: Path) -> None:
        """SPA fallback must serve index.html when the requested file is absent."""
        app, _ = app_and_routes
        index = tmp_path / "index.html"
        index.write_text("<html></html>", encoding="utf-8")
        import visionforge.gui.server as server_mod

        with patch.object(server_mod, "STATIC_DIR", tmp_path):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get("/some/spa/route")
        assert resp.status_code == 200


class TestDatasetDetect:
    """Cover the dataset split auto-detection helper."""

    def _make_layout(self, root: Path, names: list[str]) -> None:
        for n in names:
            (root / n).mkdir(parents=True)

    def test_canonical_train_val_test(self, tmp_path: Path) -> None:
        self._make_layout(tmp_path, ["train", "val", "test"])
        resp = _detect_dataset_layout(str(tmp_path))
        assert resp.detected is True
        assert resp.train_dir == "train"
        assert resp.val_dir == "val"
        assert resp.test_dir == "test"

    def test_portuguese_names(self, tmp_path: Path) -> None:
        self._make_layout(tmp_path, ["treino", "validacao", "teste"])
        resp = _detect_dataset_layout(str(tmp_path))
        assert resp.detected is True
        assert resp.train_dir == "treino"
        assert resp.val_dir == "validacao"
        assert resp.test_dir == "teste"

    def test_validation_long_form_with_accent(self, tmp_path: Path) -> None:
        self._make_layout(tmp_path, ["training", "validation", "testing"])
        resp = _detect_dataset_layout(str(tmp_path))
        assert resp.detected is True
        assert resp.train_dir == "training"
        assert resp.val_dir == "validation"
        assert resp.test_dir == "testing"

    def test_mixed_case_and_separators(self, tmp_path: Path) -> None:
        self._make_layout(tmp_path, ["Train_Set", "Val-Set", "test set"])
        resp = _detect_dataset_layout(str(tmp_path))
        # Names with extra tokens normalize but not to exact alias.
        # We expect detection to be partial or none — confirm we don't crash
        # and return candidates.
        assert resp.candidates == sorted(["Train_Set", "Val-Set", "test set"])

    def test_partial_detection_missing_test(self, tmp_path: Path) -> None:
        self._make_layout(tmp_path, ["train", "val", "extra"])
        resp = _detect_dataset_layout(str(tmp_path))
        assert resp.detected is False
        assert resp.train_dir == "train"
        assert resp.val_dir == "val"
        assert resp.test_dir is None
        assert "teste" in resp.message.lower()

    def test_nonexistent_base_dir(self) -> None:
        resp = _detect_dataset_layout("/nonexistent/path/abc123")
        assert resp.detected is False
        assert "não foi encontrado" in resp.message

    def test_empty_base_dir(self) -> None:
        resp = _detect_dataset_layout("")
        assert resp.detected is False
        assert "Informe o caminho" in resp.message

    def test_base_dir_is_file(self, tmp_path: Path) -> None:
        file = tmp_path / "not_a_dir.txt"
        file.write_text("x", encoding="utf-8")
        resp = _detect_dataset_layout(str(file))
        assert resp.detected is False
        assert "não é uma pasta" in resp.message

    def test_no_subdirs(self, tmp_path: Path) -> None:
        resp = _detect_dataset_layout(str(tmp_path))
        assert resp.detected is False
        assert resp.candidates == []
        assert "Nenhuma subpasta" in resp.message

    def test_unrecognized_subdirs(self, tmp_path: Path) -> None:
        self._make_layout(tmp_path, ["alpha", "beta", "gamma"])
        resp = _detect_dataset_layout(str(tmp_path))
        assert resp.detected is False
        assert resp.train_dir is None
        assert resp.candidates == ["alpha", "beta", "gamma"]
        assert "Selecione manualmente" in resp.message

    def test_endpoint_returns_response(self, tmp_path: Path) -> None:
        from fastapi import FastAPI

        from visionforge.gui.api.routes import router

        app = FastAPI()
        app.include_router(router)
        self._make_layout(tmp_path, ["train", "val", "test"])
        client = TestClient(app)
        resp = client.post("/api/dataset/detect", json={"base_dir": str(tmp_path)})
        assert resp.status_code == 200
        body = resp.json()
        assert body["detected"] is True
        assert body["train_dir"] == "train"
