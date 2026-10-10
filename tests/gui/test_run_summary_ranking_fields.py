"""What the History list carries so runs can be ranked per dataset.

The ranking groups runs by the data they saw, so the list must say which data
that was: the fingerprint's digest (and the method that produced it, since a
manifest digest and a content digest of the same files never match) where the
run has one, the dataset path otherwise. It also marks a run a user stop cut,
which is listed but never ranked, and states which way each headline metric
improves, so the page does not carry a second copy of the rule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from visionforge.tasks.registry import TaskInfo

_DIGEST = "ab" * 32


@pytest.fixture
def app_and_routes():  # type: ignore[return]
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    return app, routes_mod


def _write_run(models: Path, name: str, **extra: Any) -> None:
    run_dir = models / name / "20260522_100000_000000"
    run_dir.mkdir(parents=True)
    data: dict[str, Any] = {
        "id": f"{name}_20260522_100000_000000",
        "experiment": name,
        "timestamp": "2026-05-22T10:00:00",
        "status": "completed",
        "run_dir": str(run_dir.resolve()),
        "config": {"task": "multiclass", "data": {"base_dir": "C:/data/coffee"}},
        "metrics": {"total_epochs": 3, "test_accuracy": 0.9, "best_val_loss": 0.3},
        "history": [],
        "artifacts": {},
        "tests": [],
    }
    data.update(extra)
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")


def _listed(app_and_routes: tuple, tmp_path: Path) -> dict[str, dict[str, Any]]:
    app, routes_mod = app_and_routes
    with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
        resp = TestClient(app).get("/api/runs")
    assert resp.status_code == 200
    return {run["experiment_name"]: run for run in resp.json()}


class TestDatasetIdentityOnTheList:
    def test_a_fingerprinted_run_sends_its_digest_and_method(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        _write_run(
            tmp_path,
            "fp",
            dataset_fingerprint={
                "digest": _DIGEST,
                "method": "manifest",
                "root": "C:/data/coffee",
                "n_files": 10,
                "total_bytes": 100,
                "note": "paths+sizes only",
            },
        )

        run = _listed(app_and_routes, tmp_path)["fp"]

        assert run["dataset_digest"] == _DIGEST
        assert run["dataset_method"] == "manifest"
        assert run["dataset_root"] == "C:/data/coffee"

    def test_an_unavailable_fingerprint_proves_nothing_so_only_the_path_is_sent(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        _write_run(
            tmp_path,
            "gone",
            dataset_fingerprint={
                "digest": "",
                "method": "unavailable",
                "root": "C:/data/coffee",
                "note": "base_dir is not a directory",
            },
        )

        run = _listed(app_and_routes, tmp_path)["gone"]

        assert run["dataset_digest"] is None
        assert run["dataset_method"] is None
        assert run["dataset_root"] == "C:/data/coffee"

    def test_a_run_older_than_the_fingerprint_falls_back_to_the_config_path(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        _write_run(tmp_path, "old")

        run = _listed(app_and_routes, tmp_path)["old"]

        assert run["dataset_digest"] is None
        assert run["dataset_method"] is None
        assert run["dataset_root"] == "C:/data/coffee"

    def test_a_run_with_no_dataset_at_all_has_no_identity(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        _write_run(tmp_path, "none", config={"task": "multiclass"})

        run = _listed(app_and_routes, tmp_path)["none"]

        assert run["dataset_digest"] is None
        assert run["dataset_root"] is None


class TestStoppedMarker:
    def test_a_run_cut_by_a_stop_is_marked(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        _write_run(tmp_path, "cut", stopped=True)

        assert _listed(app_and_routes, tmp_path)["cut"]["stopped"] is True

    def test_a_run_that_never_wrote_the_marker_was_not_stopped(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        _write_run(tmp_path, "whole")

        assert _listed(app_and_routes, tmp_path)["whole"]["stopped"] is False


class TestMetricDirectionsOnTheList:
    def test_each_headline_metric_gets_the_rule_of_the_ranking(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        _write_run(tmp_path, "cls")

        run = _listed(app_and_routes, tmp_path)["cls"]

        assert run["final_metrics"] == {"accuracy": 0.9, "val_loss": 0.3}
        assert run["metric_directions"] == {"accuracy": "higher", "val_loss": "lower"}

    def test_a_validation_fallback_name_is_judged_like_its_test_twin(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        _write_run(
            tmp_path,
            "val",
            config={"task": "regression", "data": {"base_dir": "C:/data/x"}},
            metrics={"total_epochs": 3, "r2": 0.4, "mae": 3.0, "rmse": 4.0},
        )

        run = _listed(app_and_routes, tmp_path)["val"]

        assert run["metric_directions"] == {
            "val_r2": "higher",
            "val_mae": "lower",
            "val_rmse": "lower",
        }

    def test_a_declared_direction_of_a_researchers_task_wins_over_the_name(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        _, routes_mod = app_and_routes
        task = TaskInfo(
            key="shapes",
            label="Shapes",
            accent="#aabbcc",
            description="",
            metrics={"score": "lower", "iou": "higher"},
            primary_metric="score",
        )
        _write_run(
            tmp_path,
            "mine",
            task="custom:shapes",
            config={"data": {"base_dir": "C:/data/shapes"}},
            metrics={"total_epochs": 3, "score": 0.3, "iou": 0.7},
        )

        with patch.object(routes_mod, "get_task", lambda _key: task):
            run = _listed(app_and_routes, tmp_path)["mine"]

        assert run["metric_directions"] == {"score": "lower", "iou": "higher"}
