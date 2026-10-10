"""The run detail says which way each of its metrics improves.

The rule used to live twice: ``infer_direction`` in the backend ranked sweeps and
replicated comparisons, and a copy of it in the History comparison highlighted
the best cell. The server now sends the answer it would have used, so the page
cannot disagree with a ranking -- and a researcher's task, whose declared
directions only the server can read, is believed on every screen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from visionforge.tasks.registry import TaskInfo

_RUN_ID = "20260522_100000_000000"


@pytest.fixture
def app_and_routes():  # type: ignore[return]
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    return app, routes_mod


def _write_run(
    models: Path,
    *,
    metrics: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    task: str | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    run_dir = models / "exp1" / _RUN_ID
    run_dir.mkdir(parents=True)
    data: dict[str, Any] = {
        "id": f"exp1_{_RUN_ID}",
        "experiment": "exp1",
        "timestamp": "2026-05-22T10:00:00",
        "status": "completed",
        "run_dir": str(run_dir.resolve()),
        "config": config or {},
        "metrics": metrics,
        "history": history or [],
        "artifacts": {},
        "tests": [],
    }
    if task is not None:
        data["task"] = task
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
    return run_dir


def _detail(app_and_routes: tuple, tmp_path: Path, **run: Any) -> dict[str, Any]:
    app, routes_mod = app_and_routes
    run_dir = _write_run(tmp_path, **run)
    with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
        resp = TestClient(app).get(f"/api/runs/{run_dir.name}")
    assert resp.status_code == 200
    body: dict[str, Any] = resp.json()
    return body


class TestBuiltInRunDirections:
    def test_every_metric_the_run_reports_gets_the_rule_of_the_ranking(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        body = _detail(
            app_and_routes,
            tmp_path,
            config={"task": "regression"},
            metrics={
                "best_val_loss": 0.4,
                "test_r2": 0.9,
                "test_mae": 2.0,
                "test_rmse": 3.0,
                "rmse": 3.1,
                "best_epoch": 4,
            },
        )

        assert body["metric_directions"] == {
            "best_val_loss": "lower",
            "test_r2": "higher",
            "test_mae": "lower",
            "test_rmse": "lower",
            "rmse": "lower",
            "best_epoch": "higher",
        }

    def test_the_history_series_are_covered_too(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        body = _detail(
            app_and_routes,
            tmp_path,
            metrics={"total_epochs": 2},
            history=[
                {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_accuracy": 0.5},
                {"epoch": 2, "train_loss": 0.8, "val_loss": 0.7, "val_accuracy": 0.6},
            ],
        )

        got = body["metric_directions"]
        assert got["train_loss"] == "lower"
        assert got["val_loss"] == "lower"
        assert got["val_accuracy"] == "higher"
        assert "epoch" not in got

    def test_it_is_the_same_function_the_orchestrators_rank_with(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        from visionforge.core.significance import infer_direction

        names = ["test_accuracy", "test_mse", "map50", "image_distance", "val_error"]
        body = _detail(app_and_routes, tmp_path, metrics=dict.fromkeys(names, 0.5))

        assert body["metric_directions"] == {n: infer_direction(n) for n in names}

    def test_a_run_with_nothing_measured_has_an_empty_map(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        body = _detail(app_and_routes, tmp_path, metrics={})

        assert body["metric_directions"] == {}


class TestCustomTaskDirections:
    @staticmethod
    def _task() -> TaskInfo:
        # "score" would be read as higher-is-better by its name; the task says
        # lower wins.
        return TaskInfo(
            key="shapes",
            label="Shapes",
            accent="#aabbcc",
            description="",
            metrics={"score": "lower", "iou": "higher"},
            primary_metric="score",
        )

    def test_a_declared_direction_wins_over_the_name(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        _, routes_mod = app_and_routes
        with patch.object(routes_mod, "get_task", lambda _key: self._task()):
            body = _detail(
                app_and_routes,
                tmp_path,
                task="custom:shapes",
                metrics={"score": 0.3, "iou": 0.7, "loss": 0.2, "total_epochs": 3},
                history=[
                    {"epoch": 1, "train_loss": 1.0, "val_score": 0.3, "val_iou": 0.7}
                ],
            )

        got = body["metric_directions"]
        assert got["score"] == "lower"
        assert got["iou"] == "higher"
        # The engine writes the history as val_<metric>: same declaration.
        assert got["val_score"] == "lower"
        assert got["val_iou"] == "higher"
        # What the task did not declare is judged by its name, as in the ranking.
        assert got["loss"] == "lower"
        assert got["train_loss"] == "lower"

    def test_a_task_that_is_no_longer_registered_falls_back_to_the_name(
        self, app_and_routes: tuple, tmp_path: Path
    ) -> None:
        _, routes_mod = app_and_routes

        def missing(key: str) -> TaskInfo:
            raise KeyError(key)

        with patch.object(routes_mod, "get_task", missing):
            body = _detail(
                app_and_routes,
                tmp_path,
                task="custom:gone",
                metrics={"score": 0.3, "error": 0.1},
            )

        assert body["metric_directions"] == {"score": "higher", "error": "lower"}
