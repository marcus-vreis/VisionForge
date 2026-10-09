"""Every ranking puts the best unit first, whichever way the metric points.

`run_sweep` and `run_model_comparison` always sorted descending, so for a
lower-is-better metric -- `rmse`/`mae`/`loss` chosen in the API, or a custom
task that declares its metric "lower" -- the worst trial was named best, and so
were the report's `best_trial`, the ranking CSV's `rank` and the LaTeX table.
The direction comes from the task's declaration where there is one, else from
`infer_direction`, the rule the replicated comparison already used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from visionforge.core.comparison import run_model_comparison
from visionforge.core.replicated_comparison import run_replicated_comparison
from visionforge.core.sweep import run_sweep
from visionforge.core.task_runner import (
    RunResult,
    rank_by_metric,
    runner_metric_direction,
)


class _Echo:
    @staticmethod
    def model_validate(d: dict[str, Any]) -> dict[str, Any]:
        return d


class _ScoredRunner:
    """Each run reports ``metric`` from a table keyed by model name or lr."""

    config_type: Any = _Echo

    def __init__(
        self,
        metric: str,
        scores: dict[Any, float | None],
        declared: dict[str, str] | None = None,
    ) -> None:
        self.metric = metric
        self.scores = scores
        if declared is not None:
            self.metric_directions = declared

    def run(self, cfg: dict[str, Any]) -> RunResult:
        key = cfg["model"]["name"]
        if key not in self.scores:
            key = cfg["training"]["learning_rate"]
        value = self.scores[key]
        metrics = {} if value is None else {self.metric: value}
        return RunResult(metrics=metrics, status="success")

    def metrics(self, result: RunResult) -> dict[str, float]:
        return dict(result.metrics)

    def primary_metric(self) -> str:
        return self.metric


def _base() -> dict[str, Any]:
    return {
        "name": "x",
        "model": {"name": "a"},
        "training": {"learning_rate": 0.01, "seed": 1},
    }


class TestTheDirectionRule:
    def test_a_declared_direction_wins_over_the_name(self) -> None:
        runner = _ScoredRunner("score", {}, declared={"score": "lower"})
        assert runner_metric_direction(runner, "score") == "lower"

    def test_without_a_declaration_the_name_decides(self) -> None:
        runner = _ScoredRunner("rmse", {})
        assert runner_metric_direction(runner, "rmse") == "lower"
        assert runner_metric_direction(runner, "r2") == "higher"

    def test_a_missing_value_ranks_last_either_way(self) -> None:
        values = {"a": 0.5, "b": None, "c": 0.1}
        for direction, expected in (("lower", "cab"), ("higher", "acb")):
            ranked = rank_by_metric(list(values), values.get, direction)
            assert "".join(ranked) == expected


class TestComparison:
    def test_lower_is_better_ranks_ascending(self) -> None:
        runner = _ScoredRunner("rmse", {"a": 0.5, "b": 0.2, "c": 0.9})

        trials = run_model_comparison(runner, _base(), ["a", "b", "c"], "rmse")

        assert [t.model_arch for t in trials] == ["b", "a", "c"]

    def test_higher_is_better_is_unchanged(self) -> None:
        runner = _ScoredRunner("r2", {"a": 0.5, "b": 0.2, "c": 0.9})

        trials = run_model_comparison(runner, _base(), ["a", "b", "c"], "r2")

        assert [t.model_arch for t in trials] == ["c", "a", "b"]

    def test_a_task_declaration_beats_the_name(self) -> None:
        # "score" reads as higher-is-better; this task says otherwise.
        runner = _ScoredRunner(
            "score", {"a": 0.5, "b": 0.2, "c": 0.9}, declared={"score": "lower"}
        )

        trials = run_model_comparison(runner, _base(), ["a", "b", "c"], "score")

        assert trials[0].model_arch == "b"


class TestSweep:
    @pytest.mark.parametrize(
        ("metric", "expected_first"), [("mae", 0.1), ("accuracy", 0.3)]
    )
    def test_grid_ranks_by_the_metric_direction(
        self, metric: str, expected_first: float
    ) -> None:
        runner = _ScoredRunner(metric, {0.1: 0.2, 0.2: 0.5, 0.3: 0.9})

        trials = run_sweep(
            runner,
            _base(),
            {"training.learning_rate": [0.1, 0.2, 0.3]},
            mode="grid",
            metric=metric,
        )

        assert trials[0].overrides["training.learning_rate"] == expected_first

    @pytest.mark.parametrize(
        ("metric", "optuna_direction"), [("loss", "minimize"), ("r2", "maximize")]
    )
    def test_optuna_searches_in_the_metric_direction(
        self, metric: str, optuna_direction: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import optuna

        seen: dict[str, Any] = {}
        real = optuna.create_study

        def spy(**kwargs: Any) -> Any:
            seen.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(optuna, "create_study", spy)
        runner = _ScoredRunner(metric, {0.01: 0.5})

        run_sweep(
            runner,
            {**_base(), "training": {"learning_rate": 0.01, "momentum": 0.5}},
            {"training.momentum": {"type": "uniform", "low": 0.0, "high": 1.0}},
            mode="optuna",
            metric=metric,
            n_trials=2,
        )

        assert seen["direction"] == optuna_direction


class TestReplicatedComparison:
    def test_a_task_declaration_reaches_best_by_mean(self) -> None:
        class _ByVariant(_ScoredRunner):
            def run(self, cfg: dict[str, Any]) -> RunResult:
                value = 0.2 if "_low_" in cfg["name"] else 0.8
                return RunResult(metrics={"score": value}, status="success")

        runner = _ByVariant("score", {}, declared={"score": "lower"})

        report = run_replicated_comparison(
            runner, _base(), {"low": {}, "high": {}}, [1, 2, 3], "score"
        )

        assert report["metric_direction"] == "lower"
        assert report["best_by_mean"] == "low"


class TestCustomTaskRunnerDeclaresItsDirections:
    def test_it_exposes_the_registry_directions(self) -> None:
        from types import SimpleNamespace

        from visionforge.tasks.runner import CustomTaskRunner

        info = SimpleNamespace(
            key="t",
            spec_cls=SimpleNamespace(Config=dict),
            metrics={"score": "lower"},
            primary_metric="score",
        )
        runner = CustomTaskRunner(info)  # type: ignore[arg-type]

        assert runner_metric_direction(runner, "score") == "lower"


class TestTheExportsFollowTheRanking:
    """best_trial, the ranking CSV and the LaTeX "#" all read the sweep's order."""

    def test_a_lower_is_better_sweep_end_to_end(self, tmp_path: Path) -> None:
        import asyncio
        import csv

        import visionforge.gui.api.routes as routes_mod
        from visionforge.core.cancellation import CancellationToken
        from visionforge.gui.api.schemas import SweepRequest

        runner = _ScoredRunner("rmse", {0.1: 0.7, 0.2: 0.3, 0.3: 0.5})
        base = {**_base(), "output": {"reports_dir": str(tmp_path)}}
        req = SweepRequest(
            config=base, search_space={"training.learning_rate": [0.1, 0.2, 0.3]}
        )
        routes_mod._active_cancel_token = CancellationToken()
        routes_mod._event_queue = None
        try:
            asyncio.run(routes_mod._execute_sweep(runner, base, req, "rmse", "r"))
            state = dict(routes_mod._current_run or {})
        finally:
            routes_mod._active_cancel_token = None
            routes_mod._current_run = None

        report = state["report"]
        assert report["metric_direction"] == "lower"
        assert report["best_trial"]["metrics"]["rmse"] == 0.3
        with (Path(report["report_dir"]) / "sweep_ranking.csv").open(
            encoding="utf-8", newline=""
        ) as f:
            rows = list(csv.DictReader(f))
        assert [(r["rank"], r["rmse"]) for r in rows] == [
            ("1", "0.3"),
            ("2", "0.5"),
            ("3", "0.7"),
        ]
        tex = (Path(report["report_dir"]) / "sweep_table.tex").read_text("utf-8")
        first = next(
            line for line in tex.splitlines() if line.strip().startswith("1 &")
        )
        assert "0.3000" in first
