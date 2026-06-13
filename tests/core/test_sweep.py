from __future__ import annotations

from typing import Any

import pytest

from visionforge.core.sweep import run_sweep, validate_sweep_space
from visionforge.core.task_runner import RunResult


class _FakeConfig:
    """Stand-in config type whose model_validate just echoes the dict."""

    @classmethod
    def model_validate(cls, d: dict[str, Any]) -> dict[str, Any]:
        return d


class _FakeRunner:
    """Runner whose score peaks at learning_rate == 0.05 (higher is better)."""

    config_type = _FakeConfig

    def __init__(self, fail_lr: float | None = None) -> None:
        self._fail_lr = fail_lr
        self.seen: list[float] = []

    def run(self, cfg: dict[str, Any]) -> RunResult:
        lr = cfg["training"]["learning_rate"]
        self.seen.append(lr)
        if self._fail_lr is not None and lr == self._fail_lr:
            return RunResult(status="failed", error="boom")
        return RunResult(
            metrics={"score": 1.0 - abs(lr - 0.05)},
            status="success",
            training_time_s=0.1,
        )

    def metrics(self, result: RunResult) -> dict[str, float]:
        return result.metrics

    def primary_metric(self) -> str:
        return "score"


def _base() -> dict[str, Any]:
    return {
        "name": "x",
        "model": {"name": "resnet18"},
        "training": {"learning_rate": 0.01},
    }


class TestGridSweep:
    def test_runs_all_points_and_ranks_best_first(self) -> None:
        trials = run_sweep(
            _FakeRunner(),
            _base(),
            {"training.learning_rate": [0.01, 0.05, 0.1]},
            mode="grid",
            metric="score",
        )
        assert len(trials) == 3
        assert all(t.status == "success" for t in trials)
        assert trials[0].overrides["training.learning_rate"] == 0.05

    def test_cartesian_product_of_two_axes(self) -> None:
        trials = run_sweep(
            _FakeRunner(),
            _base(),
            {
                "training.learning_rate": [0.01, 0.1],
                "model.name": ["resnet18", "resnet50"],
            },
            mode="grid",
            metric="score",
        )
        assert len(trials) == 4

    def test_failed_trials_sort_last(self) -> None:
        trials = run_sweep(
            _FakeRunner(fail_lr=0.1),
            _base(),
            {"training.learning_rate": [0.05, 0.1]},
            mode="grid",
            metric="score",
        )
        assert trials[0].status == "success"
        assert trials[-1].status == "failed"
        assert trials[-1].error == "boom"


class TestRandomSweep:
    def test_deterministic_with_seed(self) -> None:
        space = {
            "training.learning_rate": {"type": "uniform", "low": 0.01, "high": 0.1}
        }
        a = run_sweep(
            _FakeRunner(),
            _base(),
            space,
            mode="random",
            metric="score",
            n_trials=5,
            seed=7,
        )
        b = run_sweep(
            _FakeRunner(),
            _base(),
            space,
            mode="random",
            metric="score",
            n_trials=5,
            seed=7,
        )
        lrs_a = sorted(t.overrides["training.learning_rate"] for t in a)
        lrs_b = sorted(t.overrides["training.learning_rate"] for t in b)
        assert lrs_a == lrs_b
        assert len(a) == 5

    def test_choice_and_log_uniform(self) -> None:
        space = {
            "training.learning_rate": {
                "type": "log_uniform",
                "low": 0.001,
                "high": 0.1,
            },
            "model.name": {"type": "choice", "options": ["resnet18", "resnet50"]},
        }
        trials = run_sweep(
            _FakeRunner(),
            _base(),
            space,
            mode="random",
            metric="score",
            n_trials=3,
            seed=0,
        )
        assert len(trials) == 3
        assert all(
            t.overrides["model.name"] in ("resnet18", "resnet50") for t in trials
        )


class TestGuards:
    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            run_sweep(
                _FakeRunner(),
                _base(),
                {"training.learning_rate": [0.1]},
                mode="bogus",
                metric="score",
            )

    def test_validate_sweep_space_rejects_unknown_path(self) -> None:
        with pytest.raises(ValueError, match="Unknown sweep path"):
            validate_sweep_space(_base(), ["training.nope"])

    def test_validate_sweep_space_accepts_known_paths(self) -> None:
        validate_sweep_space(_base(), ["training.learning_rate", "model.name"])
