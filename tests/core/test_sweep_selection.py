from __future__ import annotations

from typing import Any, NoReturn

import pytest

from visionforge.core.sweep import run_sweep
from visionforge.core.task_runner import RunResult


class Config:
    """Keep synthetic candidates as dictionaries."""

    @classmethod
    def model_validate(cls, value: dict[str, Any]) -> dict[str, Any]:
        return value


class SplitRunner:
    """Return opposing validation and test scores for each candidate."""

    config_type = Config

    def __init__(self, test_sign: float = -1.0) -> None:
        self.test_sign = test_sign
        self.seen: list[float] = []

    def run(self, cfg: dict[str, Any]) -> RunResult:
        candidate = cfg["candidate"]
        self.seen.append(candidate)
        result = RunResult(
            metrics={"score": self.test_sign * candidate}, status="success"
        )
        result.selection_metrics = {"score": candidate}
        return result

    def metrics(self, result: RunResult) -> dict[str, float]:
        return result.metrics

    def primary_metric(self) -> str:
        return "score"


@pytest.mark.parametrize("mode", ["grid", "random", "optuna"])
def test_selection_uses_validation_and_is_invariant_to_test_scores(mode: str) -> None:
    space = (
        {"candidate": [0.1, 0.9]}
        if mode == "grid"
        else {"candidate": {"type": "choice", "options": [0.1, 0.9]}}
    )
    first, changed_test = SplitRunner(), SplitRunner(test_sign=10.0)
    trials = run_sweep(
        first, {"candidate": 0.1}, space, mode=mode, metric="score", n_trials=12, seed=7
    )
    other = run_sweep(
        changed_test,
        {"candidate": 0.1},
        space,
        mode=mode,
        metric="score",
        n_trials=12,
        seed=7,
    )
    assert {t.overrides["candidate"] for t in trials} == {0.1, 0.9}
    assert trials[0].overrides["candidate"] == 0.9
    assert [t.overrides for t in trials] == [t.overrides for t in other]
    assert first.seen == changed_test.seen
    assert all(t.metrics["score"] == t.overrides["candidate"] for t in trials)
    assert all(t.selection_split == "validation" for t in trials)
    assert all(
        t.evaluation_metrics["score"] == -t.overrides["candidate"] for t in trials
    )


def test_test_metrics_are_never_a_fallback_for_missing_validation() -> None:
    class TestOnlyRunner(SplitRunner):
        def run(self, cfg: dict[str, Any]) -> RunResult:
            return RunResult(metrics={"score": 0.99}, status="success")

    trials = run_sweep(
        TestOnlyRunner(), {"candidate": 0.1}, {}, mode="grid", metric="score"
    )
    assert trials[0].status == "failed"
    assert "validation" in trials[0].error.lower()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_validation_is_not_ranked(value: float) -> None:
    trials = run_sweep(
        SplitRunner(), {"candidate": value}, {}, mode="grid", metric="score"
    )
    assert trials[0].status == "failed"
    assert "finite" in trials[0].error.lower()


def test_anomaly_sweep_rejects_test_only_objective_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from visionforge.blocks.anomaly_runner import AnomalyRunner

    def unexpected_run(*args: Any) -> NoReturn:
        raise AssertionError("unsafe anomaly sweep started training")

    monkeypatch.setattr(AnomalyRunner, "run", unexpected_run)
    with pytest.raises(ValueError, match="validation"):
        run_sweep(AnomalyRunner(), {}, {}, mode="grid", metric="auroc")
