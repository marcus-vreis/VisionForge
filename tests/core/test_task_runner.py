from __future__ import annotations

from visionforge.core.task_runner import RunResult, TaskRunner


class TestRunResult:
    def test_success_construction(self) -> None:
        """RunResult must accept all fields and expose them as attributes."""
        r = RunResult(
            metrics={"accuracy": 0.9, "f1": 0.88},
            status="success",
            training_time_s=12.3,
            error="",
        )
        assert r.metrics == {"accuracy": 0.9, "f1": 0.88}
        assert r.status == "success"
        assert r.training_time_s == 12.3
        assert r.error == ""

    def test_failed_construction(self) -> None:
        """RunResult with failed status must carry error text and no timing."""
        r = RunResult(
            metrics={},
            status="failed",
            training_time_s=None,
            error="model exploded",
        )
        assert r.status == "failed"
        assert r.metrics == {}
        assert r.training_time_s is None
        assert r.error == "model exploded"

    def test_metrics_is_mutable_dict(self) -> None:
        """metrics field must be a plain dict, writable after construction."""
        r = RunResult(metrics={}, status="success", training_time_s=0.0, error="")
        r.metrics["loss"] = 0.25
        assert r.metrics["loss"] == 0.25


class _FakeRunner:
    """Minimal TaskRunner implementation for protocol conformance tests."""

    def run(self, cfg: object) -> RunResult:  # noqa: ARG002
        return RunResult(
            metrics={"score": 1.0}, status="success", training_time_s=0.1, error=""
        )

    def metrics(self, result: RunResult) -> dict[str, float]:
        return result.metrics

    def primary_metric(self) -> str:
        return "score"


class TestTaskRunnerProtocol:
    def test_fake_runner_is_instance(self) -> None:
        """A class satisfying the protocol surface must pass isinstance check."""
        assert isinstance(_FakeRunner(), TaskRunner)

    def test_run_returns_run_result(self) -> None:
        """run() must return a RunResult with the expected shape."""
        runner = _FakeRunner()
        result = runner.run(None)
        assert isinstance(result, RunResult)
        assert result.status == "success"

    def test_metrics_returns_dict(self) -> None:
        """metrics() must return the dict from the RunResult."""
        runner = _FakeRunner()
        result = runner.run(None)
        assert runner.metrics(result) == {"score": 1.0}

    def test_primary_metric_returns_str(self) -> None:
        """primary_metric() must return a non-empty string."""
        runner = _FakeRunner()
        pm = runner.primary_metric()
        assert isinstance(pm, str)
        assert pm  # non-empty

    def test_non_conforming_class_fails_isinstance(self) -> None:
        """A class missing protocol methods must not satisfy isinstance."""

        class BadRunner:
            pass

        assert not isinstance(BadRunner(), TaskRunner)
