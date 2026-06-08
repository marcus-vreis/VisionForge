from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from visionforge.core.comparison import ComparisonReport, GenericComparisonRunner
from visionforge.core.task_runner import RunResult


class _FakeRunner:
    """TaskRunner stub returning a canned RunResult per config (a dict)."""

    def __init__(self, primary: str = "r2") -> None:
        self._primary = primary

    def run(self, cfg: dict) -> RunResult:
        if cfg.get("boom"):
            raise RuntimeError("trial exploded")
        if cfg.get("fail"):
            return RunResult(metrics={}, status="failed", error="bad trial")
        return RunResult(
            metrics=cfg["metrics"],
            status="success",
            training_time_s=cfg.get("time", 1.0),
        )

    def metrics(self, result: RunResult) -> dict[str, float]:
        return result.metrics

    def primary_metric(self) -> str:
        return self._primary


_METRICS = ("mse", "rmse", "mae", "r2")


def _trial(name: str, **metrics: float) -> tuple[str, dict]:
    return name, {"metrics": dict(metrics)}


class TestRanking:
    def test_ranks_descending_by_metric(self, tmp_path: Path) -> None:
        """Trials must be ordered by the chosen metric, highest first."""
        runner = GenericComparisonRunner(_FakeRunner(), _METRICS)
        trials = [_trial("a", r2=0.7), _trial("b", r2=0.9), _trial("c", r2=0.5)]
        report = runner.compare(trials, rank_by="r2", out_dir=tmp_path)
        archs = [t["model_arch"] for t in report.trials]
        assert archs == ["b", "a", "c"]

    def test_records_all_metric_columns(self, tmp_path: Path) -> None:
        """Each trial record must carry every configured metric name."""
        runner = GenericComparisonRunner(_FakeRunner(), _METRICS)
        trials = [_trial("a", mse=0.1, rmse=0.3, mae=0.2, r2=0.8), _trial("b", r2=0.9)]
        report = runner.compare(trials, rank_by="r2", out_dir=tmp_path)
        top = report.trials[0]
        for m in _METRICS:
            assert m in top

    def test_unknown_rank_metric_raises(self, tmp_path: Path) -> None:
        """rank_by outside the configured metric names must raise ValueError."""
        runner = GenericComparisonRunner(_FakeRunner(), _METRICS)
        with pytest.raises(ValueError, match="rank_by"):
            runner.compare([_trial("a", r2=0.8)], rank_by="accuracy", out_dir=tmp_path)


class TestArtifacts:
    def test_summary_json_has_all_trials(self, tmp_path: Path) -> None:
        """comparison_summary.json must include failed trials too."""
        runner = GenericComparisonRunner(_FakeRunner(), _METRICS)
        trials = [_trial("a", r2=0.8), ("b", {"fail": True})]
        runner.compare(trials, rank_by="r2", out_dir=tmp_path)
        data = json.loads((tmp_path / "comparison_summary.json").read_text("utf-8"))
        assert len(data) == 2

    def test_ranking_csv_only_successful_with_metric_columns(
        self, tmp_path: Path
    ) -> None:
        """ranking.csv must list successful trials with the metric columns + rank."""
        runner = GenericComparisonRunner(_FakeRunner(), _METRICS)
        trials = [_trial("a", r2=0.8), _trial("b", r2=0.9), ("c", {"fail": True})]
        runner.compare(trials, rank_by="r2", out_dir=tmp_path)
        with (tmp_path / "ranking.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert [r["model_arch"] for r in rows] == ["b", "a"]
        assert [int(r["rank"]) for r in rows] == [1, 2]
        for m in _METRICS:
            assert m in rows[0]


class TestFailureHandling:
    def test_failed_and_raising_trials_kept_but_unranked(self, tmp_path: Path) -> None:
        """status='failed' (returned or raised) trials are recorded, ranked last."""
        runner = GenericComparisonRunner(_FakeRunner(), _METRICS)
        trials = [("a", {"boom": True}), _trial("b", r2=0.9), ("c", {"fail": True})]
        report = runner.compare(trials, rank_by="r2", out_dir=tmp_path)
        assert report.trials[0]["model_arch"] == "b"
        statuses = {t["model_arch"]: t["status"] for t in report.trials}
        assert statuses == {"a": "failed", "b": "success", "c": "failed"}
        assert "exploded" in next(
            t["error"] for t in report.trials if t["model_arch"] == "a"
        )

    def test_summary_raises_when_all_failed(self, tmp_path: Path) -> None:
        """ComparisonReport.summary() must raise when no trial succeeded."""
        runner = GenericComparisonRunner(_FakeRunner(), _METRICS)
        report = runner.compare(
            [("a", {"fail": True}), ("b", {"fail": True})],
            rank_by="r2",
            out_dir=tmp_path,
        )
        with pytest.raises(RuntimeError, match="all architectures failed"):
            report.summary()

    def test_summary_counts_and_top3(self, tmp_path: Path) -> None:
        """summary() reports total_ran, failed_count and at most three winners."""
        runner = GenericComparisonRunner(_FakeRunner(), _METRICS)
        trials = [
            _trial("a", r2=0.5),
            _trial("b", r2=0.9),
            _trial("c", r2=0.7),
            _trial("d", r2=0.6),
            ("e", {"fail": True}),
        ]
        summary = runner.compare(trials, rank_by="r2", out_dir=tmp_path).summary()
        assert summary["total_ran"] == 5
        assert summary["failed_count"] == 1
        assert len(summary["top_3"]) == 3
        assert summary["top_3"][0]["model_arch"] == "b"


class TestCleanup:
    def test_gc_collect_called_once_per_trial(self, tmp_path: Path) -> None:
        """gc.collect must run once per trial for explicit memory reclaim."""
        with (
            patch("visionforge.core.comparison.gc.collect") as gc_mock,
            patch("visionforge.core.comparison.torch.cuda.empty_cache"),
        ):
            runner = GenericComparisonRunner(_FakeRunner(), _METRICS)
            runner.compare(
                [_trial("a", r2=0.8), _trial("b", r2=0.9)],
                rank_by="r2",
                out_dir=tmp_path,
            )
        assert gc_mock.call_count == 2


def test_report_dataclass_default_is_empty() -> None:
    """A bare ComparisonReport has no trials (defensive default)."""
    assert ComparisonReport().trials == []
