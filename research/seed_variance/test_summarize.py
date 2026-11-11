from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import summarize
from summarize import (
    build_summary,
    evaluation_half_width,
    judge_h1,
    judge_h2,
    judge_h3,
    run_record_path,
    smallest_stable_k,
)


def _row(dataset: str, noise: float, k1: float | None) -> dict[str, Any]:
    return {"dataset": dataset, "noise_to_effect": noise, "reversal": {"1": k1}}


class TestH1:
    def test_exactly_half_at_threshold_supports(self) -> None:
        rows = [_row("a", 0.5, 0.0), _row("a", 0.49, 0.0)]
        assert judge_h1(rows) == {"cells": 2, "at_or_above": 1, "verdict": "supported"}

    def test_most_cells_below_rejects(self) -> None:
        rows = [_row("a", 0.4, 0.0), _row("a", 0.3, 0.0), _row("a", 2.0, 0.0)]
        assert judge_h1(rows)["verdict"] == "rejected"


class TestH2:
    def test_every_dataset_with_a_reversing_cell_supports(self) -> None:
        rows = [_row("a", 1, 0.05), _row("a", 1, 0.2), _row("b", 1, 0.11)]
        assert judge_h2(rows)["verdict"] == "supported"

    def test_ten_percent_exactly_is_not_above(self) -> None:
        rows = [_row("a", 1, 0.10), _row("b", 1, 0.0)]
        assert judge_h2(rows)["verdict"] == "rejected"

    def test_only_some_datasets_is_neither(self) -> None:
        rows = [_row("a", 1, 0.3), _row("b", 1, 0.1)]
        result = judge_h2(rows)
        assert result["verdict"] == "neither"
        assert result["dataset_has_cell_above"] == {"a": True, "b": False}

    def test_a_zero_effect_cell_counts_as_no_reversal(self) -> None:
        assert judge_h2([_row("a", 1, None)])["verdict"] == "rejected"


class TestH3:
    def test_one_architecture_not_larger_rejects(self) -> None:
        sd = {
            ("usk_10pct", "resnet18"): 0.05,
            ("usk_full", "resnet18"): 0.04,
            ("usk_10pct", "swin_t"): 0.02,
            ("usk_full", "swin_t"): 0.02,  # equal is not larger
        }
        assert judge_h3(sd)["verdict"] == "rejected"

    def test_every_architecture_larger_supports(self) -> None:
        sd = {("usk_10pct", "resnet18"): 0.05, ("usk_full", "resnet18"): 0.04}
        assert judge_h3(sd)["verdict"] == "supported"


class TestSmallestStableK:
    def test_consistent_effect_is_stable_at_one_seed(self) -> None:
        assert smallest_stable_k([0.01] * 10) == 1

    def test_one_contrary_seed_in_ten_is_stable_at_one(self) -> None:
        # k=1 reverses in exactly 1 of 10 draws: 10%, which the protocol accepts.
        assert smallest_stable_k([0.02] * 9 + [-0.01]) == 1

    def test_two_contrary_seeds_need_three(self) -> None:
        # k=1: 2/10 = 20%. k=3: subsets with both negatives and one positive have
        # mean (0.02 - 0.02) / 3 = 0, a reversal: 8 of C(10,3) = 120, about 7%.
        assert smallest_stable_k([0.02] * 8 + [-0.01, -0.01]) == 3


RESULTS = Path(__file__).resolve().parent / "results"


@pytest.fixture
def clean_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``outputs/`` to fall back on: only the files the repository publishes."""
    monkeypatch.setattr(summarize, "REPO", tmp_path)


@pytest.mark.usefixtures("clean_clone")
class TestCleanClone:
    def test_evaluation_half_width_comes_from_the_public_runs(self) -> None:
        for path in sorted(RESULTS.glob("*__*.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            assert evaluation_half_width(report) is not None, path.name

    def test_summary_json_regenerates_byte_for_byte(self) -> None:
        summary, missing = build_summary()
        assert missing == []
        text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        assert text == (RESULTS / "summary.json").read_text(encoding="utf-8")


class TestRunRecordPath:
    def test_falls_back_to_the_local_outputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(summarize, "RUNS", tmp_path / "runs")
        monkeypatch.setattr(summarize, "REPO", tmp_path)
        local = tmp_path / "outputs" / "models" / "exp_s0" / "1" / "run.json"
        local.parent.mkdir(parents=True)
        local.write_text("{}", encoding="utf-8")
        assert run_record_path("outputs\\models\\exp_s0\\1") == local

    def test_none_when_neither_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(summarize, "RUNS", tmp_path / "runs")
        monkeypatch.setattr(summarize, "REPO", tmp_path)
        assert run_record_path("outputs\\models\\exp_s0\\1") is None
