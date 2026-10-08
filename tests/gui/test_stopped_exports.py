"""Exported files say what ran: ranking CSV, model card and LaTeX (ADR-111).

The JSON summary keeps every unit with its status. The files a researcher
copies into a paper or a spreadsheet must not mix a cut unit's half-trained
numbers in with finished ones, and the model card must say a run was stopped.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from visionforge.core.latex_export import cv_to_latex, replicates_to_latex
from visionforge.gui.api.routes import _render_run_markdown, _write_advanced_summary


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class TestRankingCsv:
    """Like the classification comparison's ranking.csv: finished rows, ranked."""

    def test_a_comparison_ranks_only_finished_models(self, tmp_path: Path) -> None:
        report: dict[str, Any] = {
            "metric": "r2",
            "trials": [
                {"model_arch": "a", "status": "success", "metrics": {"r2": 0.9}},
                {"model_arch": "b", "status": "success", "metrics": {"r2": 0.8}},
                {"model_arch": "c", "status": "stopped", "metrics": {"r2": 0.95}},
                {"model_arch": "d", "status": "failed", "metrics": {}},
            ],
        }

        out = _write_advanced_summary(
            {"name": "cmp", "output": {"reports_dir": str(tmp_path)}},
            "comparison",
            report,
        )

        rows = _rows(Path(out) / "comparison_ranking.csv")
        assert [(r["rank"], r["model_arch"]) for r in rows] == [("1", "a"), ("2", "b")]

    def test_a_kfold_lists_only_finished_folds_without_a_rank(
        self, tmp_path: Path
    ) -> None:
        report: dict[str, Any] = {
            "n_folds": 3,
            "fold_results": [
                {"fold": 0, "status": "success", "metrics": {"r2": 0.81}},
                {"fold": 1, "status": "stopped", "metrics": {"r2": 0.12}},
            ],
            "aggregate": {"r2": {"mean": 0.81, "std": None, "n": 1}},
        }

        out = _write_advanced_summary(
            {"name": "cv", "output": {"reports_dir": str(tmp_path)}}, "cv", report
        )

        rows = _rows(Path(out) / "cv_ranking.csv")
        assert [r["fold"] for r in rows] == ["0"]
        assert "rank" not in rows[0]


class TestModelCard:
    def _card(self, **extra: Any) -> str:
        data: dict[str, Any] = {
            "experiment": "exp",
            "status": "completed",
            "metrics": {"total_epochs": 2},
            "history": [],
            "artifacts": {},
            "config": {"training": {"epochs": 5}},
            **extra,
        }
        return _render_run_markdown(Path("."), data)

    def test_a_stopped_run_says_how_far_it_got(self) -> None:
        assert "- **Stopped:** yes (2 of 5 epochs)" in self._card(stopped=True)

    def test_a_run_that_was_not_stopped_says_nothing(self) -> None:
        assert "Stopped" not in self._card(stopped=False)
        assert "Stopped" not in self._card()  # run.json written before ADR-111

    def test_a_stopped_kfold_counts_folds(self) -> None:
        card = self._card(
            stopped=True,
            metrics={
                "total_epochs": 0,
                "cv_aggregate": {"n_folds": 3, "n_folds_ok": 1},
            },
        )

        assert "- **Stopped:** yes (1 of 3 folds finished)" in card

    def test_a_patchcore_stopped_after_its_bank(self) -> None:
        card = self._card(
            stopped=True,
            metrics={"total_epochs": 1},
            config={"model": {"name": "patchcore"}, "training": {"epochs": 1}},
        )

        assert "- **Stopped:** yes (memory bank built, scoring skipped)" in card


class TestLatexNamesTheSampleSd:
    def test_the_kfold_mean_row(self) -> None:
        tex = cv_to_latex(
            {
                "n_folds": 2,
                "fold_results": [
                    {"fold": 0, "status": "success", "metrics": {"r2": 0.7}},
                    {"fold": 1, "status": "success", "metrics": {"r2": 0.8}},
                ],
                "aggregate": {"r2": {"mean": 0.75, "std": 0.07, "n": 2}},
            }
        )

        assert r"\textbf{Mean $\pm$ SD ($n-1$)}" in tex

    def test_the_replicates_header(self) -> None:
        tex = replicates_to_latex(
            {"metric": "r2", "seeds": [1, 2], "aggregates": {"r2": {"mean": 0.8}}}
        )

        header = next(line for line in tex.splitlines() if "Metric &" in line)
        assert r"SD ($n-1$)" in header
