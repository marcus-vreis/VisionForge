"""LaTeX output is a deliverable, not a debug dump: these pin the exact
structure a paper depends on (booktabs rules, escaping, dispatch) so a
regression shows up here rather than in someone's manuscript.
"""

from __future__ import annotations

from typing import Any

from visionforge.core.latex_export import (
    comparison_to_latex,
    cv_to_latex,
    escape,
    replicates_to_latex,
    report_to_latex,
    sweep_to_latex,
)


def _replicates_report() -> dict[str, Any]:
    return {
        "metric": "r2",
        "seeds": [42, 43, 44],
        "total_replicates": 3,
        "successful_replicates": 3,
        "aggregates": {
            "r2": {
                "n": 3,
                "mean": 0.9123,
                "std": 0.0134,
                "ci95_low": 0.8790,
                "ci95_high": 0.9456,
                "boot95_low": 0.8901,
                "boot95_high": 0.9301,
            },
            "test_mae": {
                "n": 3,
                "mean": 0.2500,
                "std": 0.0100,
                "ci95_low": 0.2252,
                "ci95_high": 0.2748,
                "boot95_low": 0.2300,
                "boot95_high": 0.2690,
            },
        },
        "headline": {"mean": 0.9123},
    }


class TestEscaping:
    def test_metric_underscores_do_not_become_subscripts(self) -> None:
        assert escape("test_mae") == r"test\_mae"

    def test_percent_and_ampersand_are_escaped(self) -> None:
        assert escape("50% A&B") == r"50\% A\&B"

    def test_backslash_does_not_escape_its_own_replacement(self) -> None:
        # Naive ordering yields \textbackslash\{\} — broken input.
        assert escape("a\\b") == r"a\textbackslash{}b"


class TestReplicatesTable:
    def test_structure_is_a_valid_booktabs_table(self) -> None:
        tex = replicates_to_latex(_replicates_report(), experiment="reg_001")
        assert tex.count(r"\begin{table}") == 1
        assert tex.count(r"\end{table}") == 1
        for rule in (r"\toprule", r"\midrule", r"\bottomrule"):
            assert rule in tex
        assert r"\begin{tabular}{lrrcc}" in tex

    def test_reports_both_intervals_per_metric(self) -> None:
        tex = replicates_to_latex(_replicates_report())
        assert r"95\% CI (t)" in tex
        assert r"95\% CI (bootstrap)" in tex
        assert "[0.8790, 0.9456]" in tex  # t
        assert "[0.8901, 0.9301]" in tex  # bootstrap

    def test_marks_the_headline_metric(self) -> None:
        tex = replicates_to_latex(_replicates_report())
        r2_row = next(
            line for line in tex.splitlines() if line.strip().startswith("r2")
        )
        assert r"$\star$" in r2_row
        mae_row = next(
            line for line in tex.splitlines() if "test" in line and "&" in line
        )
        assert r"$\star$" not in mae_row

    def test_names_the_seeds_so_the_table_is_reproducible(self) -> None:
        assert "42, 43, 44" in replicates_to_latex(_replicates_report())

    def test_missing_bootstrap_renders_as_a_dash_not_none(self) -> None:
        report = _replicates_report()
        report["aggregates"]["r2"]["boot95_low"] = None
        report["aggregates"]["r2"]["boot95_high"] = None
        tex = replicates_to_latex(report)
        assert "None" not in tex
        assert "[---, ---]" in tex


class TestSweepTable:
    def _report(self) -> dict[str, Any]:
        return {
            "mode": "grid",
            "metric": "accuracy",
            "total_trials": 3,
            "best_trial": {"trial_index": 0},
            "trials": [
                {
                    "status": "success",
                    "overrides": {"training.learning_rate": 0.001},
                    "metrics": {"accuracy": 0.93},
                    "training_time_s": 12.5,
                },
                {
                    "status": "success",
                    "overrides": {"training.learning_rate": 0.01},
                    "metrics": {"accuracy": 0.88},
                    "training_time_s": 12.1,
                },
                {"status": "failed", "overrides": {}, "metrics": {}},
            ],
        }

    def test_lists_only_successful_trials_in_rank_order(self) -> None:
        tex = sweep_to_latex(self._report())
        rows = [
            line for line in tex.splitlines() if line.strip().startswith(("1 &", "2 &"))
        ]
        assert len(rows) == 2
        assert "0.9300" in rows[0]

    def test_column_per_searched_path_using_the_leaf_name(self) -> None:
        tex = sweep_to_latex(self._report())
        assert r"learning\_rate" in tex
        assert "training.learning" not in tex  # full dot-path would be noise

    def test_note_warns_that_one_run_per_config_is_seed_noise(self) -> None:
        assert "replicates" in sweep_to_latex(self._report())


class TestCvTable:
    def _report(self) -> dict[str, Any]:
        return {
            "n_folds": 2,
            "metric": "miou",
            "fold_results": [
                {"fold": 0, "status": "success", "metrics": {"miou": 0.71}},
                {"fold": 1, "status": "success", "metrics": {"miou": 0.75}},
            ],
            "aggregate": {"miou": {"mean": 0.73, "std": 0.02}},
        }

    def test_one_row_per_fold_plus_the_aggregate(self) -> None:
        tex = cv_to_latex(self._report())
        assert "1 & 0.7100" in tex
        assert "2 & 0.7500" in tex
        assert r"\textbf{Mean $\pm$ SD}" in tex
        assert r"0.7300 $\pm$ 0.0200" in tex

    def test_folds_are_numbered_from_one_for_the_reader(self) -> None:
        # fold indices are 0-based internally; a table saying "fold 0" reads wrong
        tex = cv_to_latex(self._report())
        assert not any(line.strip().startswith("0 &") for line in tex.splitlines())


class TestComparisonTable:
    def _comparisons(self) -> list[dict[str, Any]]:
        return [
            {
                "label_a": "novo",
                "label_b": "baseline",
                "mean_difference": 0.045,
                "ci_low": 0.021,
                "ci_high": 0.070,
                "test": "paired_t",
                "p_value": 0.0031,
                "effect_size": 1.42,
                "significant": True,
            },
            {
                "label_a": "novo",
                "label_b": "ablação",
                "mean_difference": 0.004,
                "ci_low": -0.010,
                "ci_high": 0.019,
                "test": "wilcoxon",
                "p_value": 0.4210,
                "effect_size": 0.12,
                "significant": False,
            },
        ]

    def test_stars_only_the_significant_row(self) -> None:
        tex = comparison_to_latex(self._comparisons())
        lines = [line for line in tex.splitlines() if "vs" in line]
        assert r"$^{*}$" in lines[0]
        assert r"$^{*}$" not in lines[1]

    def test_note_states_the_multiple_comparison_correction(self) -> None:
        tex = comparison_to_latex(self._comparisons())
        assert "Holm-Bonferroni" in tex
        assert r"\alpha=0.05" in tex

    def test_test_name_is_human_readable(self) -> None:
        assert r"paired t" in comparison_to_latex(self._comparisons())


class TestDispatch:
    def test_recognizes_each_report_shape(self) -> None:
        assert "tab:replicates" in (report_to_latex(_replicates_report()) or "")
        assert "tab:cv" in (report_to_latex(TestCvTable()._report()) or "")
        assert "tab:sweep" in (report_to_latex(TestSweepTable()._report()) or "")

    def test_unknown_shape_yields_no_table_instead_of_a_broken_one(self) -> None:
        assert report_to_latex({"metrics": {"accuracy": 0.9}}) is None
        assert report_to_latex({}) is None


class TestSmallSampleCaveat:
    """A tight interval from 2 seeds implies precision that does not exist;
    the table must say so where the reader will see it."""

    def test_warns_below_five_seeds(self) -> None:
        report = _replicates_report()
        report["successful_replicates"] = 2
        tex = replicates_to_latex(report)
        assert "Caution" in tex
        assert "$n<5$" in tex

    def test_no_caveat_once_there_are_enough_seeds(self) -> None:
        report = _replicates_report()
        report["successful_replicates"] = 10
        assert "Caution" not in replicates_to_latex(report)
