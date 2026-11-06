from __future__ import annotations

from typing import Any

from make_paper_assets import headline, macros, table_rows


def _cell(
    dataset: str,
    architecture: str,
    *,
    noise: float,
    k1: float,
    k5: float,
    ci: tuple[float, float] = (-0.01, 0.01),
    stable_k: int = 10,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "architecture": architecture,
        "ablation": "no_aug",
        "baseline_mean": 0.9,
        "baseline_sd": 0.02,
        "delta": -0.005,
        "delta_ci95": list(ci),
        "significant_holm": False,
        "noise_to_effect": noise,
        "reversal": {"1": k1, "3": 0.2, "5": k5},
        "smallest_stable_k": stable_k,
        "seeds_for_power": 40,
        "evaluation_half_width": 0.01,
    }


CELLS = [
    _cell("usk_full", "resnet18", noise=0.4, k1=0.2, k5=0.05, stable_k=3),
    _cell(
        "usk_full", "swin_t", noise=0.5, k1=0.6, k5=0.10, ci=(0.001, 0.02), stable_k=5
    ),
    _cell("beans", "convnext_tiny", noise=3.0, k1=0.4, k5=0.30),
]


class TestHeadline:
    def test_noise_threshold_counts_the_boundary_for_half_but_not_for_one(self) -> None:
        h = headline(CELLS)
        assert (h["noise_ge_half"], h["noise_gt_one"]) == (2, 1)

    def test_k5_at_exactly_ten_percent_is_not_above(self) -> None:
        assert headline(CELLS)["k5_above"] == 1

    def test_reversal_range_and_median(self) -> None:
        h = headline(CELLS)
        assert (h["k1_min"], h["k1_median"], h["k1_max"]) == (0.2, 0.4, 0.6)

    def test_ci_excluding_zero_on_either_side(self) -> None:
        assert headline(CELLS)["ci_excludes_zero"] == 1

    def test_stable_cells_split_by_family(self) -> None:
        h = headline(CELLS)
        assert (h["stable_cnn"], h["cells_cnn"]) == (1, 2)
        assert (h["stable_transformer"], h["cells_transformer"]) == (1, 1)

    def test_evaluation_noise_counts_each_baseline_once(self) -> None:
        twin = {**CELLS[0], "ablation": "res160"}
        wide = {**CELLS[2], "evaluation_half_width": 0.05}
        h = headline([CELLS[0], twin, CELLS[1], wide])
        assert (h["eval_below_sd"], h["pairs"]) == (2, 3)


class TestMacros:
    def test_percentages_are_escaped_for_latex(self) -> None:
        hypotheses = {
            "H1": {"verdict": "supported"},
            "H2": {"verdict": "supported"},
            "H3": {
                "verdict": "rejected",
                "per_architecture": {
                    a: {"sd_full": 0.02, "sd_10pct": 0.01}
                    for a in ("resnet18", "convnext_tiny", "swin_t")
                },
            },
        }
        text = macros(headline(CELLS), hypotheses, wall_hours=16.3)
        assert "\\newcommand{\\KOneMedian}{40\\%}" in text
        assert "\\newcommand{\\HThree}{rejected}" in text


class TestTableRows:
    def test_one_row_per_cell_and_a_rule_between_datasets(self) -> None:
        lines = table_rows(CELLS).splitlines()[1:]
        assert sum(line.endswith("\\\\") for line in lines) == 3
        assert lines.index("\\midrule") == 2

    def test_signed_effect_with_interval(self) -> None:
        first = table_rows(CELLS).splitlines()[1]
        assert "-0.005 [-0.010, +0.010]" in first
