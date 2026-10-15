"""The LaTeX tables are artifacts: English unless a language is bound (ADR-116).

Under a Portuguese page the captions, headers and notes are Portuguese; the
markup that is not language (``\\textbf{...}``, ``$n<5$``) must survive the
``str.format`` pass with single braces.
"""

from __future__ import annotations

from typing import Any

from visionforge.core.latex_export import (
    comparison_to_latex,
    cv_to_latex,
    replicates_to_latex,
    sweep_to_latex,
)
from visionforge.utils.messages import using_lang


def _replicates(n_ok: int = 3) -> dict[str, Any]:
    return {
        "metric": "r2",
        "seeds": [42, 43, 44],
        "successful_replicates": n_ok,
        "aggregates": {
            "r2": {
                "n": n_ok,
                "mean": 0.9123,
                "std": 0.0134,
                "ci95_low": 0.8790,
                "ci95_high": 0.9456,
                "boot95_low": 0.8901,
                "boot95_high": 0.9301,
            }
        },
        "headline": {"mean": 0.9123},
    }


def _sweep(stopped: bool = False) -> dict[str, Any]:
    report: dict[str, Any] = {
        "mode": "grid",
        "metric": "accuracy",
        "total_trials": 2,
        "best_trial": {"trial_index": 0},
        "trials": [
            {
                "status": "success",
                "overrides": {"training.learning_rate": 0.001},
                "metrics": {"accuracy": 0.93},
                "training_time_s": 12.5,
            },
            {"status": "stopped", "overrides": {}, "metrics": {}},
        ],
    }
    if stopped:
        report.update(planned_trials=6, stopped_count=1, stopped=True)
    return report


def _cv(stopped: bool = False) -> dict[str, Any]:
    folds = [{"fold": 0, "status": "success", "metrics": {"r2": 0.81}}]
    if stopped:
        folds.append({"fold": 1, "status": "stopped", "metrics": {"r2": 0.12}})
    return {
        "n_folds": 5 if stopped else 1,
        "fold_results": folds,
        "aggregate": {"r2": {"mean": 0.81, "std": None, "n": 1}},
        "stopped": stopped,
    }


def _comparisons(underpowered: bool = False) -> list[dict[str, Any]]:
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
            "underpowered": underpowered,
            "min_achievable_p": 0.0625,
        }
    ]


def _assert_braces_render(tex: str) -> None:
    assert "{{" not in tex
    assert "}}" not in tex


class TestPortuguese:
    def test_replicates(self) -> None:
        with using_lang("pt"):
            tex = replicates_to_latex(_replicates(n_ok=2), experiment="reg_001")

        assert r"\caption{reg\_001: réplicas com várias seeds ($n=2$ seeds).}" in tex
        assert (
            r"Métrica & Média & DP ($n-1$) & IC 95\% (t) & IC 95\% (bootstrap)" in tex
        )
        assert "métrica principal" in tex
        assert r"\textbf{Atenção:} com $n<5$ seeds" in tex
        for english in ("Metric", "Mean", "headline", "Caution", "multi-seed"):
            assert english not in tex
        _assert_braces_render(tex)

    def test_replicates_with_unfinished_seeds(self) -> None:
        report = _replicates(n_ok=2)
        report["trials"] = [
            {"seed": 42, "status": "success"},
            {"seed": 43, "status": "success"},
            {"seed": 44, "status": "stopped"},
        ]

        with using_lang("pt"):
            tex = replicates_to_latex(report)

        assert "(2 de 3 solicitadas; 1 interrompida(s))" in tex
        assert "requested" not in tex

    def test_sweep(self) -> None:
        with using_lang("pt"):
            tex = sweep_to_latex(_sweep())

        assert "Experimento: varredura de hiperparâmetros (grid), ordenada por" in tex
        assert "Tempo (s)" in tex
        assert "1 de 2 tentativas concluídas com sucesso." in tex
        assert "confirme a vencedora com réplicas" in tex
        for english in (
            "Time (s)",
            "trials succeeded",
            "Ranking",
            "hyperparameter sweep",
        ):
            assert english not in tex
        _assert_braces_render(tex)

    def test_stopped_sweep(self) -> None:
        with using_lang("pt"):
            tex = sweep_to_latex(_sweep(stopped=True))

        assert (
            "A varredura foi interrompida após 2 de 6 tentativas planejadas "
            "(1 interrompida(s) antes de terminar)." in tex
        )
        assert "was stopped" not in tex
        _assert_braces_render(tex)

    def test_cv(self) -> None:
        with using_lang("pt"):
            tex = cv_to_latex(_cv())

        assert r"\caption{Experimento: validação cruzada com 1 folds.}" in tex
        assert r"\textbf{Média $\pm$ DP ($n-1$)}" in tex
        assert "cross-validation" not in tex
        assert r"\textbf{Mean" not in tex
        _assert_braces_render(tex)

    def test_stopped_cv(self) -> None:
        with using_lang("pt"):
            tex = cv_to_latex(_cv(stopped=True))

        assert "1 de 5 folds concluídos (1 interrompida(s), 3 não executada(s))" in tex
        assert "finished" not in tex
        assert "not run" not in tex
        _assert_braces_render(tex)

    def test_comparison(self) -> None:
        with using_lang("pt"):
            tex = comparison_to_latex(_comparisons())

        assert "Comparação & Diferença & IC 95\\% & Teste" in tex
        assert "novo vs. baseline" in tex
        assert r"$^{*}$ significativo a $\alpha=0.05$" in tex
        assert "Holm-Bonferroni" in tex
        assert "paired comparison" not in tex
        assert "significant at" not in tex
        _assert_braces_render(tex)
        assert tex.count(r"$^{*}$") == 2  # the row star and the note, single braces

    def test_underpowered_comparison(self) -> None:
        with using_lang("pt"):
            tex = comparison_to_latex(_comparisons(underpowered=True))

        assert r"\textbf{Poder insuficiente:}" in tex
        assert "$p<0.0625$" in tex
        assert "Underpowered" not in tex
        _assert_braces_render(tex)


class TestEnglish:
    def test_unbound_is_english_and_equals_bound_to_english(self) -> None:
        def render() -> str:
            return replicates_to_latex(_replicates(n_ok=2)) + sweep_to_latex(
                _sweep(stopped=True)
            )

        unbound = render()
        with using_lang("en"):
            assert render() == unbound
        assert r"\textbf{Caution:} with $n<5$ seeds" in unbound
        assert "The sweep was stopped after 2 of 6 planned trials" in unbound
        _assert_braces_render(unbound)
