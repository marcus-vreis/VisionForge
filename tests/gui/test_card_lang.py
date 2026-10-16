"""The Markdown model card follows the language of the page (ADR-116).

The card is an artifact: English when no language is bound (a script), the
bound language otherwise. The route binds the one the page sent in
``X-VF-Lang``; a request that names none gets the English card it always had.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from visionforge.gui.api.routes import _render_run_markdown, _stopped_extent
from visionforge.utils.messages import using_lang


def _data(**extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "exp_1",
        "experiment": "exp",
        "timestamp": "2026-05-22T10:00:00",
        "status": "completed",
        "device_used": "cpu",
        "config": {
            "task": "binary",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "epochs": 5,
                "batch_size": 4,
                "seed": 0,
                "mixed_precision": True,
                "scheduler": {"kind": "step"},
            },
            "data": {
                "preprocessing": {"steps": [{"kind": "resize"}]},
                "transforms": {"flip": True},
            },
        },
        "metrics": {"accuracy": 0.8, "total_epochs": 2},
        "metric_cis": {
            "accuracy": {
                "ci_low": 0.7,
                "ci_high": 0.9,
                "n_samples": 40,
                "n_resamples": 1000,
            }
        },
        "history": [{"epoch": 1, "loss": 0.5}],
        "artifacts": {"model": "m.pth", "graphics": ["cm.png"]},
        "tests": [
            {
                "label": "ext",
                "base_dir": "d",
                "timestamp": "t",
                "metrics": {"accuracy": 0.5},
            }
        ],
    }
    data.update(extra)
    return data


def _card(lang: str, **extra: Any) -> str:
    with using_lang(lang):  # type: ignore[arg-type]
        return _render_run_markdown(Path("."), _data(**extra))


class TestCardPortuguese:
    def test_labels_titles_and_headers(self) -> None:
        card = _card("pt", stopped=True)

        for expected in (
            "- **ID da execução:** `exp_1`",
            "- **Data e hora:** 2026-05-22T10:00:00",
            "- **Situação:** concluída",
            "- **Interrompida:** sim (2 de 5 épocas)",
            "- **Dispositivo usado:** `cpu`",
            "- **Diretório da execução:**",
            "- **Arquivo do checkpoint:** `m.pth`",
            "## Configuração",
            "- **Arquitetura:** resnet18",
            "- **Pré-treinado:** False",
            "- **Tarefa:** binary",
            "- **Otimizador:** adam",
            "- **Taxa de aprendizado:** 0.001",
            "- **Tamanho do batch:** 4",
            "- **Épocas (máx.):** 5",
            "- **Seed:** 0",
            "- **Scheduler:** step",
            "- **Precisão mista:** ativada",
            "## Pipeline de pré-processamento",
            "## Data augmentation / Normalização",
            "| Campo | Valor |",
            "## Métricas finais",
            "| Métrica | Valor | IC 95% (bootstrap) |",
            "as 40 imagens de teste (1000 reamostragens)",
            "## Histórico de treino (1 época(s))",
            "| Época | loss |",
            "## Gráficos",
            "## Testes em conjuntos de dados adicionais (1)",
        ):
            assert expected in card
        assert "num_classes" in card  # identifiers stay as they are

    def test_none_recorded(self) -> None:
        card = _card("pt", metrics={}, metric_cis={})

        assert "_(nenhuma registrada)_" in card
        assert "none recorded" not in card

    def test_nothing_is_left_in_english(self) -> None:
        card = _card("pt", stopped=True)

        for english in (
            "Run ID",
            "Timestamp",
            "Configuration",
            "Final metrics",
            "Training history",
            "Plots",
            "Learning rate",
            "Field | Value",
            "Intervals are percentile bootstrap",
        ):
            assert english not in card


class TestCardEnglish:
    def test_bound_and_unbound_are_the_same_english_card(self) -> None:
        with using_lang("en"):
            bound = _render_run_markdown(Path("."), _data(stopped=True))
        unbound = _render_run_markdown(Path("."), _data(stopped=True))

        assert bound == unbound
        assert "- **Run ID:** `exp_1`" in unbound
        assert "- **Stopped:** yes (2 of 5 epochs)" in unbound
        assert "## Training history (1 epoch(s))" in unbound
        assert "| Metric | Value | 95% CI (bootstrap) |" in unbound


class TestStoppedExtent:
    CV = {"metrics": {"cv_aggregate": {"n_folds": 3, "n_folds_ok": 1}}}
    BANK = {"metrics": {"total_epochs": 1}, "config": {"model": {"name": "patchcore"}}}
    NO_BANK = {
        "metrics": {"total_epochs": 0},
        "config": {"model": {"name": "patchcore"}},
    }
    EPOCHS = {"metrics": {"total_epochs": 2}, "config": {"training": {"epochs": 5}}}

    def test_portuguese(self) -> None:
        with using_lang("pt"):
            assert _stopped_extent(self.CV) == "1 de 3 folds concluídos"
            assert (
                _stopped_extent(self.BANK)
                == "banco de memória construído, pontuação não realizada"
            )
            assert (
                _stopped_extent(self.NO_BANK)
                == "antes de o banco de memória ser construído"
            )
            assert _stopped_extent(self.EPOCHS) == "2 de 5 épocas"

    def test_english(self) -> None:
        with using_lang("en"):
            assert _stopped_extent(self.CV) == "1 of 3 folds finished"
            assert _stopped_extent(self.BANK) == "memory bank built, scoring skipped"
            assert _stopped_extent(self.NO_BANK) == "before the memory bank was built"
            assert _stopped_extent(self.EPOCHS) == "2 of 5 epochs"


class TestExportRoute:
    def _get(self, tmp_path: Path, headers: dict[str, str]) -> str:
        import visionforge.gui.api.routes as routes_mod
        from visionforge.gui.server import app

        run_dir = tmp_path / "exp" / "20260522_100000_000000"
        run_dir.mkdir(parents=True)
        data = _data(run_dir=str(run_dir.resolve()), stopped=True)
        (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
        with patch.object(routes_mod, "_MODELS_DIR", tmp_path):
            client = TestClient(app, raise_server_exceptions=True)
            resp = client.get(f"/api/runs/{run_dir.name}/export_md", headers=headers)
        assert resp.status_code == 200
        return str(resp.text)

    def test_a_portuguese_page_gets_a_portuguese_card(self, tmp_path: Path) -> None:
        body = self._get(tmp_path, {"X-VF-Lang": "pt"})

        assert "- **ID da execução:** `20260522_100000_000000`" in body
        assert "- **Interrompida:** sim (2 de 5 épocas)" in body
        assert "## Métricas finais" in body
        assert "Run ID" not in body

    def test_an_english_page_gets_an_english_card(self, tmp_path: Path) -> None:
        body = self._get(tmp_path, {"X-VF-Lang": "en"})

        assert "- **Run ID:** `20260522_100000_000000`" in body
        assert "ID da execução" not in body

    def test_a_request_naming_no_language_gets_the_english_card(
        self, tmp_path: Path
    ) -> None:
        body = self._get(tmp_path, {})

        assert "- **Run ID:** `20260522_100000_000000`" in body
        assert "- **Stopped:** yes (2 of 5 epochs)" in body
