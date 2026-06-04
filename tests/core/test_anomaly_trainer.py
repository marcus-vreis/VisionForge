from __future__ import annotations

import json
from pathlib import Path

import torch

from visionforge.core.anomaly_trainer import (
    AnomalyTrainer,
    compute_auroc,
    compute_threshold,
)
from visionforge.models.anomaly_factory import ConvAutoencoder, PatchCore
from visionforge.utils.anomaly_config import AnomalyConfig


class FakeAnomalyDataModule:
    """Fixed (image, label) batches; train is normal-only, test is mixed."""

    def __init__(self, size: int = 16, n_batches: int = 2) -> None:
        self._s = size
        self._n = n_batches

    def train_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (torch.randn(2, 3, self._s, self._s), torch.zeros(2, dtype=torch.long))
            for _ in range(self._n)
        ]

    def test_loader(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (torch.randn(2, 3, self._s, self._s), torch.tensor([0, 1]))
            for _ in range(self._n)
        ]


def _config(tmp_path: Path, overrides: dict | None = None) -> AnomalyConfig:
    raw: dict = {
        "name": "anom_run",
        "model": {"name": "autoencoder", "latent_dim": 8},
        "data": {"base_dir": str(tmp_path), "image_size": 32},
        "training": {
            "learning_rate": 0.01,
            "epochs": 2,
            "batch_size": 2,
            "early_stopping_patience": 2,
            "seed": 0,
        },
        "output": {"models_dir": str(tmp_path / "models")},
        "device": {"kind": "cpu"},
    }
    if overrides:
        for key, val in overrides.items():
            raw.setdefault(key, {})
            if isinstance(val, dict):
                raw[key].update(val)
            else:
                raw[key] = val
    return AnomalyConfig.model_validate(raw)


# ── metric helpers ────────────────────────────────────────────────────────────


class TestMetricHelpers:
    def test_auroc_perfect_separation(self) -> None:
        scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
        labels = torch.tensor([0, 0, 1, 1])
        assert compute_auroc(scores, labels) == 1.0

    def test_auroc_reversed(self) -> None:
        scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
        labels = torch.tensor([0, 0, 1, 1])
        assert compute_auroc(scores, labels) == 0.0

    def test_auroc_single_class_returns_half(self) -> None:
        scores = torch.tensor([0.3, 0.6])
        labels = torch.tensor([0, 0])
        assert compute_auroc(scores, labels) == 0.5

    def test_threshold_percentile(self) -> None:
        normal = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        # 100th percentile is the max
        assert compute_threshold(normal, 100.0) == 4.0
        # 0th percentile is the min
        assert compute_threshold(normal, 0.0) == 0.0


# ── fit ───────────────────────────────────────────────────────────────────────


class TestFitAutoencoder:
    def test_runs_and_writes_run_json(self, tmp_path: Path) -> None:
        trainer = AnomalyTrainer(_config(tmp_path))
        result = trainer.fit(ConvAutoencoder(latent_dim=8), FakeAnomalyDataModule())
        assert result.model_path.is_file()
        run_json = json.loads(
            (result.model_path.parent / "run.json").read_text(encoding="utf-8")
        )
        assert run_json["status"] == "completed"
        for key in ("auroc", "threshold", "image_f1", "best_epoch"):
            assert key in run_json["metrics"]

    def test_emits_sse_events(self, tmp_path: Path) -> None:
        events: list[dict] = []
        trainer = AnomalyTrainer(_config(tmp_path))
        trainer.fit(
            ConvAutoencoder(latent_dim=8),
            FakeAnomalyDataModule(),
            progress_callback=events.append,
        )
        kinds = [e["event"] for e in events]
        assert kinds[0] == "start"
        assert kinds[-1] == "end"
        epoch_ev = next(e for e in events if e["event"] == "epoch_end")
        assert "val_auroc" in epoch_ev


class TestFitPatchCore:
    def test_patchcore_fit_runs(self, tmp_path: Path) -> None:
        cfg = _config(
            tmp_path,
            {
                "model": {
                    "name": "patchcore",
                    "backbone": "resnet18",
                    "pretrained": False,
                    "coreset_ratio": 0.5,
                },
                "data": {"base_dir": str(tmp_path), "image_size": 64},
            },
        )
        trainer = AnomalyTrainer(cfg)
        model = PatchCore(backbone="resnet18", pretrained=False, coreset_ratio=0.5)
        result = trainer.fit(model, FakeAnomalyDataModule(size=64))
        assert result.model_path.is_file()
        assert result.total_epochs == 1  # single fit pass

    def test_evaluate_returns_metrics(self, tmp_path: Path) -> None:
        trainer = AnomalyTrainer(_config(tmp_path))
        dm = FakeAnomalyDataModule()
        model = ConvAutoencoder(latent_dim=8)
        auroc, threshold, f1 = trainer.evaluate(
            model, dm.train_loader(), dm.test_loader()
        )
        assert 0.0 <= auroc <= 1.0
        assert 0.0 <= f1 <= 1.0
        assert isinstance(threshold, float)
