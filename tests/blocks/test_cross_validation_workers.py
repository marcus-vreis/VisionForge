"""The classification K-fold resolves `num_workers` like every other loader.

The GUI sends `num_workers = -1` ("automático", ADR-103). `DataModule` turns
that into what the machine can afford; the K-fold block built its own loaders
from the raw value, and DataLoader refuses a negative count -- so every fold of
every K-fold started from the interface failed before training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from visionforge.utils.config import ExperimentConfig
from visionforge.utils.selftest_data import build_classification_dataset


class TestResolveNumWorkers:
    """The one policy every classification loader uses (core/data.py)."""

    @pytest.fixture(autouse=True)
    def _affordable_is_four(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import visionforge.core.data as data_mod

        monkeypatch.setattr(data_mod, "suggested_workers", lambda loader_pools: 4)

    def test_automatic_takes_what_the_machine_affords(self) -> None:
        from visionforge.core.data import resolve_num_workers

        assert resolve_num_workers(-1, n_samples=10_000) == 4

    def test_a_request_above_the_budget_is_capped(self) -> None:
        from visionforge.core.data import resolve_num_workers

        assert resolve_num_workers(16, n_samples=10_000) == 4

    def test_a_request_within_the_budget_is_kept(self) -> None:
        from visionforge.core.data import resolve_num_workers

        assert resolve_num_workers(2, n_samples=10_000) == 2

    def test_a_small_dataset_loads_in_process(self) -> None:
        from visionforge.core.data import resolve_num_workers

        assert resolve_num_workers(2, n_samples=100) == 0

    def test_zero_stays_zero(self) -> None:
        from visionforge.core.data import resolve_num_workers

        assert resolve_num_workers(0, n_samples=10_000) == 0

    def test_the_data_module_uses_the_same_policy(self, tmp_path: Path) -> None:
        from visionforge.core.data import DataModule, resolve_num_workers

        base = build_classification_dataset(tmp_path / "ds")
        for requested in (-1, 0, 2, 16):
            config = _config(base, tmp_path, num_workers=requested)
            module = DataModule(config)
            expected = resolve_num_workers(requested, n_samples=12)
            assert module._num_workers == expected, requested
            module.close()


def _config(base: Path, tmp_path: Path, **data: Any) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "kfold_workers",
            "block": "cross_validation",
            "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
            "training": {"epochs": 1, "batch_size": 4, "seed": 0},
            "data": {
                "base_dir": str(base),
                "pin_memory": False,
                "transforms": {"image_size": 32},
                **data,
            },
            "output": {
                "models_dir": str(tmp_path / "models"),
                "reports_dir": str(tmp_path / "reports"),
                "graphics_dir": str(tmp_path / "graphics"),
                "logs_dir": str(tmp_path / "logs"),
            },
            "device": {"kind": "cpu"},
            "cross_validation": {"n_folds": 2, "shuffle": False},
        }
    )


class TestKFoldWithAutomaticWorkers:
    def test_every_fold_trains(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import visionforge.core.data as data_mod
        from visionforge.blocks.cross_validation import CrossValidationBlock

        # In-process loading keeps the test fast; what is under test is that
        # -1 is resolved at all instead of reaching DataLoader.
        monkeypatch.setattr(data_mod, "suggested_workers", lambda loader_pools: 0)
        base = build_classification_dataset(tmp_path / "ds")
        block = CrossValidationBlock()
        block.setup(_config(base, tmp_path, num_workers=-1))

        block.run()

        assert [r["status"] for r in block._fold_results] == ["success", "success"]
        assert block.report()["n_folds_ok"] == 2

    def test_the_fold_loaders_get_the_resolved_count(self, tmp_path: Path) -> None:
        from torch.utils.data import Subset
        from torchvision.datasets import ImageFolder

        from visionforge.blocks.cross_validation import _FoldDataModule

        base = build_classification_dataset(tmp_path / "ds")
        config = _config(base, tmp_path, num_workers=-1)
        dataset = ImageFolder(str(base / "train"))
        fold = _FoldDataModule(
            Subset(dataset, [0, 1, 2]),
            Subset(dataset, [3, 4]),
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            config,
            num_workers=3,
        )

        assert fold.train_loader().num_workers == 3
        assert fold.val_loader().num_workers == 3
