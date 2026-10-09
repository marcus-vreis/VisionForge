"""The K-fold run.json reports the epochs its folds actually ran.

Fold records carried no epoch count, so the top-level `total_epochs` summed a
key nobody wrote and was always 0 -- History showed "0 epochs" for a K-fold that
had trained for hours.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from visionforge.blocks.cross_validation import CrossValidationBlock
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.selftest_data import build_classification_dataset


def _config(base: Path, tmp_path: Path, epochs: int) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "kfold_epochs",
            "block": "cross_validation",
            "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
            "training": {"epochs": epochs, "batch_size": 4, "seed": 0},
            "data": {
                "base_dir": str(base),
                "num_workers": 0,
                "pin_memory": False,
                "transforms": {"image_size": 32},
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


def _run_json(config: ExperimentConfig) -> dict[str, Any]:
    path = config.output.models_dir / f"{config.name}_cv" / "run.json"
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


class TestKFoldEpochs:
    @pytest.fixture
    def trained(self, tmp_path: Path) -> tuple[CrossValidationBlock, ExperimentConfig]:
        base = build_classification_dataset(tmp_path / "ds")
        config = _config(base, tmp_path, epochs=2)
        block = CrossValidationBlock()
        block.setup(config)
        block.run()
        return block, config

    def test_each_fold_records_its_epochs(
        self, trained: tuple[CrossValidationBlock, ExperimentConfig]
    ) -> None:
        block, _ = trained

        assert [r["epochs_completed"] for r in block._fold_results] == [2, 2]

    def test_run_json_total_epochs_is_the_sum_over_folds(
        self, trained: tuple[CrossValidationBlock, ExperimentConfig]
    ) -> None:
        _, config = trained

        data = _run_json(config)

        assert data["metrics"]["total_epochs"] == 4
        folds = data["metrics"]["fold_results"]
        assert [f["epochs_completed"] for f in folds] == [2, 2]

    def test_a_fold_that_died_before_training_ran_no_epochs(
        self, tmp_path: Path
    ) -> None:
        base = build_classification_dataset(tmp_path / "ds")
        config = _config(base, tmp_path, epochs=1)
        block = CrossValidationBlock()
        block.setup(config)

        with patch("visionforge.blocks.cross_validation.Trainer") as trainer_cls:
            trainer_cls.return_value.fit.side_effect = RuntimeError("boom")
            block.run()

        assert [r["status"] for r in block._fold_results] == ["failed", "failed"]
        assert [r["epochs_completed"] for r in block._fold_results] == [0, 0]
        assert _run_json(config)["metrics"]["total_epochs"] == 0

    def test_a_fold_cut_by_the_stop_keeps_the_epochs_it_ran(
        self, tmp_path: Path
    ) -> None:
        base = build_classification_dataset(tmp_path / "ds")
        config = _config(base, tmp_path, epochs=5)
        block = CrossValidationBlock()
        block.setup(config)

        cut = MagicMock()
        cut.stopped = True
        cut.total_epochs = 2
        cut.best_val_loss = 0.5
        cut.model_path = tmp_path / "missing.pth"
        with (
            patch("visionforge.blocks.cross_validation.Trainer") as trainer_cls,
            patch("visionforge.blocks.cross_validation.torch.load", return_value={}),
            patch("visionforge.blocks.cross_validation.Evaluator") as evaluator_cls,
            patch("torch.nn.Module.load_state_dict"),
        ):
            trainer_cls.return_value.fit.return_value = cut
            evaluator_cls.return_value.evaluate.return_value = MagicMock(
                accuracy=0.5, f1=0.5
            )
            block.run()

        assert block._fold_results[0]["status"] == "stopped"
        assert block._fold_results[0]["epochs_completed"] == 2
