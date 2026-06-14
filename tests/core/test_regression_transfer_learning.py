from __future__ import annotations

from pathlib import Path

import pytest
import torch.nn as nn
from pydantic import ValidationError

from visionforge.core.regression_trainer import RegressionTrainer
from visionforge.models.regression_factory import RegressionModelFactory
from visionforge.utils.regression_config import (
    RegressionConfig,
    RegressionModelConfig,
    RegressionTransferLearningConfig,
)


def _config(tmp_path: Path, transfer: dict | None) -> RegressionConfig:
    base = tmp_path / "ds"
    base.mkdir(parents=True, exist_ok=True)
    return RegressionConfig.model_validate(
        {
            "name": "reg_tl",
            "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
            "data": {"base_dir": str(base), "target_columns": ["t"]},
            "training": {"learning_rate": 0.01},
            "transfer_learning": transfer,
        }
    )


def _model() -> nn.Module:
    return RegressionModelFactory.create(
        RegressionModelConfig(name="resnet18", num_targets=1, pretrained=False)
    )


class TestTransferLearningConfig:
    def test_defaults_to_none(self, tmp_path: Path) -> None:
        assert _config(tmp_path, None).transfer_learning is None

    def test_multiplier_must_be_in_unit_interval(self) -> None:
        with pytest.raises(ValidationError):
            RegressionTransferLearningConfig(backbone_lr_multiplier=0.0)
        with pytest.raises(ValidationError):
            RegressionTransferLearningConfig(backbone_lr_multiplier=1.5)


class TestFeatureExtraction:
    def test_freezes_backbone_and_optimizes_head_only(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path, {"mode": "feature_extraction"})
        trainer = RegressionTrainer(cfg)
        model = _model()

        trainer._apply_transfer_learning(model)
        head, backbone = trainer._split_named_params(model)
        assert all(not p.requires_grad for p in backbone)
        assert all(p.requires_grad for p in head)

        opt = trainer._build_optimizer(model)
        assert len(opt.param_groups) == 1
        # only the (trainable) head params reach the optimizer
        assert len(opt.param_groups[0]["params"]) == len(head)


class TestFineTuning:
    def test_two_groups_with_discriminative_lr(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path, {"mode": "fine_tuning", "backbone_lr_multiplier": 0.1})
        trainer = RegressionTrainer(cfg)
        model = _model()

        trainer._apply_transfer_learning(model)  # fine_tuning leaves all trainable
        assert all(p.requires_grad for p in model.parameters())

        opt = trainer._build_optimizer(model)
        assert len(opt.param_groups) == 2
        backbone_group, head_group = opt.param_groups
        assert backbone_group["lr"] == pytest.approx(0.01 * 0.1)
        assert head_group["lr"] == pytest.approx(0.01)


class TestNoTransferLearning:
    def test_single_group_over_all_params(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path, None)
        trainer = RegressionTrainer(cfg)
        model = _model()

        trainer._apply_transfer_learning(model)  # no-op
        opt = trainer._build_optimizer(model)
        assert len(opt.param_groups) == 1
        assert len(opt.param_groups[0]["params"]) == len(list(model.parameters()))
        assert opt.param_groups[0]["lr"] == pytest.approx(0.01)
