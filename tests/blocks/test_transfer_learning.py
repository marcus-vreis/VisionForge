from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from visionforge.utils.config import ExperimentConfig

if TYPE_CHECKING:
    from visionforge.blocks.transfer_learning import TransferLearningBlock as _TLBlock


class TinyModel(nn.Module):
    """Minimal two-layer model for transfer-learning tests.

    Simulates backbone + head separation so freeze/unfreeze can be verified.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(3 * 32 * 32, 64)
        self.layer2 = nn.Linear(64, 32)
        # Acts as the classifier head
        self.fc = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x.flatten(1))
        x = self.layer2(x)
        return self.fc(x)


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Minimal ImageFolder structure for transfer-learning block tests."""
    from PIL import Image

    for split in ["train", "val", "test"]:
        for cls in ["class_a", "class_b"]:
            folder = tmp_path / split / cls
            folder.mkdir(parents=True)
            img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
            img.save(folder / "image.png")
    return tmp_path


@pytest.fixture
def base_config_dict(dataset_root: Path, tmp_path: Path) -> dict[str, Any]:
    return {
        "name": "tl_test",
        "task": "binary",
        "block": "transfer_learning",
        "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
        "training": {
            "learning_rate": 0.01,
            "epochs": 1,
            "batch_size": 2,
            "early_stopping_patience": 2,
            "seed": 0,
        },
        "data": {
            "base_dir": str(dataset_root),
            "num_workers": 0,
            "pin_memory": False,
            "transforms": {"image_size": 32},
        },
        "output": {
            "models_dir": str(tmp_path / "models"),
            "graphics_dir": str(tmp_path / "graphics"),
            "logs_dir": str(tmp_path / "logs"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "transfer_learning": {"mode": "feature_extraction"},
    }


@pytest.fixture
def fe_config(base_config_dict: dict[str, Any]) -> ExperimentConfig:
    """Config for feature_extraction mode."""
    return ExperimentConfig.model_validate(base_config_dict)


@pytest.fixture
def ft_config(base_config_dict: dict[str, Any]) -> ExperimentConfig:
    """Config for fine_tuning mode (unfreeze everything)."""
    raw = {**base_config_dict, "transfer_learning": {"mode": "fine_tuning"}}
    return ExperimentConfig.model_validate(raw)


@pytest.fixture
def ft_partial_config(base_config_dict: dict[str, Any]) -> ExperimentConfig:
    """Config for fine_tuning with unfreeze_from_layer=layer2."""
    raw = {
        **base_config_dict,
        "transfer_learning": {
            "mode": "fine_tuning",
            "unfreeze_from_layer": "layer2",
            "backbone_lr_multiplier": 0.1,
        },
    }
    return ExperimentConfig.model_validate(raw)


# ── TransferLearningConfig validation ─────────────────────────────────────────


class TestTransferLearningConfig:
    def test_default_mode_is_feature_extraction(self) -> None:
        """Default mode must be feature_extraction."""
        from visionforge.utils.config import TransferLearningConfig

        cfg = TransferLearningConfig()
        assert cfg.mode == "feature_extraction"

    def test_fine_tuning_mode_accepted(self) -> None:
        """fine_tuning is a valid mode."""
        from visionforge.utils.config import TransferLearningConfig

        cfg = TransferLearningConfig(mode="fine_tuning")
        assert cfg.mode == "fine_tuning"

    def test_invalid_mode_raises(self) -> None:
        """An unknown mode must raise a validation error."""
        from pydantic import ValidationError

        from visionforge.utils.config import TransferLearningConfig

        with pytest.raises(ValidationError):
            TransferLearningConfig(mode="invalid")  # type: ignore[arg-type]

    def test_unfreeze_from_layer_defaults_none(self) -> None:
        """unfreeze_from_layer defaults to None (unfreeze all)."""
        from visionforge.utils.config import TransferLearningConfig

        cfg = TransferLearningConfig()
        assert cfg.unfreeze_from_layer is None

    def test_backbone_lr_multiplier_default(self) -> None:
        """backbone_lr_multiplier defaults to 0.1."""
        from visionforge.utils.config import TransferLearningConfig

        cfg = TransferLearningConfig()
        assert cfg.backbone_lr_multiplier == 0.1

    def test_backbone_lr_multiplier_rejects_zero(self) -> None:
        """backbone_lr_multiplier must be > 0."""
        from pydantic import ValidationError

        from visionforge.utils.config import TransferLearningConfig

        with pytest.raises(ValidationError):
            TransferLearningConfig(backbone_lr_multiplier=0.0)

    def test_backbone_lr_multiplier_rejects_above_one(self) -> None:
        """backbone_lr_multiplier must be <= 1.0."""
        from pydantic import ValidationError

        from visionforge.utils.config import TransferLearningConfig

        with pytest.raises(ValidationError):
            TransferLearningConfig(backbone_lr_multiplier=1.1)

    def test_backbone_lr_multiplier_accepts_one(self) -> None:
        """backbone_lr_multiplier=1.0 is valid (upper bound is inclusive)."""
        from visionforge.utils.config import TransferLearningConfig

        cfg = TransferLearningConfig(backbone_lr_multiplier=1.0)
        assert cfg.backbone_lr_multiplier == 1.0


class TestExperimentConfigTransferLearning:
    def test_block_literal_accepts_transfer_learning(
        self, fe_config: ExperimentConfig
    ) -> None:
        """block='transfer_learning' must be accepted by ExperimentConfig."""
        assert fe_config.block == "transfer_learning"

    def test_transfer_learning_field_present(self, fe_config: ExperimentConfig) -> None:
        """transfer_learning field must be set in ExperimentConfig."""
        assert fe_config.transfer_learning is not None

    def test_transfer_learning_defaults_to_none(
        self, dataset_root: Path, tmp_path: Path
    ) -> None:
        """transfer_learning is optional — defaults to None when not specified."""
        config = ExperimentConfig.model_validate(
            {
                "name": "exp",
                "task": "binary",
                "block": "classification",
                "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
                "training": {"learning_rate": 0.01, "epochs": 1, "batch_size": 2},
                "data": {
                    "base_dir": str(dataset_root),
                    "num_workers": 0,
                    "pin_memory": False,
                },
            }
        )
        assert config.transfer_learning is None


# ── setup / teardown ──────────────────────────────────────────────────────────


class TestSetup:
    def test_setup_requires_transfer_learning_config(
        self, dataset_root: Path, tmp_path: Path
    ) -> None:
        """setup() must raise ValueError when transfer_learning is None."""
        from visionforge.blocks.transfer_learning import TransferLearningBlock

        config = ExperimentConfig.model_validate(
            {
                "name": "exp",
                "task": "binary",
                "block": "classification",
                "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
                "training": {"learning_rate": 0.01, "epochs": 1, "batch_size": 2},
                "data": {
                    "base_dir": str(dataset_root),
                    "num_workers": 0,
                    "pin_memory": False,
                },
            }
        )
        block = TransferLearningBlock()
        with pytest.raises(ValueError, match="transfer_learning"):
            block.setup(config)

    def test_valid_setup_does_not_raise(self, fe_config: ExperimentConfig) -> None:
        """setup() with a valid config must not raise."""
        from visionforge.blocks.transfer_learning import TransferLearningBlock

        block = TransferLearningBlock()
        block.setup(fe_config)

    def test_report_before_run_returns_empty(self, fe_config: ExperimentConfig) -> None:
        """report() before run() must return an empty dict."""
        from visionforge.blocks.transfer_learning import TransferLearningBlock

        block = TransferLearningBlock()
        block.setup(fe_config)
        assert block.report() == {}


# ── feature extraction freezing ───────────────────────────────────────────────


class TestFeatureExtractionFreezing:
    def test_all_backbone_params_frozen_after_setup(
        self, fe_config: ExperimentConfig
    ) -> None:
        """feature_extraction must freeze every backbone parameter."""
        from visionforge.blocks.transfer_learning import TransferLearningBlock

        model = TinyModel()
        block = TransferLearningBlock()
        block.setup(fe_config)
        block._apply_freeze(model)

        # layer1 and layer2 are backbone; fc is the head
        for name, param in model.named_parameters():
            if name.startswith("fc"):
                assert param.requires_grad, f"{name} should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_only_head_params_in_optimizer_for_feature_extraction(
        self, fe_config: ExperimentConfig
    ) -> None:
        """Optimizer for feature_extraction must contain only head parameters."""
        from visionforge.blocks.transfer_learning import TransferLearningBlock

        model = TinyModel()
        block = TransferLearningBlock()
        block.setup(fe_config)
        block._apply_freeze(model)
        optimizer = block._build_transfer_optimizer(model)

        # Single param group: only fc params
        assert len(optimizer.param_groups) == 1
        all_params = [p for g in optimizer.param_groups for p in g["params"]]
        # Must include fc.weight and fc.bias
        head_params = list(model.fc.parameters())
        assert len(all_params) == len(head_params)


class TestFrozenBatchNormStillRecalibrates:
    """The documented boundary of "frozen" (ADR-105).

    `requires_grad = False` stops the optimizer from moving a parameter. The
    BatchNorm running statistics are buffers written inside the forward pass, so
    a frozen module left in `train()` mode keeps updating them. That behaviour
    is deliberate; these tests exist so it cannot change unnoticed, because a
    paper saying "the backbone was frozen" claims something stronger than what
    actually happens.
    """

    @staticmethod
    def _frozen_resnet() -> nn.Module:
        from torchvision import models

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 4)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        return model

    @staticmethod
    def _train_a_few_batches(model: nn.Module) -> None:
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.1
        )
        model.train()
        for _ in range(3):
            optimizer.zero_grad()
            logits = model(torch.randn(4, 3, 64, 64))
            nn.functional.cross_entropy(logits, torch.randint(0, 4, (4,))).backward()
            optimizer.step()

    def test_only_the_head_weights_move(self) -> None:
        model = self._frozen_resnet()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}

        self._train_a_few_batches(model)

        moved = [
            n
            for n, p in model.named_parameters()
            if not torch.equal(p.detach(), before[n])
        ]
        assert moved == ["fc.weight", "fc.bias"]

    def test_the_normalization_statistics_do_move(self) -> None:
        """The part "frozen" does not cover."""
        model = self._frozen_resnet()
        before = {n: b.detach().clone() for n, b in model.named_buffers()}

        self._train_a_few_batches(model)

        moved = [
            n
            for n, b in model.named_buffers()
            if not torch.equal(b.detach(), before[n])
        ]

        # 20 BatchNorm modules × mean, variance, batch counter.
        assert len(moved) == 60
        assert any(n.endswith("running_mean") for n in moved)

    def test_eval_mode_would_hold_them_still(self) -> None:
        """The alternative ADR-105 declined: a different method, not a fix."""
        model = self._frozen_resnet()
        before = {n: b.detach().clone() for n, b in model.named_buffers()}

        model.eval()
        with torch.no_grad():
            for _ in range(3):
                model(torch.randn(4, 3, 64, 64))

        moved = [
            n
            for n, b in model.named_buffers()
            if not torch.equal(b.detach(), before[n])
        ]

        assert moved == []


# ── fine-tuning layer selection ───────────────────────────────────────────────


class TestFineTuningFreezing:
    def test_fine_tuning_unfreezes_all_when_no_layer_specified(
        self, ft_config: ExperimentConfig
    ) -> None:
        """fine_tuning with unfreeze_from_layer=None must leave all params trainable."""
        from visionforge.blocks.transfer_learning import TransferLearningBlock

        model = TinyModel()
        block = TransferLearningBlock()
        block.setup(ft_config)
        block._apply_freeze(model)

        for name, param in model.named_parameters():
            assert param.requires_grad, f"{name} should be trainable"

    def test_fine_tuning_freezes_layers_before_unfreeze_point(
        self, ft_partial_config: ExperimentConfig
    ) -> None:
        """Layers before unfreeze_from_layer must remain frozen in fine_tuning mode."""
        from visionforge.blocks.transfer_learning import TransferLearningBlock

        model = TinyModel()
        block = TransferLearningBlock()
        block.setup(ft_partial_config)
        block._apply_freeze(model)

        # layer1 is before layer2 — must be frozen
        for param in model.layer1.parameters():
            assert not param.requires_grad

        # layer2 and fc must be unfrozen
        for param in model.layer2.parameters():
            assert param.requires_grad
        for param in model.fc.parameters():
            assert param.requires_grad

    def test_fine_tuning_unknown_layer_raises(
        self, base_config_dict: dict[str, Any]
    ) -> None:
        """Specifying a layer name that doesn't exist must raise ValueError from _apply_freeze."""
        from visionforge.blocks.transfer_learning import TransferLearningBlock

        raw = {
            **base_config_dict,
            "transfer_learning": {
                "mode": "fine_tuning",
                "unfreeze_from_layer": "nonexistent_layer",
            },
        }
        config = ExperimentConfig.model_validate(raw)
        model = TinyModel()
        block = TransferLearningBlock()
        block.setup(config)
        with pytest.raises(ValueError, match="nonexistent_layer"):
            block._apply_freeze(model)


# ── optimizer param groups ─────────────────────────────────────────────────────


class TestOptimizerParamGroups:
    def test_fine_tuning_two_param_groups(
        self, ft_partial_config: ExperimentConfig
    ) -> None:
        """fine_tuning optimizer must have two param groups: backbone and head."""
        from visionforge.blocks.transfer_learning import TransferLearningBlock

        model = TinyModel()
        block = TransferLearningBlock()
        block.setup(ft_partial_config)
        block._apply_freeze(model)
        optimizer = block._build_transfer_optimizer(model)

        assert len(optimizer.param_groups) == 2

    def test_fine_tuning_backbone_lr_is_scaled(
        self, ft_partial_config: ExperimentConfig
    ) -> None:
        """Backbone param group LR must equal lr * backbone_lr_multiplier."""
        from visionforge.blocks.transfer_learning import TransferLearningBlock

        model = TinyModel()
        block = TransferLearningBlock()
        block.setup(ft_partial_config)
        block._apply_freeze(model)
        optimizer = block._build_transfer_optimizer(model)

        tl = ft_partial_config.transfer_learning
        assert tl is not None
        expected_backbone_lr = (
            ft_partial_config.training.learning_rate * tl.backbone_lr_multiplier
        )
        expected_head_lr = ft_partial_config.training.learning_rate

        lrs = {g["lr"] for g in optimizer.param_groups}
        assert expected_backbone_lr in lrs
        assert expected_head_lr in lrs

    def test_fine_tuning_full_unfreeze_two_param_groups(
        self, ft_config: ExperimentConfig
    ) -> None:
        """fine_tuning with no layer filter still produces two param groups."""
        from visionforge.blocks.transfer_learning import TransferLearningBlock

        model = TinyModel()
        block = TransferLearningBlock()
        block.setup(ft_config)
        block._apply_freeze(model)
        optimizer = block._build_transfer_optimizer(model)

        assert len(optimizer.param_groups) == 2

    def test_feature_extraction_single_param_group(
        self, fe_config: ExperimentConfig
    ) -> None:
        """feature_extraction optimizer must have exactly one param group."""
        from visionforge.blocks.transfer_learning import TransferLearningBlock

        model = TinyModel()
        block = TransferLearningBlock()
        block.setup(fe_config)
        block._apply_freeze(model)
        optimizer = block._build_transfer_optimizer(model)

        assert len(optimizer.param_groups) == 1


# ── run end-to-end ─────────────────────────────────────────────────────────────


class TestRun:
    def _run_mocked(self, config: ExperimentConfig, model: nn.Module) -> _TLBlock:
        from visionforge.blocks.transfer_learning import TransferLearningBlock

        block = TransferLearningBlock()
        block.setup(config)

        with (
            patch(
                "visionforge.blocks.transfer_learning.ModelFactory.create",
                return_value=model,
            ),
            patch("visionforge.blocks.transfer_learning.Trainer") as mock_trainer_cls,
            patch("visionforge.blocks.transfer_learning.Evaluator") as mock_eval_cls,
            patch("visionforge.blocks.transfer_learning.torch.load", return_value={}),
            patch("torch.nn.Module.load_state_dict"),
        ):
            mock_train_result = MagicMock()
            mock_train_result.best_epoch = 1
            mock_train_result.best_val_loss = 0.3
            mock_train_result.total_epochs = 1
            mock_train_result.model_path = (
                config.output.models_dir / config.name / "run" / "best_model.pth"
            )
            mock_trainer_cls.return_value.fit.return_value = mock_train_result

            mock_eval_result = MagicMock()
            mock_eval_result.accuracy = 0.9
            mock_eval_result.f1 = 0.88
            mock_eval_result.precision = 0.9
            mock_eval_result.recall = 0.87
            mock_eval_result.auc_roc = 0.95
            mock_eval_result.confusion_matrix = [[4, 1], [0, 5]]
            mock_eval_cls.return_value.evaluate.return_value = mock_eval_result

            block.run()

        return block

    def test_run_feature_extraction_completes(
        self, fe_config: ExperimentConfig
    ) -> None:
        """run() in feature_extraction mode must complete without raising."""
        self._run_mocked(fe_config, TinyModel())

    def test_run_fine_tuning_completes(self, ft_config: ExperimentConfig) -> None:
        """run() in fine_tuning mode must complete without raising."""
        self._run_mocked(ft_config, TinyModel())

    def test_trainer_receives_custom_optimizer(
        self, ft_partial_config: ExperimentConfig
    ) -> None:
        """Trainer.fit must be called with a two-group optimizer in fine_tuning."""
        from visionforge.blocks.transfer_learning import TransferLearningBlock

        model = TinyModel()
        block = TransferLearningBlock()
        block.setup(ft_partial_config)

        captured_optimizer: list[torch.optim.Optimizer] = []

        with (
            patch(
                "visionforge.blocks.transfer_learning.ModelFactory.create",
                return_value=model,
            ),
            patch("visionforge.blocks.transfer_learning.Trainer") as mock_trainer_cls,
            patch("visionforge.blocks.transfer_learning.Evaluator") as mock_eval_cls,
            patch("visionforge.blocks.transfer_learning.torch.load", return_value={}),
            patch("torch.nn.Module.load_state_dict"),
        ):
            mock_train_result = MagicMock()
            mock_train_result.best_epoch = 1
            mock_train_result.best_val_loss = 0.3
            mock_train_result.total_epochs = 1
            mock_train_result.model_path = (
                ft_partial_config.output.models_dir
                / ft_partial_config.name
                / "run"
                / "best_model.pth"
            )

            def fit_capture(  # noqa: ANN001
                m: Any,
                data: Any,
                optimizer: Any = None,
                progress_callback: Any = None,
                cancel_token: Any = None,
                resume_dir: Any = None,
            ) -> Any:
                if optimizer is not None:
                    captured_optimizer.append(optimizer)
                return mock_train_result

            mock_trainer_cls.return_value.fit.side_effect = fit_capture

            mock_eval_result = MagicMock()
            mock_eval_result.accuracy = 0.9
            mock_eval_result.f1 = 0.88
            mock_eval_result.precision = 0.9
            mock_eval_result.recall = 0.87
            mock_eval_result.auc_roc = 0.95
            mock_eval_result.confusion_matrix = [[4, 1], [0, 5]]
            mock_eval_cls.return_value.evaluate.return_value = mock_eval_result

            block.run()

        # Block must have built and applied the optimizer internally
        # The key check: block has _optimizer with 2 groups
        assert hasattr(block, "_optimizer")
        assert len(block._optimizer.param_groups) == 2  # type: ignore[union-attr]


# ── report structure ──────────────────────────────────────────────────────────


class TestReport:
    def _run_and_report(self, config: ExperimentConfig) -> dict[str, Any]:
        from visionforge.blocks.transfer_learning import TransferLearningBlock

        block = TransferLearningBlock()
        block.setup(config)

        with (
            patch(
                "visionforge.blocks.transfer_learning.ModelFactory.create",
                return_value=TinyModel(),
            ),
            patch("visionforge.blocks.transfer_learning.Trainer") as mock_trainer_cls,
            patch("visionforge.blocks.transfer_learning.Evaluator") as mock_eval_cls,
            patch("visionforge.blocks.transfer_learning.torch.load", return_value={}),
            patch("torch.nn.Module.load_state_dict"),
        ):
            mock_train_result = MagicMock()
            mock_train_result.best_epoch = 1
            mock_train_result.best_val_loss = 0.3
            mock_train_result.total_epochs = 1
            mock_train_result.model_path = (
                config.output.models_dir / config.name / "run" / "best_model.pth"
            )
            mock_trainer_cls.return_value.fit.return_value = mock_train_result

            mock_eval_result = MagicMock()
            mock_eval_result.accuracy = 0.9
            mock_eval_result.f1 = 0.88
            mock_eval_result.precision = 0.9
            mock_eval_result.recall = 0.87
            mock_eval_result.auc_roc = 0.95
            mock_eval_result.confusion_matrix = [[4, 1], [0, 5]]
            mock_eval_cls.return_value.evaluate.return_value = mock_eval_result

            block.run()

        return block.report()

    def test_report_has_mode(self, fe_config: ExperimentConfig) -> None:
        """report() must include the transfer learning mode."""
        result = self._run_and_report(fe_config)
        assert "mode" in result

    def test_report_has_frozen_layers(self, fe_config: ExperimentConfig) -> None:
        """report() must include frozen_layers list."""
        result = self._run_and_report(fe_config)
        assert "frozen_layers" in result

    def test_report_has_lr_groups(self, fe_config: ExperimentConfig) -> None:
        """report() must include lr_groups list."""
        result = self._run_and_report(fe_config)
        assert "lr_groups" in result

    def test_report_has_train_section(self, fe_config: ExperimentConfig) -> None:
        """report() must include a train section with standard keys."""
        result = self._run_and_report(fe_config)
        assert "train" in result
        assert "best_epoch" in result["train"]
        assert "best_val_loss" in result["train"]

    def test_report_has_eval_section(self, fe_config: ExperimentConfig) -> None:
        """report() must include an eval section with accuracy."""
        result = self._run_and_report(fe_config)
        assert "eval" in result
        assert "accuracy" in result["eval"]

    def test_report_mode_is_feature_extraction(
        self, fe_config: ExperimentConfig
    ) -> None:
        """report mode must match config mode for feature_extraction."""
        result = self._run_and_report(fe_config)
        assert result["mode"] == "feature_extraction"

    def test_report_mode_is_fine_tuning(self, ft_config: ExperimentConfig) -> None:
        """report mode must match config mode for fine_tuning."""
        result = self._run_and_report(ft_config)
        assert result["mode"] == "fine_tuning"


# ── run.json artifact ─────────────────────────────────────────────────────────


class TestRunJson:
    def test_run_json_written_with_tl_metadata(
        self, fe_config: ExperimentConfig, tmp_path: Path
    ) -> None:
        """run() must write run.json that includes transfer_learning metadata."""
        from visionforge.blocks.transfer_learning import TransferLearningBlock

        block = TransferLearningBlock()
        block.setup(fe_config)

        run_dir = (
            fe_config.output.models_dir / fe_config.name / "20240101_000000_000000"
        )
        run_dir.mkdir(parents=True)
        run_json_path = run_dir / "run.json"
        run_json_path.write_text(
            json.dumps(
                {
                    "id": "tl_test_run",
                    "metrics": {},
                    "artifacts": {"model": "", "graphics": [], "report": None},
                    "config": {},
                }
            ),
            encoding="utf-8",
        )

        mock_train_result = MagicMock()
        mock_train_result.best_epoch = 1
        mock_train_result.best_val_loss = 0.3
        mock_train_result.total_epochs = 1
        mock_train_result.model_path = run_dir / "best_model.pth"

        with (
            patch(
                "visionforge.blocks.transfer_learning.ModelFactory.create",
                return_value=TinyModel(),
            ),
            patch("visionforge.blocks.transfer_learning.Trainer") as mock_trainer_cls,
            patch("visionforge.blocks.transfer_learning.Evaluator") as mock_eval_cls,
            patch("visionforge.blocks.transfer_learning.torch.load", return_value={}),
            patch("torch.nn.Module.load_state_dict"),
        ):
            mock_trainer_cls.return_value.fit.return_value = mock_train_result

            mock_eval_result = MagicMock()
            mock_eval_result.accuracy = 0.9
            mock_eval_result.f1 = 0.88
            mock_eval_result.precision = 0.9
            mock_eval_result.recall = 0.87
            mock_eval_result.auc_roc = 0.95
            mock_eval_result.confusion_matrix = [[4, 1], [0, 5]]
            mock_eval_cls.return_value.evaluate.return_value = mock_eval_result

            block.run()

        updated = json.loads(run_json_path.read_text(encoding="utf-8"))
        assert "transfer_learning" in updated
        assert "mode" in updated["transfer_learning"]
        assert "frozen_layers" in updated["transfer_learning"]


# ── registry ──────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_registry_discovers_transfer_learning_block(self) -> None:
        """BlockRegistry must include TransferLearningBlock after import."""
        import visionforge.blocks.transfer_learning  # noqa: F401
        from visionforge.blocks.registry import BlockRegistry

        registry = BlockRegistry.discover()
        assert "TransferLearningBlock" in registry


# ── config exported in __all__ ────────────────────────────────────────────────


class TestConfigExports:
    def test_transfer_learning_config_in_all(self) -> None:
        """TransferLearningConfig must be in visionforge.utils.config.__all__."""
        import visionforge.utils.config as cfg_module

        assert "TransferLearningConfig" in cfg_module.__all__


# ── live progress ─────────────────────────────────────────────────────────────


class TestProgressStreaming:
    """The block trained fine but streamed nothing: the GUI's progress bar sat
    dead for a whole transfer-learning run while `routes.py` carried a comment
    saying the block "doesn't take a progress_callback yet". Found by running
    the real matrix on a real dataset, so it is pinned here."""

    def test_epoch_events_reach_the_callback(self, ft_config: ExperimentConfig) -> None:
        from visionforge.blocks.transfer_learning import TransferLearningBlock

        events: list[dict[str, Any]] = []
        block = TransferLearningBlock()
        block.setup(ft_config)
        block._progress_callback = events.append
        block.run()

        kinds = [e.get("event") for e in events]
        assert "epoch_end" in kinds, f"no epoch_end streamed, got {kinds}"

    def test_the_gui_wires_the_block_to_the_event_pump(self) -> None:
        """The block accepting a callback is only half of it — routes.py has to
        be the one attaching it, which an isinstance check used to exclude."""
        import inspect

        from visionforge.gui.api import routes

        source = inspect.getsource(routes._execute_experiment)
        assert "TransferLearningBlock" in source


# ── bootstrap confidence intervals (ADR-074) ──────────────────────────────────


class TestBootstrapConfidenceIntervals:
    """Transfer learning reports the same intervals a plain run does.

    Deliberately a *real* run — the mocked-Evaluator tests above cannot cover
    this: a MagicMock's y_true reads as a zero-length array, so the interval
    code correctly declines and the path is never exercised.
    """

    @pytest.fixture
    def wide_config(self, tmp_path: Path) -> ExperimentConfig:
        from PIL import Image

        rng = np.random.default_rng(1)
        root = tmp_path / "data"
        for split, per_class in (("train", 8), ("val", 8), ("test", 16)):
            for cls in ("class_a", "class_b"):
                folder = root / split / cls
                folder.mkdir(parents=True)
                for i in range(per_class):
                    pixels = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
                    Image.fromarray(pixels).save(folder / f"image_{i}.png")

        return ExperimentConfig.model_validate(
            {
                "name": "tl_ci_test",
                "task": "binary",
                "block": "transfer_learning",
                "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
                "training": {
                    "learning_rate": 0.01,
                    "epochs": 1,
                    "batch_size": 4,
                    "seed": 3,
                },
                "data": {
                    "base_dir": str(root),
                    "num_workers": 0,
                    "pin_memory": False,
                    "transforms": {"image_size": 32},
                },
                "output": {
                    "models_dir": str(tmp_path / "models"),
                    "graphics_dir": str(tmp_path / "graphics"),
                    "logs_dir": str(tmp_path / "logs"),
                    "reports_dir": str(tmp_path / "reports"),
                },
                "transfer_learning": {"mode": "feature_extraction"},
            }
        )

    def test_run_json_and_report_carry_intervals(
        self, wide_config: ExperimentConfig
    ) -> None:
        from visionforge.blocks.transfer_learning import TransferLearningBlock

        block = TransferLearningBlock()
        block.setup(wide_config)
        with patch(
            "visionforge.blocks.transfer_learning.ModelFactory.create",
            return_value=TinyModel(),
        ):
            block.run()

        run_json = next(
            (wide_config.output.models_dir / wide_config.name).glob("*/run.json")
        )
        data = json.loads(run_json.read_text(encoding="utf-8"))

        assert data["metric_cis"]["accuracy"]["n_samples"] == 32
        assert (
            block.report()["eval"]["confidence_intervals"]["accuracy"]
            == data["metric_cis"]["accuracy"]
        )
