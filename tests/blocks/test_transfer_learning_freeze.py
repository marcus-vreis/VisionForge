"""Feature extraction has to freeze the backbone on every architecture.

It did not. "The last named child" is the head on ResNet (`fc`) and on the
attention models (`heads`, `head`), but on VGG and AlexNet that child is a
`classifier` block holding three Linear layers — 119M and 54M parameters. The
mode whose entire promise is that the backbone does not move was leaving 89%
and 96% of the network trainable (ADR-101).
"""

from __future__ import annotations

import pytest

from visionforge.blocks.transfer_learning import TransferLearningBlock
from visionforge.models.factory import ModelFactory, final_linear
from visionforge.utils.config import ExperimentConfig

# Every architecture the classification task offers.
ARCHITECTURES = [
    "resnet18",
    "resnet50",
    "efficientnet_b1",
    "vgg16",
    "alexnet",
    "vit_b_16",
    "swin_t",
    "convnext_tiny",
]


def _frozen_model(arch: str, mode: str = "feature_extraction"):
    cfg = ExperimentConfig.model_validate(
        {
            "name": "freeze",
            "task": "multiclass",
            "block": "transfer_learning",
            "model": {"name": arch, "num_classes": 4, "pretrained": False},
            "training": {"learning_rate": 0.001, "epochs": 1},
            "data": {"base_dir": "."},
            "output": {"models_dir": "."},
            "device": {"kind": "cpu"},
            "transfer_learning": {"mode": mode, "backbone_lr_multiplier": 0.1},
        }
    )
    block = TransferLearningBlock()
    block.setup(cfg)
    model = ModelFactory.create(cfg.model)
    block._apply_freeze(model)
    return model, block, cfg


class TestFeatureExtractionFreezesTheBackbone:
    @pytest.mark.parametrize("arch", ARCHITECTURES)
    def test_only_the_head_is_trainable(self, arch: str) -> None:
        model, _block, _cfg = _frozen_model(arch)

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # The head of a 4-class model is tiny next to any backbone here; the
        # bug being guarded against left 89% trainable on vgg16.
        assert trainable / total < 0.01, (
            f"{arch}: {trainable:,} of {total:,} parameters trainable "
            f"({100 * trainable / total:.2f}%) — the backbone is not frozen"
        )

    @pytest.mark.parametrize("arch", ARCHITECTURES)
    def test_the_trainable_parameters_are_exactly_the_head(self, arch: str) -> None:
        model, _block, _cfg = _frozen_model(arch)

        head = final_linear(model)
        assert head is not None
        head_ids = {id(p) for p in head.parameters()}
        trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}

        assert trainable_ids == head_ids

    @pytest.mark.parametrize("arch", ["vgg16", "alexnet"])
    def test_the_multi_linear_classifiers_are_the_regression_case(
        self, arch: str
    ) -> None:
        """These two are why the rule changed; name them so it stays fixed."""
        model, _block, _cfg = _frozen_model(arch)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 4096 -> 4 plus bias. The old rule kept the two 4096-wide layers too.
        assert trainable == 4096 * 4 + 4


class TestFineTuningSplitsTheLearningRate:
    @pytest.mark.parametrize("arch", ["resnet18", "vgg16", "vit_b_16"])
    def test_backbone_gets_the_reduced_rate(self, arch: str) -> None:
        model, block, cfg = _frozen_model(arch, mode="fine_tuning")

        optimizer = block._build_transfer_optimizer(model)
        rates = [group["lr"] for group in optimizer.param_groups]

        assert len(optimizer.param_groups) == 2
        assert rates == [
            cfg.training.learning_rate * 0.1,
            cfg.training.learning_rate,
        ]

    def test_fine_tuning_leaves_everything_trainable(self) -> None:
        model, _block, _cfg = _frozen_model("resnet18", mode="fine_tuning")

        assert all(p.requires_grad for p in model.parameters())
