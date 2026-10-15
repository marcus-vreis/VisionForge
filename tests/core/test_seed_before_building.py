"""The same config and seed build the same model, whatever ran before (ADR-121).

The seed-variance pilot trained resnet18 at seed 0 twice in one process and got
0.774 and 0.681 test accuracy: the new classifier head was drawn from the global
RNG when the block built the model, before the trainer seeded anything. Here the
"model" is a stand-in that draws its weights from the global RNG at creation, as a
new head does, and the RNG is left in two different states before two builds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from visionforge.core.trainer import seed_before_building


class _Stop(Exception):
    """Raised by the stand-in factory once the build has been observed."""


def _drawing_factory(seen: list[torch.Tensor]) -> Any:
    def create(model_config: Any) -> Any:
        seen.append(torch.randn(8))  # what a freshly initialised head draws
        raise _Stop

    return create


def _two_builds(run: Any) -> list[torch.Tensor]:
    """Build twice with the global RNG scrambled differently each time."""
    seen: list[torch.Tensor] = []
    for leftover in (123, 999):
        torch.manual_seed(leftover)
        torch.randn(1000)  # an earlier run consumed some randomness
        with pytest.raises(_Stop):
            run(seen)
    return seen


def test_the_helper_seeds_from_the_training_block() -> None:
    class _Training:
        seed = 7
        deterministic = True

    seed_before_building(_Training())
    first = torch.randn(4)
    torch.manual_seed(0)
    seed_before_building(_Training())

    assert torch.equal(first, torch.randn(4))


def test_classification_builds_the_same_model_for_the_same_seed(tmp_path: Path) -> None:
    from visionforge.blocks.classification import ClassificationBlock
    from visionforge.utils.config import ExperimentConfig

    config = ExperimentConfig.model_validate(
        {
            "name": "seed_build",
            "task": "multiclass",
            "model": {"name": "resnet18", "num_classes": 4, "pretrained": False},
            "training": {"seed": 0},
            "data": {"base_dir": str(tmp_path)},
            "output": {"models_dir": str(tmp_path / "models")},
        }
    )

    def run(seen: list[torch.Tensor]) -> None:
        block = ClassificationBlock()
        block.setup(config)
        with patch(
            "visionforge.blocks.classification.ModelFactory.create",
            _drawing_factory(seen),
        ):
            block.run()

    first, second = _two_builds(run)
    assert torch.equal(first, second)


def test_regression_builds_the_same_model_for_the_same_seed(tmp_path: Path) -> None:
    from visionforge.blocks.regression import RegressionBlock
    from visionforge.utils.regression_config import RegressionConfig

    config = RegressionConfig.model_validate(
        {
            "name": "seed_build",
            "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
            "data": {"base_dir": str(tmp_path), "target_columns": ["target"]},
            "training": {"seed": 0},
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )

    def run(seen: list[torch.Tensor]) -> None:
        block = RegressionBlock()
        block.setup(config)
        with patch(
            "visionforge.blocks.regression.RegressionModelFactory.create",
            _drawing_factory(seen),
        ):
            block.run()

    first, second = _two_builds(run)
    assert torch.equal(first, second)


def test_different_seeds_still_build_different_models(tmp_path: Path) -> None:
    """The fix pins the build to the seed; it must not pin every build together."""

    class _Training:
        deterministic = True

        def __init__(self, seed: int) -> None:
            self.seed = seed

    seed_before_building(_Training(0))
    a = torch.randn(8)
    seed_before_building(_Training(1))
    b = torch.randn(8)

    assert not torch.equal(a, b)
