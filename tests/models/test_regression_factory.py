from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from visionforge.models.regression_factory import RegressionModelFactory
from visionforge.utils.regression_config import RegressionModelConfig


def make_model_config(**overrides: Any) -> RegressionModelConfig:
    defaults: dict[str, Any] = {
        "name": "resnet18",
        "num_targets": 1,
        "pretrained": False,
    }
    defaults.update(overrides)
    return RegressionModelConfig.model_validate(defaults)


class TestRegressionHead:
    @pytest.mark.parametrize("name", ["resnet18", "resnet34", "resnet50", "resnet101"])
    def test_resnet_output_size(self, name: str) -> None:
        model = RegressionModelFactory.create(
            make_model_config(name=name, num_targets=2)
        )
        out = model(torch.randn(1, 3, 64, 64))
        assert out.shape == (1, 2)

    @pytest.mark.parametrize("name", ["efficientnet_b1", "efficientnet_b7"])
    def test_efficientnet_output_size(self, name: str) -> None:
        model = RegressionModelFactory.create(
            make_model_config(name=name, num_targets=3)
        )
        out = model(torch.randn(1, 3, 64, 64))
        assert out.shape == (1, 3)

    @pytest.mark.parametrize("name", ["vgg16", "vgg19", "alexnet"])
    def test_vgg_alexnet_output_size(self, name: str) -> None:
        model = RegressionModelFactory.create(
            make_model_config(name=name, num_targets=1)
        )
        out = model(torch.randn(1, 3, 64, 64))
        assert out.shape == (1, 1)

    def test_single_target_default(self) -> None:
        model = RegressionModelFactory.create(make_model_config())
        out = model(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 1)


class TestRegressionFactoryWeights:
    def test_pretrained_false_produces_module(self) -> None:
        model = RegressionModelFactory.create(make_model_config(pretrained=False))
        assert isinstance(model, nn.Module)

    def test_local_weights_loaded(self, tmp_path: Path) -> None:
        # Save the regressor's own state dict, then reload it through the factory.
        base = RegressionModelFactory.create(make_model_config(name="resnet18"))
        weights = tmp_path / "reg.pth"
        torch.save(base.state_dict(), weights)
        loaded = RegressionModelFactory.create(
            make_model_config(name="resnet18", weights_path=weights)
        )
        assert isinstance(loaded, nn.Module)
        out = loaded(torch.randn(1, 3, 64, 64))
        assert out.shape == (1, 1)
