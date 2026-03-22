from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from visionforge.models.factory import ModelFactory
from visionforge.utils.config import ModelConfig


def make_model_config(**overrides: Any) -> ModelConfig:
    defaults: dict[str, Any] = {
        "name": "resnet18",
        "num_classes": 1,
        "pretrained": False,
    }
    defaults.update(overrides)
    return ModelConfig.model_validate(defaults)


class TestModelFactoryClassifierReplacement:
    @pytest.mark.parametrize("name", ["resnet18", "resnet34", "resnet50", "resnet101"])
    def test_resnet_output_size(self, name: str) -> None:
        """ResNet classifiers must output num_classes neurons."""
        config = make_model_config(name=name, num_classes=3)
        model = ModelFactory.create(config)
        out = model(torch.randn(1, 3, 64, 64))
        assert out.shape == (1, 3)

    @pytest.mark.parametrize("name", ["efficientnet_b1", "efficientnet_b7"])
    def test_efficientnet_output_size(self, name: str) -> None:
        """EfficientNet classifiers must output num_classes neurons."""
        config = make_model_config(name=name, num_classes=5)
        model = ModelFactory.create(config)
        out = model(torch.randn(1, 3, 64, 64))
        assert out.shape == (1, 5)

    @pytest.mark.parametrize("name", ["vgg16", "vgg19", "alexnet"])
    def test_vgg_alexnet_output_size(self, name: str) -> None:
        """VGG and AlexNet classifiers must output num_classes neurons."""
        config = make_model_config(name=name, num_classes=2)
        model = ModelFactory.create(config)
        out = model(torch.randn(1, 3, 64, 64))
        assert out.shape == (1, 2)

    def test_binary_output_single_neuron(self) -> None:
        """Binary config (num_classes=1) must produce a single-neuron output."""
        config = make_model_config(name="resnet18", num_classes=1)
        model = ModelFactory.create(config)
        out = model(torch.randn(1, 3, 64, 64))
        assert out.shape == (1, 1)


class TestModelFactoryWeights:
    def test_pretrained_false_produces_module(self) -> None:
        """ModelFactory must return an nn.Module when pretrained=False."""
        config = make_model_config(pretrained=False)
        model = ModelFactory.create(config)
        assert isinstance(model, nn.Module)

    def test_local_weights_loaded(self, tmp_path: Path) -> None:
        """When weights_path is set, local weights must be loaded."""
        # Save a minimal valid state dict to a file.
        config = make_model_config(name="resnet18", num_classes=1, pretrained=False)
        real_model = ModelFactory.create(config)
        weights_file = tmp_path / "weights.pth"
        torch.save(real_model.state_dict(), weights_file)

        config_with_path = make_model_config(
            name="resnet18",
            num_classes=1,
            pretrained=False,
            weights_path=weights_file,
        )
        loaded_model = ModelFactory.create(config_with_path)
        assert isinstance(loaded_model, nn.Module)

    def test_local_weights_skip_imagenet(self, tmp_path: Path) -> None:
        """When weights_path is set, ImageNet weights must not be fetched."""
        config = make_model_config(name="resnet18", num_classes=1, pretrained=False)
        real_model = ModelFactory.create(config)
        weights_file = tmp_path / "weights.pth"
        torch.save(real_model.state_dict(), weights_file)

        config_with_path = make_model_config(
            name="resnet18",
            num_classes=1,
            pretrained=True,
            weights_path=weights_file,
        )

        with patch("torchvision.models.resnet18") as mock_builder:
            mock_model = MagicMock(spec=nn.Module)
            mock_model.fc = nn.Linear(512, 1)
            mock_builder.return_value = mock_model
            ModelFactory.create(config_with_path)
            # weights=None should have been passed (not "IMAGENET1K_V1")
            _, kwargs = mock_builder.call_args
            assert kwargs.get("weights") is None
