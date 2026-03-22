from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn
import torchvision.models as tv_models

from visionforge.utils.config import ModelConfig


class ModelFactory:
    """Instantiates a CNN from a ModelConfig."""

    @staticmethod
    def create(config: ModelConfig) -> nn.Module:
        """Build and return a model ready for training.

        Args:
            config: model configuration.

        Returns:
            nn.Module with the final classifier replaced to match num_classes.
        """
        model = ModelFactory._build_backbone(config)
        ModelFactory._replace_classifier(model, config.name, config.num_classes)

        if config.weights_path is not None:
            ModelFactory._load_local_weights(model, config)

        return model

    @staticmethod
    def _build_backbone(config: ModelConfig) -> nn.Module:
        """Load the backbone architecture, optionally with ImageNet weights."""
        # Use ImageNet weights only when pretrained=True and no local path is given.
        use_imagenet = config.pretrained and config.weights_path is None
        weights = "DEFAULT" if use_imagenet else None

        builders: dict[str, Callable[..., nn.Module]] = {
            "resnet18": tv_models.resnet18,
            "resnet34": tv_models.resnet34,
            "resnet50": tv_models.resnet50,
            "resnet101": tv_models.resnet101,
            "efficientnet_b1": tv_models.efficientnet_b1,
            "efficientnet_b7": tv_models.efficientnet_b7,
            "vgg16": tv_models.vgg16,
            "vgg19": tv_models.vgg19,
            "alexnet": tv_models.alexnet,
        }
        return builders[config.name](weights=weights)

    @staticmethod
    def _replace_classifier(model: nn.Module, name: str, num_classes: int) -> None:
        """Swap the final linear layer to match num_classes."""
        if name.startswith("resnet"):
            resnet = cast(tv_models.ResNet, model)
            resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        elif name.startswith("efficientnet"):
            eff = cast(
                nn.Sequential,
                model.classifier if hasattr(model, "classifier") else model,
            )  # type: ignore[union-attr]
            old = cast(nn.Linear, eff[1])
            eff[1] = nn.Linear(old.in_features, num_classes)
        elif name.startswith("vgg") or name == "alexnet":
            clf = cast(
                nn.Sequential,
                model.classifier if hasattr(model, "classifier") else model,
            )  # type: ignore[union-attr]
            old = cast(nn.Linear, clf[6])
            clf[6] = nn.Linear(old.in_features, num_classes)

    @staticmethod
    def _load_local_weights(model: nn.Module, config: ModelConfig) -> None:
        """Load weights from a local .pth file into the model."""
        from loguru import logger

        state_dict = torch.load(
            str(config.weights_path), map_location="cpu", weights_only=True
        )
        result = model.load_state_dict(state_dict, strict=False)  # type: ignore[arg-type]
        if result.missing_keys:
            logger.warning("Local weights missing keys: {}", result.missing_keys)
        if result.unexpected_keys:
            logger.warning("Local weights unexpected keys: {}", result.unexpected_keys)


__all__ = ["ModelFactory"]
