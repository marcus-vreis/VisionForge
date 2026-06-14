from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torchvision.models as tv_models

from visionforge.models.registry import build_custom_model
from visionforge.models.timm_source import build_timm_model
from visionforge.utils.config import ModelConfig


def build_backbone(name: str, *, use_imagenet: bool) -> nn.Module:
    """Load a backbone architecture, optionally with ImageNet weights.

    Shared by every CNN-headed task (classification + regression). Builders are
    resolved at call time (not module import) so tests can patch the underlying
    ``torchvision.models.*`` constructors.
    """
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
    weights = "DEFAULT" if use_imagenet else None
    return builders[name](weights=weights)


def replace_final_layer(model: nn.Module, name: str, num_outputs: int) -> None:
    """Swap the final linear layer to emit ``num_outputs`` values.

    Works for both classification (``num_classes`` logits) and regression
    (``num_targets`` continuous outputs) — the layer is identical; only the
    downstream loss differs. No activation is appended.
    """
    if name.startswith("resnet"):
        resnet = cast(tv_models.ResNet, model)
        resnet.fc = nn.Linear(resnet.fc.in_features, num_outputs)
    elif name.startswith("efficientnet"):
        eff = cast(
            nn.Sequential,
            model.classifier if hasattr(model, "classifier") else model,
        )  # type: ignore[union-attr]
        old = cast(nn.Linear, eff[1])
        eff[1] = nn.Linear(old.in_features, num_outputs)
    elif name.startswith("vgg") or name == "alexnet":
        clf = cast(
            nn.Sequential,
            model.classifier if hasattr(model, "classifier") else model,
        )  # type: ignore[union-attr]
        old = cast(nn.Linear, clf[6])
        clf[6] = nn.Linear(old.in_features, num_outputs)


def load_local_weights(model: nn.Module, weights_path: Path) -> None:
    """Load weights from a local .pth file into the model (non-strict)."""
    from loguru import logger

    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    result = model.load_state_dict(state_dict, strict=False)  # type: ignore[arg-type]
    if result.missing_keys:
        logger.warning("Local weights missing keys: {}", result.missing_keys)
    if result.unexpected_keys:
        logger.warning("Local weights unexpected keys: {}", result.unexpected_keys)


class ModelFactory:
    """Instantiates a CNN from a ModelConfig."""

    @staticmethod
    def create(config: ModelConfig) -> nn.Module:
        """Build and return a model ready for training.

        Args:
            config: model configuration.

        Returns:
            nn.Module emitting ``num_classes`` logits — either a torchvision
            backbone with a swapped head, or a user-registered custom model when
            ``config.custom_model`` is set (ADR-048).
        """
        if config.custom_model is not None:
            model = build_custom_model(
                config.custom_model, num_outputs=config.num_classes
            )
            if config.weights_path is not None:
                load_local_weights(model, config.weights_path)
            return model

        if config.timm_model is not None:
            model = build_timm_model(
                config.timm_model,
                num_outputs=config.num_classes,
                pretrained=config.pretrained,
            )
            if config.weights_path is not None:
                load_local_weights(model, config.weights_path)
            return model

        # Use ImageNet weights only when pretrained=True and no local path is given.
        use_imagenet = config.pretrained and config.weights_path is None
        model = build_backbone(config.name, use_imagenet=use_imagenet)
        replace_final_layer(model, config.name, config.num_classes)

        if config.weights_path is not None:
            load_local_weights(model, config.weights_path)

        return model


__all__ = [
    "ModelFactory",
    "build_backbone",
    "replace_final_layer",
    "load_local_weights",
]
