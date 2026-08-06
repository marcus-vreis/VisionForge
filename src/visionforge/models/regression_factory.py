"""Build a CNN with a regression head from a RegressionModelConfig.

Reuses the shared backbone builders and head-swap helper from
``models.factory``; the only difference from classification is that the final
linear layer emits ``num_targets`` continuous outputs (no activation — the raw
values are the predictions, and the regression loss consumes them directly).
"""

from __future__ import annotations

import torch.nn as nn

from visionforge.models.factory import (
    build_backbone,
    load_local_weights,
    replace_final_layer,
)
from visionforge.models.registry import build_custom_model
from visionforge.models.timm_source import build_timm_model
from visionforge.utils.regression_config import RegressionModelConfig


class RegressionModelFactory:
    """Instantiates a CNN regressor from a RegressionModelConfig."""

    @staticmethod
    def create(config: RegressionModelConfig) -> nn.Module:
        """Build a backbone with a ``num_targets``-wide linear head.

        Returns:
            nn.Module emitting ``config.num_targets`` raw continuous outputs —
            either a torchvision backbone with a swapped head, or a user-registered
            custom model when ``config.custom_model`` is set (ADR-048/049).
        """
        if config.custom_model is not None:
            model = build_custom_model(
                config.custom_model, num_outputs=config.num_targets
            )
            if config.weights_path is not None:
                load_local_weights(model, config.weights_path)
            return model

        if config.timm_model is not None:
            model = build_timm_model(
                config.timm_model,
                num_outputs=config.num_targets,
                pretrained=config.pretrained,
            )
            if config.weights_path is not None:
                load_local_weights(model, config.weights_path)
            return model

        use_imagenet = config.pretrained and config.weights_path is None
        model = build_backbone(config.name, use_imagenet=use_imagenet)
        replace_final_layer(model, config.name, config.num_targets)

        if config.weights_path is not None:
            load_local_weights(model, config.weights_path)

        return model


__all__ = ["RegressionModelFactory"]
