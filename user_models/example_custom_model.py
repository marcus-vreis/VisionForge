"""Example drop-in custom model — copy as a template, or delete.

Registers ``example_tiny_cnn``: a minimal conv net that adapts to any input size
(global average pool) and emits ``num_classes`` logits. Use it from a config with
``model.custom_model: example_tiny_cnn``. See user_models/README.md and ADR-048.
"""

from __future__ import annotations

import torch.nn as nn

from visionforge.models.registry import register_model


@register_model("example_tiny_cnn")
def build_example_tiny_cnn(num_classes: int) -> nn.Module:
    """A tiny 3-block CNN classifier — illustrative, not competitive."""
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, num_classes),
    )
