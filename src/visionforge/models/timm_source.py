"""timm as an extra model source (ADR-051).

``timm`` exposes hundreds of pretrained architectures. ``timm.create_model`` builds
one with a fresh head sized to ``num_classes``, so it serves classification
(``num_classes``) and regression (``num_targets``) directly — the same
output-dimension contract as the custom registry. timm is an optional dependency
(the ``timm`` extra); the import is lazy so VisionForge runs without it installed.
"""

from __future__ import annotations

import torch.nn as nn


def build_timm_model(name: str, *, num_outputs: int, pretrained: bool) -> nn.Module:
    """Create a timm model whose head emits ``num_outputs`` values.

    Raises:
        ImportError: if the optional ``timm`` dependency is not installed.
    """
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - only without the timm extra
        raise ImportError(
            "timm ships with VisionForge and is missing — the install "
            'looks damaged: pip install --force-reinstall "visionforge-studio".'
        ) from exc

    model: nn.Module = timm.create_model(
        name, pretrained=pretrained, num_classes=num_outputs
    )
    return model


__all__ = ["build_timm_model"]
