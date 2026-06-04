"""Build semantic-segmentation models from a SegmentationModelConfig (Phase 8).

Two model sources:

- **torchvision families** (DeepLabV3 / FCN / LR-ASPP) — built with the head sized
  to ``num_classes`` and a (optionally ImageNet-pretrained) backbone. When
  ``pretrained`` is False, ``weights_backbone=None`` so construction never
  downloads weights (CPU CI / tests). These return an ``OrderedDict`` with an
  ``"out"`` logits tensor (torchvision convention).
- **U-Net** — a hand-rolled encoder/decoder (not in torchvision). Returns a plain
  ``[N, num_classes, H, W]`` logits tensor; ``pretrained`` is ignored (no standard
  U-Net weights exist).

``segmentation_logits`` normalizes either output shape to a plain tensor so the
trainer (brick 4) is model-agnostic. See PHASE8_SEGMENTATION_PLAN.md.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
)
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    fcn_resnet50,
    fcn_resnet101,
    lraspp_mobilenet_v3_large,
)

from visionforge.models.factory import load_local_weights
from visionforge.utils.segmentation_config import SegmentationModelConfig

# name -> (builder, backbone weights enum, supports aux_loss kwarg)
_TORCHVISION_SEG: dict[str, tuple[Callable[..., nn.Module], Any, bool]] = {
    "deeplabv3_resnet50": (deeplabv3_resnet50, ResNet50_Weights, True),
    "deeplabv3_resnet101": (deeplabv3_resnet101, ResNet101_Weights, True),
    "deeplabv3_mobilenet_v3_large": (
        deeplabv3_mobilenet_v3_large,
        MobileNet_V3_Large_Weights,
        True,
    ),
    "fcn_resnet50": (fcn_resnet50, ResNet50_Weights, True),
    "fcn_resnet101": (fcn_resnet101, ResNet101_Weights, True),
    "lraspp_mobilenet_v3_large": (
        lraspp_mobilenet_v3_large,
        MobileNet_V3_Large_Weights,
        False,
    ),
}


def segmentation_logits(output: Any) -> torch.Tensor:
    """Return the ``[N, C, H, W]`` logits from a segmentation model output.

    torchvision segmentation models return an ``OrderedDict`` keyed ``"out"``;
    the hand-rolled U-Net returns a plain tensor. This normalizes both.
    """
    if isinstance(output, Mapping):
        return cast(torch.Tensor, output["out"])
    return cast(torch.Tensor, output)


class _DoubleConv(nn.Module):
    """(conv → BN → ReLU) × 2, the U-Net building block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.block(x)
        return out


class UNet(nn.Module):
    """Canonical 4-level U-Net with skip connections.

    Decoder steps realign to the encoder feature size with bilinear interpolation,
    so arbitrary (even non-power-of-two) input resolutions are handled and the
    output matches the input H×W.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2, base: int = 64):
        super().__init__()
        self.inc = _DoubleConv(in_channels, base)
        self.enc1 = _DoubleConv(base, base * 2)
        self.enc2 = _DoubleConv(base * 2, base * 4)
        self.enc3 = _DoubleConv(base * 4, base * 8)
        self.bottleneck = _DoubleConv(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)

        self.up3 = _DoubleConv(base * 16 + base * 8, base * 8)
        self.up2 = _DoubleConv(base * 8 + base * 4, base * 4)
        self.up1 = _DoubleConv(base * 4 + base * 2, base * 2)
        self.up0 = _DoubleConv(base * 2 + base, base)
        self.outc = nn.Conv2d(base, num_classes, 1)

    @staticmethod
    def _up_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c0 = self.inc(x)
        c1 = self.enc1(self.pool(c0))
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        b = self.bottleneck(self.pool(c3))

        d3 = self.up3(self._up_cat(b, c3))
        d2 = self.up2(self._up_cat(d3, c2))
        d1 = self.up1(self._up_cat(d2, c1))
        d0 = self.up0(self._up_cat(d1, c0))
        logits: torch.Tensor = self.outc(d0)
        return logits


class SegmentationModelFactory:
    """Instantiates a segmentation model from a SegmentationModelConfig."""

    @staticmethod
    def _build(name: str, *, num_classes: int, pretrained: bool) -> nn.Module:
        if name == "unet":
            return UNet(in_channels=3, num_classes=num_classes)
        if name in _TORCHVISION_SEG:
            builder, weights_enum, supports_aux = _TORCHVISION_SEG[name]
            kwargs: dict[str, Any] = {
                "weights": None,
                "num_classes": num_classes,
                "weights_backbone": weights_enum.DEFAULT if pretrained else None,
            }
            if supports_aux:
                kwargs["aux_loss"] = False
            return builder(**kwargs)
        raise ValueError(f"Unknown segmentation model: {name}")

    @staticmethod
    def create(config: SegmentationModelConfig) -> nn.Module:
        """Build a segmentation model with a ``num_classes``-wide dense head.

        Returns:
            nn.Module emitting per-pixel logits (``[N, num_classes, H, W]`` for
            U-Net; an ``OrderedDict`` with that under ``"out"`` for torchvision —
            normalize with ``segmentation_logits``).
        """
        use_pretrained = config.pretrained and config.weights_path is None
        model = SegmentationModelFactory._build(
            config.name, num_classes=config.num_classes, pretrained=use_pretrained
        )
        if config.weights_path is not None:
            load_local_weights(model, config.weights_path)
        return model


__all__ = ["SegmentationModelFactory", "UNet", "segmentation_logits"]
