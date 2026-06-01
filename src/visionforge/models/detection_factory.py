"""Build torchvision detection models with the classification head resized.

This is the torchvision backend's model source (hybrid plan, ADR-033/034). The
Faster R-CNN family is wired first (the user named "Faster"); SSD / RetinaNet
raise a clear NotImplementedError until their head-replacement is added.

torchvision detectors count background as class 0, so the box predictor is sized
to ``num_classes + 1``. When ``pretrained`` is False we also pass
``weights_backbone=None`` so construction never downloads weights — important for
CPU CI and unit tests.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# name -> (builder, weights enum) for the Faster R-CNN family.
_FRCNN_BUILDERS = {
    "fasterrcnn_resnet50_fpn": (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
    ),
    "fasterrcnn_mobilenet_v3_large_fpn": (
        fasterrcnn_mobilenet_v3_large_fpn,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    ),
}

# Declared in DetectionModelConfig but not wired yet — each needs its own
# family-specific head replacement (SSDClassificationHead / RetinaNet head).
_NOT_YET_WIRED = (
    "retinanet_resnet50_fpn",
    "ssd300_vgg16",
    "ssdlite320_mobilenet_v3_large",
)


def build_torchvision_detector(
    name: str, num_classes: int, pretrained: bool
) -> nn.Module:
    """Return a torchvision detector with its head sized for ``num_classes``.

    Args:
        name: a torchvision detector name (see DetectionModelConfig).
        num_classes: number of foreground classes; background is added internally.
        pretrained: load COCO weights when True; random init (no download) when False.

    Raises:
        NotImplementedError: for a declared-but-unwired family (SSD / RetinaNet).
        ValueError: for an unknown name.
    """
    if name in _FRCNN_BUILDERS:
        builder, weights_enum = _FRCNN_BUILDERS[name]
        if pretrained:
            model = builder(weights=weights_enum.DEFAULT)
        else:
            model = builder(weights=None, weights_backbone=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        detector: nn.Module = model
        return detector

    if name in _NOT_YET_WIRED:
        raise NotImplementedError(
            f"torchvision detector '{name}' is not wired yet; "
            f"supported torchvision models: {sorted(_FRCNN_BUILDERS)}."
        )

    raise ValueError(f"Unknown torchvision detector: {name}")


__all__ = ["build_torchvision_detector"]
