"""Build torchvision detection models with the classification head resized.

This is the torchvision backend's model source (hybrid plan, ADR-033/034).
Supported families: Faster R-CNN, SSD / SSDLite, RetinaNet.

Every torchvision detector reserves label 0 for background, so each head is
sized to ``num_classes + 1`` — consistent with `DetectionDataset`, which emits
labels ``yolo_class + 1``. When ``pretrained`` is False we also pass
``weights_backbone=None`` so construction never downloads weights (CPU CI / tests).
"""

from __future__ import annotations

from functools import partial

import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_Weights,
    SSD300_VGG16_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
    ssd300_vgg16,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

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

# SSD family: builder, weights enum, canonical input size, head class, norm_layer.
_SSD_BUILDERS = {
    "ssd300_vgg16": (
        ssd300_vgg16,
        SSD300_VGG16_Weights,
        (300, 300),
        SSDClassificationHead,
        None,
    ),
    "ssdlite320_mobilenet_v3_large": (
        ssdlite320_mobilenet_v3_large,
        SSDLite320_MobileNet_V3_Large_Weights,
        (320, 320),
        SSDLiteClassificationHead,
        partial(nn.BatchNorm2d, eps=0.001, momentum=0.03),
    ),
}


def _build_faster_rcnn(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    builder, weights_enum = _FRCNN_BUILDERS[name]
    if pretrained:
        model = builder(weights=weights_enum.DEFAULT)
    else:
        model = builder(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    detector: nn.Module = model
    return detector


def _build_ssd(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    builder, weights_enum, size, head_cls, norm_layer = _SSD_BUILDERS[name]
    if pretrained:
        model = builder(weights=weights_enum.DEFAULT)
    else:
        model = builder(weights=None, weights_backbone=None)
    in_channels = det_utils.retrieve_out_channels(model.backbone, size)
    num_anchors = model.anchor_generator.num_anchors_per_location()
    if norm_layer is None:
        model.head.classification_head = head_cls(
            in_channels, num_anchors, num_classes + 1
        )
    else:
        model.head.classification_head = head_cls(
            in_channels, num_anchors, num_classes + 1, norm_layer
        )
    detector: nn.Module = model
    return detector


def _build_retinanet(num_classes: int, pretrained: bool) -> nn.Module:
    if pretrained:
        model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
    else:
        model = retinanet_resnet50_fpn(weights=None, weights_backbone=None)
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=model.backbone.out_channels,
        num_anchors=num_anchors,
        num_classes=num_classes + 1,
        norm_layer=partial(nn.GroupNorm, 32),
    )
    detector: nn.Module = model
    return detector


def build_torchvision_detector(
    name: str, num_classes: int, pretrained: bool
) -> nn.Module:
    """Return a torchvision detector with its head sized for ``num_classes``.

    Args:
        name: a torchvision detector name (see DetectionModelConfig).
        num_classes: number of foreground classes; background is added internally.
        pretrained: load COCO weights when True; random init (no download) when False.

    Raises:
        ValueError: for an unknown name.
    """
    if name in _FRCNN_BUILDERS:
        return _build_faster_rcnn(name, num_classes, pretrained)
    if name in _SSD_BUILDERS:
        return _build_ssd(name, num_classes, pretrained)
    if name == "retinanet_resnet50_fpn":
        return _build_retinanet(num_classes, pretrained)
    raise ValueError(f"Unknown torchvision detector: {name}")


__all__ = ["build_torchvision_detector"]
