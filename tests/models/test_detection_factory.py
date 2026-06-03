from __future__ import annotations

from typing import Any

import pytest
import torch

from visionforge.models.detection_factory import build_torchvision_detector


def _train_forward_loss(model: torch.nn.Module, num_classes: int) -> Any:
    """Run one train-mode forward using the *max* foreground label.

    If the classification head were sized to num_classes instead of
    num_classes + 1, a target label == num_classes would be out of range and the
    loss would raise — so a clean loss dict confirms the background-slot sizing.
    """
    model.train()
    # 320x320 so the SSDLite mobilenet backbone produces its expected feature
    # maps; batch of 2 so the head's BatchNorm has >1 value at the 1x1 map.
    target = {
        "boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0]]),
        "labels": torch.tensor([num_classes], dtype=torch.int64),
    }
    images = [torch.rand(3, 320, 320), torch.rand(3, 320, 320)]
    targets = [target, dict(target)]
    return model(images, targets)


class TestFasterRCNNFamily:
    def test_resnet50_head_sized_for_num_classes_plus_background(self) -> None:
        model = build_torchvision_detector(
            "fasterrcnn_resnet50_fpn", num_classes=3, pretrained=False
        )
        assert model.roi_heads.box_predictor.cls_score.out_features == 4

    def test_mobilenet_variant_builds_with_resized_head(self) -> None:
        model = build_torchvision_detector(
            "fasterrcnn_mobilenet_v3_large_fpn", num_classes=2, pretrained=False
        )
        assert model.roi_heads.box_predictor.cls_score.out_features == 3

    def test_eval_forward_returns_detection_dicts(self) -> None:
        model = build_torchvision_detector(
            "fasterrcnn_resnet50_fpn", num_classes=2, pretrained=False
        )
        model.eval()
        with torch.no_grad():
            out = model([torch.rand(3, 128, 128)])
        assert isinstance(out, list) and len(out) == 1
        assert {"boxes", "labels", "scores"} <= set(out[0].keys())


class TestSSDFamily:
    @pytest.mark.parametrize("name", ["ssd300_vgg16", "ssdlite320_mobilenet_v3_large"])
    def test_builds_and_accepts_max_label(self, name: str) -> None:
        model = build_torchvision_detector(name, num_classes=2, pretrained=False)
        losses = _train_forward_loss(model, num_classes=2)
        assert isinstance(losses, dict) and len(losses) > 0
        assert all(isinstance(v, torch.Tensor) for v in losses.values())


class TestRetinaNet:
    def test_builds_and_accepts_max_label(self) -> None:
        model = build_torchvision_detector(
            "retinanet_resnet50_fpn", num_classes=3, pretrained=False
        )
        losses = _train_forward_loss(model, num_classes=3)
        assert isinstance(losses, dict) and len(losses) > 0


class TestUnknown:
    def test_unknown_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown torchvision detector"):
            build_torchvision_detector("yolo11n", num_classes=2, pretrained=False)
