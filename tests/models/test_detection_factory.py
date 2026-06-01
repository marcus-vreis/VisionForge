from __future__ import annotations

import pytest
import torch

from visionforge.models.detection_factory import build_torchvision_detector


class TestFasterRCNNFamily:
    def test_resnet50_head_sized_for_num_classes_plus_background(self) -> None:
        model = build_torchvision_detector(
            "fasterrcnn_resnet50_fpn", num_classes=3, pretrained=False
        )
        # background is class 0, so the box predictor has num_classes + 1 outputs.
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


class TestUnwiredAndUnknown:
    @pytest.mark.parametrize(
        "name",
        ["retinanet_resnet50_fpn", "ssd300_vgg16", "ssdlite320_mobilenet_v3_large"],
    )
    def test_declared_but_unwired_raises_not_implemented(self, name: str) -> None:
        with pytest.raises(NotImplementedError, match="not wired yet"):
            build_torchvision_detector(name, num_classes=2, pretrained=False)

    def test_unknown_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown torchvision detector"):
            build_torchvision_detector("yolo11n", num_classes=2, pretrained=False)
