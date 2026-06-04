from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image

from visionforge.core.gradcam import GradCAM, overlay_cam, resolve_target_layer


class TinyCNN(nn.Module):
    """Conv -> ReLU -> GAP -> Linear; a minimal CAM-able classifier."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TestResolveTargetLayer:
    def test_returns_last_conv_of_tiny_cnn(self) -> None:
        model = TinyCNN()
        layer = resolve_target_layer(model)
        assert layer is model.conv

    def test_returns_a_conv_for_resnet(self) -> None:
        import torchvision.models as tv

        model = tv.resnet18(weights=None)
        layer = resolve_target_layer(model)
        assert isinstance(layer, nn.Conv2d)

    def test_raises_when_no_conv(self) -> None:
        model = nn.Sequential(nn.Linear(4, 2))
        try:
            resolve_target_layer(model)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


class TestGradCAM:
    def test_cam_shape_matches_input(self) -> None:
        model = TinyCNN()
        cam = GradCAM(model, resolve_target_layer(model))
        heat = cam(torch.randn(1, 3, 16, 16))
        cam.remove()
        assert heat.shape == (16, 16)

    def test_cam_normalized_zero_to_one(self) -> None:
        model = TinyCNN()
        cam = GradCAM(model, resolve_target_layer(model))
        heat = cam(torch.randn(1, 3, 16, 16))
        cam.remove()
        assert torch.isfinite(heat).all()
        assert float(heat.min()) >= 0.0
        assert float(heat.max()) <= 1.0 + 1e-5

    def test_explicit_target_class(self) -> None:
        model = TinyCNN(num_classes=3)
        cam = GradCAM(model, resolve_target_layer(model))
        heat = cam(torch.randn(1, 3, 16, 16), target_class=2)
        cam.remove()
        assert heat.shape == (16, 16)

    def test_can_run_twice(self) -> None:
        model = TinyCNN()
        cam = GradCAM(model, resolve_target_layer(model))
        h1 = cam(torch.randn(1, 3, 8, 8))
        h2 = cam(torch.randn(1, 3, 8, 8))
        cam.remove()
        assert h1.shape == h2.shape == (8, 8)


class TestOverlay:
    def test_overlay_returns_rgb_image_of_input_size(self) -> None:
        base = torch.rand(3, 24, 24)
        cam = torch.rand(24, 24)
        out = overlay_cam(base, cam)
        assert isinstance(out, Image.Image)
        assert out.mode == "RGB"
        assert out.size == (24, 24)

    def test_overlay_accepts_pil_base(self) -> None:
        base = Image.new("RGB", (20, 20))
        cam = torch.rand(20, 20)
        out = overlay_cam(base, cam)
        assert out.size == (20, 20)
