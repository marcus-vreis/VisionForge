from __future__ import annotations

from pathlib import Path

import pytest
import torch

from visionforge.models.segmentation_factory import (
    SegmentationModelFactory,
    UNet,
    segmentation_logits,
)
from visionforge.utils.segmentation_config import SegmentationModelConfig

# torchvision families are parametrized but kept light: pretrained=False so no
# weights are ever downloaded, batch=1 and a tiny 64x64 input so CPU CI stays fast.
_TORCHVISION_NAMES = [
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
    "deeplabv3_mobilenet_v3_large",
    "fcn_resnet50",
    "fcn_resnet101",
    "lraspp_mobilenet_v3_large",
]


def _model(name: str, num_classes: int = 3) -> torch.nn.Module:
    cfg = SegmentationModelConfig(
        name=name,  # type: ignore[arg-type]
        num_classes=num_classes,
        pretrained=False,
    )
    return SegmentationModelFactory.create(cfg)


class TestTorchvisionFamilies:
    @pytest.mark.parametrize("name", _TORCHVISION_NAMES)
    def test_builds_and_forward_shape(self, name: str) -> None:
        model = _model(name, num_classes=3).eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            logits = segmentation_logits(model(x))
        assert logits.shape == (1, 3, 64, 64)

    def test_num_classes_drives_output_channels(self) -> None:
        model = _model("fcn_resnet50", num_classes=5).eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            logits = segmentation_logits(model(x))
        assert logits.shape[1] == 5


class TestUNet:
    def test_builds_and_forward_shape(self) -> None:
        model = _model("unet", num_classes=4).eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            logits = segmentation_logits(model(x))
        assert logits.shape == (1, 4, 64, 64)

    def test_unet_returns_plain_tensor(self) -> None:
        model = _model("unet", num_classes=2).eval()
        x = torch.randn(1, 3, 48, 48)
        with torch.no_grad():
            out = model(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 2, 48, 48)

    def test_unet_handles_non_power_of_two_size(self) -> None:
        # Up path must realign to the input resolution even for odd sizes.
        model = _model("unet", num_classes=3).eval()
        x = torch.randn(1, 3, 70, 50)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 70, 50)

    def test_unet_class_constructs_directly(self) -> None:
        net = UNet(in_channels=3, num_classes=2, base=16)
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (1, 2, 32, 32)


class TestSegmentationLogits:
    def test_unwraps_dict_output(self) -> None:
        t = torch.randn(1, 3, 8, 8)
        assert segmentation_logits({"out": t}) is t

    def test_passes_through_tensor(self) -> None:
        t = torch.randn(1, 3, 8, 8)
        assert segmentation_logits(t) is t


class TestWeightsAndErrors:
    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown segmentation"):
            SegmentationModelFactory._build(
                "segnet999", num_classes=2, pretrained=False
            )

    def test_local_weights_round_trip(self, tmp_path: Path) -> None:
        model = _model("unet", num_classes=3)
        ckpt = tmp_path / "w.pth"
        torch.save(model.state_dict(), ckpt)
        cfg = SegmentationModelConfig(
            name="unet", num_classes=3, pretrained=False, weights_path=ckpt
        )
        reloaded = SegmentationModelFactory.create(cfg)
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            assert segmentation_logits(reloaded(x)).shape == (1, 3, 32, 32)
