"""Grad-CAM explainability for classification models.

Grad-CAM (Gradient-weighted Class Activation Mapping) produces a heatmap over the
input highlighting the regions most responsible for a predicted class. It hooks a
target convolutional layer to capture forward activations and backward gradients,
weights the activation channels by the global-average-pooled gradients, and
ReLU-sums them into a class-localizing map.

This is a post-hoc analysis of an already-trained classification checkpoint — it
needs no new config tree and is dependency-free (pure torch). See
documentation/GRADCAM_PLAN.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ImageNet de-normalization (the classification pipeline normalizes with these);
# used to recover a viewable RGB image for the overlay.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_target_layer(model: nn.Module) -> nn.Conv2d:
    """Return the last ``nn.Conv2d`` in ``model`` (the conventional Grad-CAM target).

    Arch-agnostic: correct for ResNet/EfficientNet/VGG/AlexNet without per-family
    branching, since module registration order follows the forward path.

    Raises:
        ValueError: if the model contains no convolutional layer.
    """
    last_conv: nn.Conv2d | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("Model has no nn.Conv2d layer to target for Grad-CAM.")
    return last_conv


class GradCAM:
    """Computes Grad-CAM heatmaps for a model and a target conv layer.

    Register once, call per image, then ``remove()`` to detach the hooks. Safe to
    call multiple times before removal.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self._model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(
        self, _module: nn.Module, _inp: Any, output: torch.Tensor
    ) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module: nn.Module, _grad_in: Any, grad_out: Any) -> None:
        self._gradients = grad_out[0].detach()

    def __call__(
        self, input_tensor: torch.Tensor, target_class: int | None = None
    ) -> torch.Tensor:
        """Return a ``[H, W]`` heatmap in [0, 1] for ``input_tensor`` (batch of 1).

        ``target_class`` defaults to the model's predicted class.
        """
        self._model.eval()
        # Require grad on the input so the full backward hook fires cleanly
        # (otherwise torch warns that no inputs require gradients).
        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = self._model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self._model.zero_grad(set_to_none=True)
        logits[:, target_class].sum().backward()

        assert self._activations is not None and self._gradients is not None
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # [1, K, 1, 1]
        cam = torch.relu((weights * self._activations).sum(dim=1))  # [1, h, w]

        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return cam[0].detach()

    def remove(self) -> None:
        """Detach the forward/backward hooks."""
        for h in self._handles:
            h.remove()
        self._handles = []


def _jet_colormap(cam: np.ndarray) -> np.ndarray:
    """Map a [0, 1] heatmap to an RGB jet image (``uint8``), no matplotlib dep."""
    x = np.clip(cam, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _denormalize(image: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization on a ``[3, H, W]`` tensor → ``uint8`` HWC RGB."""
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
    img = (image.detach().cpu() * std + mean).clamp(0.0, 1.0)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def overlay_cam(
    base: torch.Tensor | Image.Image,
    cam: torch.Tensor,
    *,
    alpha: float = 0.5,
    normalized: bool = True,
) -> Image.Image:
    """Overlay a Grad-CAM heatmap on a base image and return a PIL RGB image.

    Args:
        base: the input image — a ``[3, H, W]`` tensor (de-normalized when
            ``normalized``) or a PIL image.
        cam: the ``[H, W]`` heatmap in [0, 1].
        alpha: blend weight of the heatmap over the base image.
        normalized: when ``base`` is a tensor, undo ImageNet normalization first.
    """
    if isinstance(base, Image.Image):
        base_rgb = np.array(base.convert("RGB"))
    elif normalized:
        base_rgb = _denormalize(base)
    else:
        arr = base.detach().cpu().clamp(0.0, 1.0)
        base_rgb = (arr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    heat = _jet_colormap(cam.detach().cpu().numpy())
    if heat.shape[:2] != base_rgb.shape[:2]:
        heat_img = Image.fromarray(heat).resize(
            (base_rgb.shape[1], base_rgb.shape[0]), Image.Resampling.BILINEAR
        )
        heat = np.array(heat_img)

    blended = (base_rgb * (1.0 - alpha) + heat * alpha).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


__all__ = ["GradCAM", "resolve_target_layer", "overlay_cam"]
