"""`deterministic=True` turns on PyTorch's deterministic kernels, not only cuDNN's (ADR-199)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
import torch.nn.functional as F

from visionforge.core.trainer import _seed_everything


@pytest.fixture(autouse=True)
def _restore_global_torch_state() -> Iterator[None]:
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn


def test_deterministic_turns_on_strict_kernels_without_raising() -> None:
    _seed_everything(0, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    # An op without a deterministic kernel warns instead of stopping the run.
    assert torch.is_deterministic_algorithms_warn_only_enabled()


def test_fast_mode_turns_strict_kernels_off_again() -> None:
    _seed_everything(0, deterministic=True)
    _seed_everything(0, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_bilinear_upsampling_gradients_repeat_on_cuda() -> None:
    """The segmentation decoder's upsampling: its CUDA backward used atomicAdd (#305)."""
    _seed_everything(0, deterministic=True)
    x = torch.randn(2, 8, 16, 16, device="cuda", requires_grad=True)
    weights = (
        torch.arange(2 * 8 * 32 * 32, device="cuda").view(2, 8, 32, 32).float().sin()
    )
    grads = []
    for _ in range(5):
        x.grad = None
        out = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        (out * weights).sum().backward()
        assert x.grad is not None
        grads.append(x.grad.clone())
    assert all(torch.equal(grads[0], g) for g in grads)
