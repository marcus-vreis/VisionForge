"""Whether a DataLoader should pin its host memory."""

from __future__ import annotations

import torch

from visionforge.utils.config import DeviceConfig


def effective_pin_memory(requested: bool, device: torch.device) -> bool:
    """Return ``requested`` only when the run trains on a CUDA device.

    Pinned (page-locked) host memory speeds up host-to-GPU copies and does
    nothing for a CPU run; there PyTorch warns on every DataLoader that
    "'pin_memory' argument is set as true but no accelerator is found". The
    config default is ``True`` (right for a GPU), so the DataLoader sites ask
    this instead of passing ``cfg.data.pin_memory`` straight through. MPS is
    excluded on purpose: it ignores pinning.
    """
    return requested and device.type == "cuda"


def pin_memory_for(requested: bool, device_cfg: DeviceConfig) -> bool:
    """:func:`effective_pin_memory` for a data module that only has the config.

    Mirrors the CUDA-or-not outcome of ``resolve_device`` without calling it,
    because that logs a fallback warning and the trainer already logs it once.
    """
    wants_cuda = device_cfg.kind in ("cuda", "multi_cuda")
    device = torch.device("cuda" if wants_cuda and torch.cuda.is_available() else "cpu")
    return effective_pin_memory(requested, device)
