"""Capture the runtime environment for reproducibility.

A run is only reproducible if you know *which versions* produced it — the config
alone is not enough when PyTorch/torchvision/numpy can change numerical behavior
across releases (ADR-013). ``capture_environment`` collects the Python,
platform, and key-library versions so they can be persisted into ``run.json``.
"""

from __future__ import annotations

import platform
from importlib import metadata


def _safe_version(package: str) -> str:
    """Return an installed package's version, or ``"unknown"`` if unresolved."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"
    except Exception:  # noqa: BLE001 - never let version probing break a run
        return "unknown"


def _torch_runtime() -> dict[str, str]:
    """Probe the CUDA/cuDNN/GPU runtime torch was built against and is using.

    The pip version string alone (e.g. ``torch 2.5.1``) does not distinguish a
    CPU wheel from cu118/cu124 builds, and kernel selection differs across
    CUDA/cuDNN releases and GPU models — all of which can shift metrics between
    "identical" runs. ``"none"`` means probed-and-absent (CPU build / no GPU);
    ``"unknown"`` means the probe itself failed.
    """
    try:
        import torch

        cuda = getattr(torch.version, "cuda", None) or "none"
        cudnn = "none"
        if torch.backends.cudnn.is_available():
            cudnn = str(torch.backends.cudnn.version() or "none")
        gpu = "none"
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
        return {"cuda": str(cuda), "cudnn": cudnn, "gpu": gpu}
    except Exception:  # noqa: BLE001 - never let runtime probing break a run
        return {"cuda": "unknown", "cudnn": "unknown", "gpu": "unknown"}


def capture_environment() -> dict[str, str]:
    """Return Python/platform/library versions + CUDA runtime for the current run.

    All values are strings; a library that cannot be resolved reports
    ``"unknown"`` rather than raising, so capturing the environment never fails
    a training run.
    """
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _safe_version("torch"),
        "torchvision": _safe_version("torchvision"),
        "numpy": _safe_version("numpy"),
        "visionforge": _safe_version("visionforge"),
        **_torch_runtime(),
    }


__all__ = ["capture_environment"]
