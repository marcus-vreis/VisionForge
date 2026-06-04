"""Capture the runtime environment for reproducibility.

A run is only reproducible if you know *which versions* produced it — the config
alone is not enough when PyTorch/torchvision/numpy can change numerical behavior
across releases (CLAUDE.md §7.4). ``capture_environment`` collects the Python,
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


def capture_environment() -> dict[str, str]:
    """Return Python/platform/library versions for the current run.

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
    }


__all__ = ["capture_environment"]
