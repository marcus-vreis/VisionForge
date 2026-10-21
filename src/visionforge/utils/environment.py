"""Capture the runtime environment for reproducibility.

A run is only reproducible if you know *which versions* produced it — the config
alone is not enough when PyTorch/torchvision/numpy can change numerical behavior
across releases (ADR-013). ``capture_environment`` collects the Python,
platform, and key-library versions so they can be persisted into ``run.json``.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import tomllib
from importlib import metadata
from pathlib import Path


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
        **{
            name: _safe_version(name)
            for name in (
                "scipy",
                "scikit-learn",
                "ultralytics",
                "timm",
                "optuna",
                "pillow",
                "pydantic",
            )
        },
        # The version the package itself reports: in a checkout that is the
        # pyproject's, which the installed metadata lags until a reinstall, and
        # the distribution's name ("visionforge-studio") once reported
        # "unknown" here in every run.json.
        "visionforge": _visionforge_version(),
        **_torch_runtime(),
        **_source_revision(),
    }


def _visionforge_version() -> str:
    """``visionforge.__version__``, or "unknown" if the package cannot say."""
    try:
        from visionforge import __version__
    except ImportError:
        return "unknown"
    return str(__version__) or "unknown"


def _source_revision() -> dict[str, str]:
    """Identify only the checkout containing the imported package's source."""
    unknown = {"git_revision": "unknown", "git_dirty": "unknown"}
    try:
        source = Path(__file__).resolve()
        package_dir = source.parent.parent
        root = package_dir.parent.parent
        if package_dir.name != "visionforge" or package_dir.parent.name != "src":
            return unknown
        if not (root / ".git").exists():
            return unknown
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        if project.get("project", {}).get("name") != "visionforge-studio":
            return unknown
        # Git environment overrides can otherwise identify a different repository.
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        }

        def git(*args: str) -> str:
            result = subprocess.run(
                ["git", "-c", "core.fsmonitor=false", "-C", str(root), *args],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
                env=env,
            )
            return result.stdout.strip()

        if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
            return unknown
        revision = git("rev-parse", "HEAD")
        if re.fullmatch(r"[0-9a-f]{40,64}", revision) is None:
            return unknown
        dirty = bool(
            git(
                "status",
                "--porcelain",
                "--untracked-files=normal",
                "--ignore-submodules=all",
            )
        )
        return {"git_revision": revision, "git_dirty": str(dirty).lower()}
    except Exception:  # noqa: BLE001 - provenance cannot fail a training run
        return unknown


__all__ = ["capture_environment"]
