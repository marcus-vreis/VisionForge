"""VisionForge — local-first computer-vision experimentation platform."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from visionforge.utils.doctor import checkout_version as _checkout_version

try:
    __version__ = _version("visionforge-studio")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = "0.0.0+unknown"

# In a checkout the metadata is as old as the last `pip install -e .`; the
# pyproject.toml next to the code is the version actually running.
__version__ = _checkout_version() or __version__

__all__ = ["__version__"]
