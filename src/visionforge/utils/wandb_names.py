"""W&B project and entity names, checked without importing ``visionforge.core``.

The profile settings validate these names, and the settings module is imported
by the lightweight system routes, which must not load torch (ADR-172).
"""

from __future__ import annotations

import re

_NAME = re.compile(r"[\w.-]{1,128}", re.ASCII)


def is_valid_wandb_name(value: str) -> bool:
    """True for a project or entity name W&B accepts in a URL path."""
    return bool(_NAME.fullmatch(value)) and value not in {".", ".."}
