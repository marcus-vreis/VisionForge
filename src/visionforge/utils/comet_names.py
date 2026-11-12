"""Comet workspace and project names, checked without importing ``visionforge.core``.

The profile settings validate these names, and the settings module is imported
by the lightweight system routes, which must not load torch (ADR-200). Comet
keeps project names in lower case, so only lower case is accepted: the name
read back is the name the user typed.
"""

from __future__ import annotations

import re

_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,99}", re.ASCII)


def is_valid_comet_name(value: str) -> bool:
    """True for a lower-case workspace or project name Comet keeps as typed."""
    return bool(_NAME.fullmatch(value))
