"""JSON that any reader accepts: a non-finite float is written as ``null``.

Python's ``json`` writes NaN and infinity as the bare tokens ``NaN`` and
``Infinity``, which are not JSON: a browser's ``JSON.parse`` and most other
parsers refuse the whole file. A seed or fold whose metric came back non-finite
is a known outcome (ADR-061), so the result files that carry it use these.
"""

from __future__ import annotations

import json
import math
from typing import Any


def json_safe(value: Any) -> Any:
    """``value`` with every NaN / infinity replaced by ``None``."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    return value


def dumps_strict(value: Any, *, indent: int | None = 2) -> str:
    """Serialise ``value`` as valid JSON; ``allow_nan=False`` catches any leak."""
    return json.dumps(json_safe(value), indent=indent, allow_nan=False)


__all__ = ["dumps_strict", "json_safe"]
