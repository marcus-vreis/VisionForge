"""Local paths out of a run record that is about to leave the computer (ADR-174).

A run.json written on this machine names the run folder and the dataset folder
by absolute path, and a path under the home folder carries the account name
(``C:\\Users\\<name>\\...``, ``/home/<name>/...``). A record shared on a public
archive keeps only the final name of each path: the dataset stays recognizable
(``exemplo-anomalia``) and the fingerprint still identifies its contents.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

# A drive or UNC path inside free text, up to the first space or quote.
_EMBEDDED_WINDOWS = re.compile(r"(?:\b[A-Za-z]:[\\/]|\\\\)[^\s'\"<>|]*")
# The POSIX folders that hold accounts, mounts and notebooks (Colab's /content).
_EMBEDDED_POSIX = re.compile(
    r"(?<![\w.~/])/(?:home|Users|root|content|mnt|media|tmp|var|opt|srv|workspace)"
    r"/[^\s'\"<>|]*"
)


def without_local_paths(value: Any) -> Any:
    """``value`` with every local absolute path reduced to its final name.

    Dictionaries and lists are rebuilt; numbers, booleans and ``None`` pass
    through unchanged.
    """
    if isinstance(value, dict):
        return {key: without_local_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [without_local_paths(item) for item in value]
    if isinstance(value, str):
        return _text_without_local_paths(value)
    return value


def _text_without_local_paths(text: str) -> str:
    if _is_absolute(text):
        return _final_name(text)
    text = _EMBEDDED_WINDOWS.sub(lambda match: _final_name(match.group()), text)
    return _EMBEDDED_POSIX.sub(lambda match: _final_name(match.group()), text)


def _is_absolute(text: str) -> bool:
    if PureWindowsPath(text).is_absolute() or text.startswith("~/"):
        return True
    return text.startswith("/") and len(PurePosixPath(text).parts) > 1


def _final_name(path: str) -> str:
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
