"""Local store for dataset-provider API keys.

Typing a Roboflow key or a Kaggle token on every download is the kind of
friction that stops people using a feature at all. This keeps one copy on the
machine that will use it.

Two decisions worth stating, because credentials are involved:

* **Per user, not per project.** Custom models and tasks live next to the
  working directory on purpose — two projects should not share them. A key is
  the opposite: it belongs to the person, so it lives in ``~/.visionforge/``
  and is not something a project folder can accidentally carry into a git
  repository or a synced drive.
* **Read back masked.** The value is returned to the browser as
  ``rf_••••••3f2a``, never in full. The GUI only needs to show that a key is
  stored and which one; the download runs server-side, where the real value
  already is. A masked read means a screenshot of the panel is not a leak.

The file is written with owner-only permissions where the platform supports
it. On Windows that call is a no-op, so the file inherits the profile's ACL —
worth knowing, not worth pretending otherwise.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

from loguru import logger

# Providers that can hold a key. `torchvision` and `huggingface`-public need
# none, but Hugging Face private datasets do, so it is offered.
PROVIDERS = ("roboflow", "kaggle", "huggingface")

_FILENAME = "credentials.json"


def config_dir() -> Path:
    """Per-user VisionForge config directory (``VISIONFORGE_HOME`` overrides)."""
    override = os.environ.get("VISIONFORGE_HOME")
    if override:
        return Path(override)
    return Path.home() / ".visionforge"


def _credentials_path() -> Path:
    return config_dir() / _FILENAME


def mask(value: str) -> str:
    """Render a key as a recognizable but unusable stub.

    Keeps the last four characters so the researcher can tell *which* key is
    stored — the common question is "is this my old one?", not "what is it?".
    """
    if not value:
        return ""
    if len(value) <= 4:
        return "•" * len(value)
    return "•" * max(4, len(value) - 4) + value[-4:]


def _read_all() -> dict[str, str]:
    path = _credentials_path()
    if not path.is_file():
        return {}
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        # A corrupt store must not break the app: the user can just save again.
        logger.warning("Could not read stored credentials: {}", exc)
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if isinstance(v, str)}


def _write_all(data: dict[str, str]) -> None:
    path = _credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if sys.platform != "win32":
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def save_credential(provider: str, value: str) -> None:
    """Store (or replace) one provider's key.

    Raises:
        ValueError: for an unknown provider or a blank value.
    """
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Expected one of: {', '.join(PROVIDERS)}."
        )
    value = value.strip()
    if not value:
        raise ValueError("Credential value cannot be empty.")
    data = _read_all()
    data[provider] = value
    _write_all(data)
    # Log the event, never the value.
    logger.info("Stored {} credential in {}", provider, _credentials_path())


def load_credential(provider: str) -> str | None:
    """Return the stored key for a provider, or None."""
    return _read_all().get(provider) or None


def forget_credential(provider: str) -> bool:
    """Delete one provider's key. True when something was actually removed."""
    data = _read_all()
    if provider not in data:
        return False
    del data[provider]
    _write_all(data)
    logger.info("Removed {} credential", provider)
    return True


def credential_status() -> dict[str, dict[str, Any]]:
    """Per provider: whether a key is stored, and its masked form."""
    data = _read_all()
    return {
        provider: {
            "saved": provider in data,
            "masked": mask(data.get(provider, "")),
        }
        for provider in PROVIDERS
    }


def resolve_credential(provider: str, supplied: str | None) -> str | None:
    """The key a download should use: what the caller passed, else the stored one.

    Explicit beats stored, so a one-off key can be used without overwriting
    what is saved.
    """
    if supplied and supplied.strip():
        return supplied.strip()
    return load_credential(provider)


__all__ = [
    "PROVIDERS",
    "config_dir",
    "credential_status",
    "forget_credential",
    "load_credential",
    "mask",
    "resolve_credential",
    "save_credential",
]
