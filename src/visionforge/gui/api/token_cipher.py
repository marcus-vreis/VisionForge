"""Encryption at rest for the per-profile tokens (ADR-253).

The operator gives a Fernet key in ``VISIONFORGE_TOKEN_KEY``; with it a token
file holds ciphertext, and a copy of the config folder (a backup, a disk
image) is no longer a copy of everyone's tokens. Several keys, separated by
commas, rotate: the first seals, all of them open.

The key comes from the environment only, like the GitHub OAuth secret
(ADR-250): a command line shows in the process list, and a job's child process
does not receive the variable (ADR-248).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptography.fernet import Fernet

KEY_ENV = "VISIONFORGE_TOKEN_KEY"
# Marks a sealed file. A saved token has no ":" this early by accident: the
# prefix is longer than any provider's own token prefix.
SEALED_PREFIX = "vf-fernet-v1:"


class TokenKeyError(ValueError):
    """``VISIONFORGE_TOKEN_KEY`` cannot be used as given."""


@dataclass(frozen=True)
class TokenCipher:
    """Seals with the first key and opens with any of them."""

    keys: tuple[Fernet, ...]

    def seal(self, binding: str, value: str) -> str:
        """``value`` as a sealed file body that only opens under ``binding``."""
        # Typed here: where ``cryptography`` is not installed mypy sees Any.
        sealed: bytes = self.keys[0].encrypt(f"{binding}\0{value}".encode())
        return SEALED_PREFIX + sealed.decode("ascii")

    def unseal(self, binding: str, text: str) -> tuple[str, bool] | None:
        """The token and whether the first key sealed it.

        ``None`` when no key opens the file or it was sealed for another
        ``binding`` (a file moved onto another profile or service).
        """
        from cryptography.fernet import InvalidToken

        body = text[len(SEALED_PREFIX) :].strip().encode()
        for index, key in enumerate(self.keys):
            try:
                plain = key.decrypt(body).decode()
            except (InvalidToken, UnicodeDecodeError):
                continue
            bound, separator, value = plain.partition("\0")
            if not separator or bound != binding:
                return None
            return value, index == 0
        return None


def is_sealed(text: str) -> bool:
    """True for a file body written by :meth:`TokenCipher.seal`."""
    return text.startswith(SEALED_PREFIX)


def token_cipher() -> TokenCipher | None:
    """The cipher the environment asks for, or ``None`` when it asks for none.

    Read on each call, so the environment is the one source of truth.

    Raises:
        TokenKeyError: the variable is set and ``cryptography`` is not
            installed, or one of its keys is not a Fernet key. The message
            never carries the key.
    """
    names = os.environ.get(KEY_ENV, "").replace(",", " ").split()
    if not names:
        return None
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise TokenKeyError(
            f"{KEY_ENV} is set, but the 'cryptography' package is not "
            'installed: pip install "visionforge-studio[server]".'
        ) from exc
    try:
        return TokenCipher(tuple(Fernet(name) for name in names))
    except ValueError:
        raise TokenKeyError(
            f"{KEY_ENV} must hold Fernet keys (32 url-safe base64 bytes each, "
            f"comma-separated); of the {len(names)} given, at least one is not."
        ) from None


def tokens_encrypted() -> bool:
    """Whether a token saved now is sealed: a usable key is in the environment."""
    try:
        return token_cipher() is not None
    except TokenKeyError:
        return False
