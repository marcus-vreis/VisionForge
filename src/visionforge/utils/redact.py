"""Mask credentials in text that is about to be shown or logged.

A provider's SDK does not promise to keep a key out of its error messages. The
Roboflow client posts to ``API_URL + "/?api_key=" + api_key``, so when the
network is down ``requests`` raises a ``ConnectionError`` whose text is
``Max retries exceeded with url: /?api_key=<KEY>``. Forwarding that message to
the browser, or handing the exception to a logger that prints its traceback,
writes the key somewhere it was never meant to be.

``redact_secrets`` is the one place that knows how to prevent that. It works on
the text rather than on the exception, because the text is what leaves the
process, and it works in two ways that cover each other:

* **By shape** — ``api_key=…``, ``token=…``, ``Authorization: Bearer …`` and a
  few neighbours are masked whether or not the value is known. This catches a
  key the caller did not think to pass in.
* **By value** — any exact occurrence of a secret the caller *does* know
  (the credential the request used) is masked wherever it appears, including
  in free text and in its URL-encoded form, which no pattern could recognize.

Everything else in the message is left alone: an error with its cause masked
is still a useful error, and one masked into ``***`` is not.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from urllib.parse import quote, quote_plus

MASK = "***"

# Shorter than any real API key or token. Masking a one- or two-character
# "secret" would turn every such letter in the message into a mask and leave
# nothing readable; the by-shape patterns still catch it where it matters.
_MIN_SECRET_LENGTH = 4

# What names a secret. No prefix is matched on purpose: `access_token`,
# `x-api-key` and `KAGGLE_API_TOKEN` all end in one of these, and only the
# *value* is replaced, so there is nothing to gain from consuming the prefix.
_NAME = r"(?:api[_-]?key|token|secret|password|passwd)"
_HEADER = r"(?:proxy-)?authorization|x-api-key|x-auth-token|x-access-token"
_OPT_QUOTE = r"""(?:\\?["'])?"""

_PATTERNS: tuple[re.Pattern[str], ...] = (
    # `?api_key=KEY`, `&token=KEY`, `access_token=KEY` — a URL or a form body.
    re.compile(rf"{_NAME}=(?P<val>[^&\s\"'<>)\]}},;]+)", re.IGNORECASE),
    # `"api_key": "KEY"` — JSON, or the repr of a dict (single quotes, or
    # escaped quotes when the JSON is itself inside a string).
    re.compile(
        rf"""{_NAME}(?P<q>\\?["'])\s*:\s*(?P<vq>\\?["'])(?P<val>[^"'\\]+)(?P=vq)""",
        re.IGNORECASE,
    ),
    # `Authorization: Bearer KEY`, also as `'Authorization': 'Bearer KEY'`.
    re.compile(
        rf"(?:{_HEADER}){_OPT_QUOTE}\s*[:=]\s*{_OPT_QUOTE}"
        r"(?:(?:bearer|basic|token|digest|apikey)\s+)?(?P<val>[^\s\"',;}\\]+)",
        re.IGNORECASE,
    ),
    # A bare `Bearer KEY`. Eight characters or more, so the English word
    # ("the bearer of …") is not mistaken for the scheme.
    re.compile(r"\bbearer\s+(?P<val>[A-Za-z0-9._~+/=-]{8,})", re.IGNORECASE),
    # `https://user:PASSWORD@host`, as in a proxy URL.
    re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@]+:(?P<val>[^/\s@]+)@", re.IGNORECASE),
)


def _mask_value(match: re.Match[str]) -> str:
    """Replace only the ``val`` group, keeping the text around it."""
    start = match.start()
    text = match.group(0)
    return text[: match.start("val") - start] + MASK + text[match.end("val") - start :]


def _literal_forms(secret: str) -> set[str]:
    """The ways a known secret can appear: as is, and percent-encoded in a URL."""
    return {secret, quote(secret, safe=""), quote(secret), quote_plus(secret)}


def redact_secrets(text: str, *known_secrets: str | None) -> str:
    """Return ``text`` with credentials masked.

    Args:
        text: An error message, a traceback, or any string about to be shown or
            logged.
        *known_secrets: Credentials the caller knows were in play (the key the
            request used, the one saved on disk). Each exact occurrence is
            masked. ``None``, blank and very short values are ignored.

    Returns:
        The same text with the secrets replaced by ``***``. Text with nothing
        to mask comes back unchanged, and redacting twice changes nothing more.
    """
    forms: set[str] = set()
    for secret in known_secrets:
        stripped = (secret or "").strip()
        if len(stripped) >= _MIN_SECRET_LENGTH:
            forms |= _literal_forms(stripped)

    # Longest first, so a secret that contains another is masked whole instead
    # of leaving the rest of it behind.
    for form in sorted(forms, key=len, reverse=True):
        text = text.replace(form, MASK)

    mask: Callable[[re.Match[str]], str] = _mask_value
    for pattern in _PATTERNS:
        text = pattern.sub(mask, text)
    return text


__all__ = ["MASK", "redact_secrets"]
