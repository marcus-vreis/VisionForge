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

# Shorter than any real API key or token (Roboflow, Kaggle and Hugging Face keys
# are 20+ characters). A junk saved value such as `test` would otherwise turn a
# trace path into `***s/foo/***_x.py` and leave nothing readable; the by-shape
# patterns still catch a short one wherever it sits behind `api_key=` and the like.
_MIN_SECRET_LENGTH = 8

# What names a secret. No prefix is matched on purpose: `access_token`,
# `x-api-key` and `KAGGLE_API_TOKEN` all end in one of these, and only the
# *value* is replaced, so there is nothing to gain from consuming the prefix.
_NAME = r"(?:api[_-]?key|token|secret|password|passwd)"
# The names that may be followed by a colon (`password: hunter22`). `token` only
# counts when it is part of a compound name (`access_token`, `x-auth-token`):
# "Kaggle token: expired" is an ordinary sentence and must stay readable.
_COLON_NAME = r"(?:api[_-]?key|secret|password|passwd|(?<=[_-])token)"
# Spaces and tabs only: a name at the end of one line must not reach for a value
# on the next.
_SP = r"[ \t]*"
# A quote, possibly backslash-escaped: JSON inside a string and the repr of a dict
# inside a message both carry `\"`.
_Q = r"""\\?["']"""
# The kind of credential, not the credential: in `x-auth-token: Bearer KEY` the
# value is `KEY`, and the word before it stays visible. Without this a pattern
# takes `Bearer` as the value and leaves the key behind in clear.
_SCHEME = r"(?:(?:bearer|basic|token|digest|apikey)[ \t]+)?"
# One character of an unquoted value: anything but whitespace, a quote, or the
# delimiters that close a URL, a call or a literal around it. A backslash counts
# too, except where it starts an escape: `\"` (the closing quote of a value inside
# a string) and `\n`, `\r`, `\t` (the line break a repr prints as two characters).
# `abc\nnext` is the secret `abc` and the text `\nnext`; `pa\ss12345` is all
# secret. Written as alternatives that cannot overlap, so matching stays linear.
_VALUE_CHAR = r"""(?:[^&\s"'<>)\]},;\\]|\\(?!["'nrt]))"""
_VALUE = rf"{_VALUE_CHAR}+"
# The same, inside a percent-encoded URL: `%26` is an encoded `&`.
_ENCODED_VALUE = rf"(?:(?!%26){_VALUE_CHAR})+"
# What sits between a pair's opening and closing quote: anything on the same line.
# It starts with a character that is neither a space nor a quote. The first rule
# keeps a string literal that merely ends in the name (`"/?api_key=" + api_key`)
# from being read as an opening quote. The second is what makes an empty value
# (`token=""`) stay empty: otherwise its closing quote is taken for the first
# character and the match runs on to the next quote on the line.
_QUOTED_VALUE = rf"(?P<oq>{_Q}){_SCHEME}(?P<val>(?!{_Q})[^\s][^\r\n]*?)(?P=oq)"
# Query parameters that sign or authorize a link without being named like a
# secret: Roboflow's export URL carries `?key=<token>`. Only right after `?` or
# `&`, so "the primary key" and "key=value pairs" in prose are left alone.
_SIGNING_PARAM = r"(?:key|sig|signature|x-goog-signature|x-amz-signature)"
_HEADER = r"(?:proxy-)?authorization|x-api-key|x-auth-token|x-access-token"
_OPT_QUOTE = r"""(?:\\?["'])?"""

_PATTERNS: tuple[re.Pattern[str], ...] = (
    # `api_key='my key'`, `api_key=\"KEY\"` — a quoted value, masked up to its
    # closing quote on the same line, with the quotes kept. Before the unquoted
    # form, so a value with spaces is not cut at the first one.
    re.compile(rf"{_NAME}{_SP}={_SP}{_QUOTED_VALUE}", re.IGNORECASE),
    # `?api_key=KEY`, `&token=KEY`, `access_token=KEY` — a URL or a form body —
    # and an unquoted or unterminated `api_key='KEY`.
    re.compile(
        rf"{_NAME}{_SP}={_SP}(?:{_Q})?{_SCHEME}(?P<val>{_VALUE})", re.IGNORECASE
    ),
    # `?key=TOKEN`, `&sig=…` — a signed link.
    re.compile(rf"(?<=[?&]){_SIGNING_PARAM}={_SCHEME}(?P<val>{_VALUE})", re.IGNORECASE),
    # `api_key%3DKEY`, the same pair inside a percent-encoded URL.
    re.compile(rf"{_NAME}%3D(?P<val>{_ENCODED_VALUE})", re.IGNORECASE),
    # `password: "my phrase"` — a log line or YAML-ish dump, quoted…
    re.compile(rf"{_COLON_NAME}{_SP}:{_SP}{_QUOTED_VALUE}", re.IGNORECASE),
    # …and `password: hunter22`, unquoted or unterminated.
    re.compile(
        rf"{_COLON_NAME}{_SP}:{_SP}(?:{_Q})?{_SCHEME}(?P<val>{_VALUE})", re.IGNORECASE
    ),
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
    # `https://user:PASSWORD@host`, as in a proxy URL. The scheme is capped: an
    # unbounded `[a-z0-9+.-]*` made every position of a long `a-a-a-…` run scan to
    # the end of it, which is quadratic on text nobody controls.
    re.compile(
        r"\b[a-z][a-z0-9+.-]{0,30}://[^/\s:@]+:(?P<val>[^/\s@]+)@", re.IGNORECASE
    ),
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
