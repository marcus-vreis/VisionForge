"""The language frame of unexpected failures and of request validation (#208, ADR-191).

Two kinds of error used to reach the page in a language nobody chose: a 500 was
the bare ``RuntimeError: x`` of whatever broke, and a 422 carried the English
sentence Pydantic wrote for each field. Both now speak the language of the
``X-VF-Lang`` header (ADR-116). The technical part stays as it was: the
exception's class and text in a 500, the type, location and input of each
validation error, so a bug report still names the cause.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from visionforge.utils.messages import LANG_HEADER, Lang, message, resolve_lang, tr

# The Pydantic error types a form in the page commonly trips, each with the
# catalog key of its Portuguese sentence. The placeholders of those sentences are
# the names Pydantic puts in the error's ``ctx`` (``gt``, ``min_length`` ...).
# A type not listed here keeps Pydantic's English text.
_VALIDATION_KEYS: dict[str, str] = {
    "missing": "validation.missing",
    "extra_forbidden": "validation.extra_forbidden",
    "greater_than": "validation.greater_than",
    "greater_than_equal": "validation.greater_than_equal",
    "less_than": "validation.less_than",
    "less_than_equal": "validation.less_than_equal",
    "string_too_short": "validation.string_too_short",
    "string_too_long": "validation.string_too_long",
    "too_short": "validation.too_short",
    "too_long": "validation.too_long",
    "int_parsing": "validation.int_parsing",
    "float_parsing": "validation.float_parsing",
    "bool_parsing": "validation.bool_parsing",
    "literal_error": "validation.literal_error",
}


def unexpected_error(exc: BaseException, *, detail: str | None = None) -> str:
    """The text of a 500 for a failure nobody anticipated, in the page's language.

    ``detail`` replaces the default ``Type: message`` when the caller must clean
    it first (a provider key in the message, for instance).
    """
    technical = detail if detail is not None else f"{type(exc).__name__}: {exc}"
    return tr("errors.unexpected", error=technical)


def translate_validation_errors(
    errors: Sequence[Mapping[str, Any]], lang: Lang
) -> list[Mapping[str, Any]]:
    """``errors`` with the ``msg`` of the common types worded in ``lang``.

    English keeps Pydantic's own text, which is what the catalog's English mirrors.
    An error whose type is not listed, or whose context lacks a value the
    sentence needs, is returned unchanged rather than half-translated.
    """
    if lang == "en":
        return list(errors)
    translated: list[Mapping[str, Any]] = []
    for err in errors:
        key = _VALIDATION_KEYS.get(str(err.get("type")))
        if key is None:
            translated.append(err)
            continue
        try:
            text = message(lang, key, **dict(err.get("ctx") or {}))
        except (KeyError, IndexError, ValueError):
            translated.append(err)
            continue
        translated.append({**err, "msg": text})
    return translated


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """FastAPI's 422 body, with each message in the language of the request.

    The header is read here rather than from the bound language: a body that
    fails validation may never reach the dependency that binds it. ``exc`` is
    typed as Starlette's handler signature wants; the app registers this for
    ``RequestValidationError`` only.
    """
    validation = cast("RequestValidationError", exc)
    lang = resolve_lang(request.headers.get(LANG_HEADER))
    errors = translate_validation_errors(validation.errors(), lang)
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})
