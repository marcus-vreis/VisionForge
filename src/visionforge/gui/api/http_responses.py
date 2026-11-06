"""Error-response documentation shared by the routers (ADR-132).

Owns the OpenAPI text of the status codes more than one router declares, so a
router split out of ``routes.py`` documents them without importing it.

The ``*_TEXT`` strings are for routes that spell their ``responses=`` out with
literal status keys (``{400: {"description": BAD_REQUEST_TEXT}}``): a static
analyser reads the keys of such a literal, which it cannot do through a
``{**BAD_REQUEST}`` spread of an imported dict (Sonar S8415).
"""

from __future__ import annotations

from typing import Any

BAD_REQUEST_TEXT = "Invalid request"
FORBIDDEN_TEXT = "Access denied"
NOT_FOUND_TEXT = "Resource not found"
CONFLICT_TEXT = "Operation conflicts with current state"
SERVER_ERROR_TEXT = "Operation failed"

BAD_REQUEST: dict[int | str, dict[str, Any]] = {400: {"description": BAD_REQUEST_TEXT}}
FORBIDDEN: dict[int | str, dict[str, Any]] = {403: {"description": FORBIDDEN_TEXT}}
NOT_FOUND: dict[int | str, dict[str, Any]] = {404: {"description": NOT_FOUND_TEXT}}
CONFLICT: dict[int | str, dict[str, Any]] = {409: {"description": CONFLICT_TEXT}}
SERVER_ERROR: dict[int | str, dict[str, Any]] = {
    500: {"description": SERVER_ERROR_TEXT}
}
