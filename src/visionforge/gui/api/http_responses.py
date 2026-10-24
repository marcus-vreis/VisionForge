"""Error-response documentation shared by the routers (ADR-132).

Owns the OpenAPI text of the status codes more than one router declares, so a
router split out of ``routes.py`` documents them without importing it.
"""

from __future__ import annotations

from typing import Any

BAD_REQUEST: dict[int | str, dict[str, Any]] = {400: {"description": "Invalid request"}}
NOT_FOUND: dict[int | str, dict[str, Any]] = {
    404: {"description": "Resource not found"}
}
CONFLICT: dict[int | str, dict[str, Any]] = {
    409: {"description": "Operation conflicts with current state"}
}
SERVER_ERROR: dict[int | str, dict[str, Any]] = {
    500: {"description": "Operation failed"}
}
