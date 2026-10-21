"""HTTP contract of the GUI API, snapshotted from the app's OpenAPI document (#54).

Each ``(method, path)`` is stored with its operationId, parameters, request body
schema and response schemas in ``fixtures/openapi_paths.json``. Moving endpoints
between modules must leave it untouched; an intended API change updates it on
purpose:

    VF_UPDATE_OPENAPI_CONTRACT=1 .venv/Scripts/python.exe -m pytest tests/gui/test_openapi_contract.py --no-cov

Review the fixture diff in the same PR: it is the list of endpoints that changed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from visionforge.gui.server import app

FIXTURE = Path(__file__).parent / "fixtures" / "openapi_paths.json"
_METHODS = {"get", "post", "put", "patch", "delete"}


def _schema_name(schema: dict[str, Any] | None) -> str | None:
    """Name of the referenced component, or the inline type when there is none."""
    if not schema:
        return None
    if "$ref" in schema:
        return str(schema["$ref"]).rsplit("/", 1)[-1]
    if "items" in schema:
        return f"array[{_schema_name(schema['items'])}]"
    for key in ("anyOf", "oneOf"):
        if key in schema:
            return "|".join(sorted(str(_schema_name(s)) for s in schema[key]))
    return str(schema.get("type", "inline"))


def _content_schemas(content: dict[str, Any] | None) -> dict[str, str | None]:
    """Media type -> schema name, sorted so the snapshot is deterministic."""
    return {
        media: _schema_name(body.get("schema"))
        for media, body in sorted((content or {}).items())
    }


def snapshot(openapi: dict[str, Any]) -> dict[str, Any]:
    """Reduce an OpenAPI document to what clients depend on, in sorted order."""
    out: dict[str, Any] = {}
    for path, item in sorted(openapi["paths"].items()):
        for method, op in sorted(item.items()):
            if method not in _METHODS:
                continue
            params = sorted(
                f"{p['in']}:{p['name']}:{'required' if p.get('required') else 'optional'}"
                for p in op.get("parameters", [])
            )
            out[f"{method.upper()} {path}"] = {
                "operationId": op.get("operationId"),
                "parameters": params,
                "requestBody": _content_schemas(
                    (op.get("requestBody") or {}).get("content")
                ),
                "responses": {
                    status: _content_schemas(resp.get("content"))
                    for status, resp in sorted(op.get("responses", {}).items())
                },
            }
    return out


def test_openapi_contract_is_preserved() -> None:
    live = snapshot(app.openapi())
    if os.environ.get("VF_UPDATE_OPENAPI_CONTRACT") == "1":
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(
            json.dumps(live, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert sorted(live) == sorted(expected), "endpoints added or removed"
    for key in sorted(expected):
        assert live[key] == expected[key], f"contract changed for {key}"


def test_profile_endpoints_are_in_the_contract() -> None:
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for key in ("GET /api/profiles", "POST /api/profiles", "PUT /api/profiles/{slug}"):
        assert key in expected
