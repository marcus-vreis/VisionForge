"""HTTP contract of the GUI API, snapshotted from the app's OpenAPI document (#54).

Each ``(method, path)`` is stored with its operationId, tags, parameters, request
body schema and response schemas in ``fixtures/openapi_paths.json``. Moving endpoints
between modules must leave it untouched; an intended API change updates it on
purpose:

    VF_UPDATE_OPENAPI_CONTRACT=1 .venv/Scripts/python.exe -m pytest tests/gui/test_openapi_contract.py --no-cov

Review the fixture diff in the same PR: it is the list of endpoints that changed.
"""

from __future__ import annotations

import ast
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import visionforge.gui
from visionforge.gui.server import app

FIXTURE = Path(__file__).parent / "fixtures" / "openapi_paths.json"
_METHODS = {"get", "post", "put", "patch", "delete"}
_GUI_DIR = Path(visionforge.gui.__file__).parent
_ROUTE_FILES = [*sorted((_GUI_DIR / "api").glob("*.py")), _GUI_DIR / "server.py"]
_Handler = ast.FunctionDef | ast.AsyncFunctionDef


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
                "tags": op.get("tags", []),
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


def test_resume_errors_are_documented() -> None:
    """Continuation clients can discover missing, busy and unreadable runs."""
    responses = app.openapi()["paths"]["/api/runs/{run_id}/resume"]["post"]["responses"]
    assert {"404", "409", "500"} <= responses.keys()


def test_sample_dataset_conflict_is_documented() -> None:
    """Clients can discover the existing-folder response carrying a usable path."""
    responses = app.openapi()["paths"]["/api/sample-dataset"]["post"]["responses"]
    assert {"400", "409", "500"} <= responses.keys()


def _operations(openapi: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    for path, item in sorted(openapi["paths"].items()):
        for method, op in sorted(item.items()):
            if method in _METHODS:
                yield f"{method.upper()} {path}", op


def test_every_operation_has_a_declared_tag() -> None:
    """Each operation is grouped under a domain tag listed in ``openapi_tags`` (#205)."""
    openapi = app.openapi()
    declared = {tag["name"] for tag in openapi.get("tags", [])}
    wrong = [
        f"{key} {op.get('tags')}"
        for key, op in _operations(openapi)
        if not op.get("tags") or not set(op["tags"]) <= declared
    ]
    assert wrong == []


def test_every_declared_tag_is_used() -> None:
    openapi = app.openapi()
    used = {tag for _, op in _operations(openapi) for tag in op.get("tags", [])}
    declared = [tag["name"] for tag in openapi.get("tags", [])]
    assert declared != []
    assert sorted(set(declared) - used) == []


def _raised_statuses(node: ast.AST) -> Iterator[tuple[int, int]]:
    """``(status, line)`` of every ``raise HTTPException(<int>, ...)`` under ``node``."""
    for raise_ in ast.walk(node):
        if not isinstance(raise_, ast.Raise) or not isinstance(raise_.exc, ast.Call):
            continue
        call = raise_.exc
        if (
            getattr(call.func, "id", getattr(call.func, "attr", None))
            != "HTTPException"
        ):
            continue
        status = call.args[0] if call.args else None
        for keyword in call.keywords:
            if keyword.arg == "status_code":
                status = keyword.value
        if isinstance(status, ast.IfExp):
            options: list[ast.expr | None] = [status.body, status.orelse]
        else:
            options = [status]
        for option in options:
            if isinstance(option, ast.Constant) and isinstance(option.value, int):
                yield option.value, raise_.lineno


def _route_handlers(tree: ast.Module) -> Iterator[tuple[_Handler, ast.Call, str]]:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr in _METHODS
            ):
                yield node, decorator, decorator.func.attr


def _literal_statuses(decorator: ast.Call) -> set[int]:
    """Status keys written out in ``responses={...}``; a ``**spread`` counts for none.

    Mirrors what Sonar's python:S8415 reads: it does not follow a dict built
    elsewhere, so only a literal key documents a raised status.
    """
    for keyword in decorator.keywords:
        if keyword.arg == "responses" and isinstance(keyword.value, ast.Dict):
            return {
                key.value
                for key in keyword.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, int)
            }
    return set()


def _handler_raises(
    path: Path,
) -> Iterator[tuple[str, str, ast.Call, set[tuple[int, int]]]]:
    """``(handler, method, decorator, raised)`` for each route in ``path``.

    ``raised`` also holds what the module's own helpers raise when the handler
    calls them, directly or through another helper.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions: dict[str, list[_Handler]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.setdefault(node.name, []).append(node)

    def reach(name: str, seen: set[str]) -> set[tuple[int, int]]:
        found: set[tuple[int, int]] = set()
        for definition in functions[name]:
            found |= set(_raised_statuses(definition))
            for call in ast.walk(definition):
                callee = getattr(call, "func", None)
                if (
                    isinstance(callee, ast.Name)
                    and callee.id in functions
                    and callee.id not in seen
                ):
                    seen.add(callee.id)
                    found |= reach(callee.id, seen)
        return found

    for handler, decorator, method in _route_handlers(tree):
        yield handler.name, method, decorator, reach(handler.name, {handler.name})


def test_raised_statuses_are_documented_in_the_route() -> None:
    """Every HTTPException a route can raise is a literal key of its ``responses`` (#205)."""
    operations = [op for _, op in _operations(app.openapi())]
    undocumented: list[str] = []
    unmounted: list[str] = []
    handlers = 0
    for path in _ROUTE_FILES:
        for name, method, decorator, raised in _handler_raises(path):
            handlers += 1
            literal = _literal_statuses(decorator)
            undocumented += [
                f"{path.name}:{line} {name} raises {status}"
                for status, line in sorted(raised)
                if status not in literal
            ]
            statuses = {str(status) for status, _ in raised}
            # One handler name can be mounted under several prefixes (the token
            # routes): one of its operations documents everything it raises.
            if not any(
                str(op.get("operationId", "")).startswith(f"{name}_")
                and str(op.get("operationId", "")).endswith(f"_{method}")
                and statuses <= set(op["responses"])
                for op in operations
            ):
                unmounted.append(f"{path.name}:{decorator.lineno} {name}")
    assert handlers > 100
    assert undocumented == []
    assert unmounted == []
