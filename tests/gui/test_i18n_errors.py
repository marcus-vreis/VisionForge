"""Unexpected 500s and request validation 422s speak the page's language (#208).

A 500 used to be the bare ``RuntimeError: x`` of whatever broke, and a 422 the
English sentence Pydantic wrote. Both now carry a frame in the language of the
``X-VF-Lang`` header; the technical part (the exception, the field path) stays
as it was, so a bug report still names the cause.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from visionforge.gui.api.error_messages import (
    translate_validation_errors,
    unexpected_error,
)
from visionforge.utils.messages import message, using_lang

PT = {"X-VF-Lang": "pt"}
EN = {"X-VF-Lang": "en"}
_TECH = "RuntimeError: x"


@pytest.fixture
def run_test_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """POST /api/runs/{id}/test on a run whose test step raises RuntimeError('x')."""
    from visionforge.gui.api import evaluation_routes
    from visionforge.gui.api import routes as routes_mod
    from visionforge.gui.server import app

    run_dir = tmp_path / "models" / "e" / "20260923_000000_000000"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"experiment": "e", "config": {"task": "multiclass"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(routes_mod, "_MODELS_DIR", tmp_path / "models")

    def boom(_run_dir: Path, _req: Any) -> None:
        raise RuntimeError("x")

    monkeypatch.setattr(evaluation_routes, "_execute_run_test", boom)
    return TestClient(app)


class TestUnexpectedError:
    URL = "/api/runs/20260923_000000_000000/test"

    def test_portuguese_frame_keeps_the_exception(
        self, run_test_client: TestClient
    ) -> None:
        resp = run_test_client.post(self.URL, json={"data_dir": "x"}, headers=PT)

        assert resp.status_code == 500
        assert resp.json()["detail"] == message("pt", "errors.unexpected", error=_TECH)
        assert resp.json()["detail"].startswith("Algo deu errado")

    def test_english_frame_keeps_the_exception(
        self, run_test_client: TestClient
    ) -> None:
        resp = run_test_client.post(self.URL, json={"data_dir": "x"}, headers=EN)

        assert resp.status_code == 500
        assert resp.json()["detail"] == message("en", "errors.unexpected", error=_TECH)
        assert resp.json()["detail"].endswith(_TECH)

    def test_helper_takes_an_already_redacted_detail(self) -> None:
        with using_lang("en"):
            text = unexpected_error(RuntimeError("secret"), detail="RuntimeError: ***")

        assert "secret" not in text
        assert text.endswith("RuntimeError: ***")

    def test_no_route_builds_a_bare_500_from_the_exception(self) -> None:
        api = Path(__file__).parents[2] / "src" / "visionforge" / "gui" / "api"
        offenders = [
            p.name
            for p in api.glob("*.py")
            if 'HTTPException(500, f"' in p.read_text(encoding="utf-8")
        ]

        assert offenders == []


def _zero_epochs() -> dict[str, Any]:
    return {"name": "exp", "data": {"base_dir": "x"}, "training": {"epochs": 0}}


class TestValidationErrors:
    URL = "/api/experiment/run"

    def test_portuguese_translates_the_bound(self) -> None:
        from visionforge.gui.server import app

        resp = TestClient(app).post(self.URL, json=_zero_epochs(), headers=PT)

        assert resp.status_code == 422
        [err] = [e for e in resp.json()["detail"] if e["loc"][-1] == "epochs"]
        assert err["type"] == "greater_than_equal"
        assert err["msg"] == "Deve ser maior ou igual a 1"

    def test_english_keeps_pydantics_text(self) -> None:
        from visionforge.gui.server import app

        resp = TestClient(app).post(self.URL, json=_zero_epochs(), headers=EN)

        assert resp.status_code == 422
        [err] = [e for e in resp.json()["detail"] if e["loc"][-1] == "epochs"]
        assert err["msg"] == "Input should be greater than or equal to 1"

    def test_no_header_is_portuguese(self) -> None:
        from visionforge.gui.server import app

        resp = TestClient(app).post(self.URL, json=_zero_epochs())

        [err] = [e for e in resp.json()["detail"] if e["loc"][-1] == "epochs"]
        assert err["msg"] == "Deve ser maior ou igual a 1"

    @pytest.mark.parametrize(
        ("err", "expected"),
        [
            ({"type": "missing"}, "Campo obrigatório"),
            ({"type": "greater_than", "ctx": {"gt": 0}}, "Deve ser maior que 0"),
            ({"type": "less_than", "ctx": {"lt": 1}}, "Deve ser menor que 1"),
            (
                {"type": "less_than_equal", "ctx": {"le": 1}},
                "Deve ser menor ou igual a 1",
            ),
            (
                {"type": "string_too_short", "ctx": {"min_length": 1}},
                "Deve ter pelo menos 1 caractere(s)",
            ),
            (
                {"type": "too_long", "ctx": {"max_length": 3}},
                "Deve ter no máximo 3 item(ns)",
            ),
            ({"type": "int_parsing"}, "Deve ser um número inteiro"),
            ({"type": "extra_forbidden"}, "Campo não permitido"),
        ],
    )
    def test_common_types_in_portuguese(
        self, err: dict[str, Any], expected: str
    ) -> None:
        [out] = translate_validation_errors([{**err, "msg": "orig"}], "pt")

        assert out["msg"] == expected

    def test_unknown_type_keeps_the_original(self) -> None:
        err = {"type": "url_parsing", "msg": "Input should be a valid URL"}

        assert translate_validation_errors([err], "pt") == [err]

    def test_missing_context_keeps_the_original(self) -> None:
        err = {"type": "greater_than", "msg": "orig"}

        assert translate_validation_errors([err], "pt") == [err]

    def test_english_is_untouched(self) -> None:
        err = {"type": "missing", "msg": "Field required"}

        assert translate_validation_errors([err], "en") == [err]
