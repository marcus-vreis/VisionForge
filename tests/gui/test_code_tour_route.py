"""GET /api/code-tour/{task} (ADR-173)."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from visionforge.gui.server import app


def _tour(query: str = "", lang: str = "pt") -> Any:
    return TestClient(app).get(
        f"/api/code-tour/classification{query}", headers={"X-VF-Lang": lang}
    )


def test_the_classification_tour_lists_its_sections_in_order() -> None:
    response = _tour()
    assert response.status_code == 200
    assert [s["id"] for s in response.json()] == [
        "data",
        "model",
        "loss",
        "train",
        "validate",
        "test",
    ]


def test_the_settings_pick_the_branch_of_this_run() -> None:
    sections = {
        s["id"]: s
        for s in _tour("?kind=multiclass&amp=false&augment=true&model=resnet18").json()
    }
    loss = sections["loss"]
    actives = {loss["lines"][n["first"]].strip(): n["active"] for n in loss["notes"]}
    assert actives == {
        "return nn.BCEWithLogitsLoss()": False,
        "return nn.CrossEntropyLoss()": True,
    }
    model = [n["active"] for n in sections["model"]["notes"]]
    assert model == [True, False, False, False, False, False]


def test_an_unknown_model_leaves_the_branches_undecided() -> None:
    model = next(
        s for s in _tour("?model=mobilenet_v3_small").json() if s["id"] == "model"
    )
    assert all(note["active"] is None for note in model["notes"])


def test_the_notes_follow_the_language() -> None:
    pt = _tour(lang="pt").json()[0]["notes"][0]["text"]
    en = _tour(lang="en").json()[0]["notes"][0]["text"]
    assert pt != en
    assert "image" in en


def test_a_task_without_a_tour_is_404() -> None:
    response = TestClient(app).get("/api/code-tour/detection")
    assert response.status_code == 404
    assert response.json()["detail"]
