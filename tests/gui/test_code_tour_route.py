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


def test_the_regression_tour_marks_the_loss_of_the_query() -> None:
    response = TestClient(app).get("/api/code-tour/regression?loss=mae")
    assert response.status_code == 200
    loss = next(s for s in response.json() if s["id"] == "loss")
    actives = {loss["lines"][n["first"]].strip(): n["active"] for n in loss["notes"]}
    assert actives["return nn.L1Loss()"] is True
    assert actives["return nn.MSELoss()"] is False


def test_the_list_names_the_tasks_with_a_tour_and_their_settings() -> None:
    response = TestClient(app).get("/api/code-tour")
    assert response.status_code == 200
    tours = {item["task"]: item["settings"] for item in response.json()}
    assert {"classification", "regression"} <= set(tours)
    assert tours["regression"] == {"loss": ["huber", "mae", "mse"]}
    assert tours["classification"]["kind"] == ["binary", "multiclass"]


def test_an_unknown_task_is_404() -> None:
    response = TestClient(app).get("/api/code-tour/foo")
    assert response.status_code == 404
    assert response.json()["detail"]


def test_the_segmentation_tour_marks_the_loss_and_is_listed() -> None:
    client = TestClient(app)
    response = client.get("/api/code-tour/segmentation?loss=dice&model=unet")
    assert response.status_code == 200
    sections = {s["id"]: s for s in response.json()}
    loss = sections["loss"]
    actives = {loss["lines"][n["first"]].strip(): n["active"] for n in loss["notes"]}
    assert actives["return _DiceCriterion(self._ignore_index)"] is True
    assert actives["return _CombinedLoss(self._ignore_index)"] is False
    assert all(n["active"] is True for n in sections["unet"]["notes"])
    tasks = [item["task"] for item in client.get("/api/code-tour").json()]
    assert "segmentation" in tasks


def test_the_anomaly_tour_follows_the_method_of_the_query() -> None:
    response = TestClient(app).get("/api/code-tour/anomaly?method=patchcore")
    assert response.status_code == 200
    sections = {s["id"]: s for s in response.json()}
    assert all(n["active"] is True for n in sections["coreset"]["notes"])
    assert all(n["active"] is False for n in sections["train"]["notes"])


def test_the_torchvision_detection_tour_marks_the_detector() -> None:
    response = TestClient(app).get(
        "/api/code-tour/detection?backend=torchvision&arch=retinanet"
    )
    assert response.status_code == 200
    sections = {s["id"]: s for s in response.json()}
    assert all(n["active"] is True for n in sections["retinanet"]["notes"])
    assert all(n["active"] is False for n in sections["ssd"]["notes"])


def test_ultralytics_detection_says_why_there_is_no_tour() -> None:
    client = TestClient(app)
    pt = client.get(
        "/api/code-tour/detection?backend=ultralytics&model=yolo11n",
        headers={"X-VF-Lang": "pt"},
    )
    en = client.get(
        "/api/code-tour/detection?backend=ultralytics",
        headers={"X-VF-Lang": "en"},
    )
    assert pt.status_code == en.status_code == 404
    assert "Ultralytics" in en.json()["detail"]
    assert pt.json()["detail"] != en.json()["detail"]
    tasks = [item["task"] for item in client.get("/api/code-tour").json()]
    assert "detection" in tasks
