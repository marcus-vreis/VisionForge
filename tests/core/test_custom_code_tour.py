"""The code tour of a researcher's own task, read from its registered class."""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from torch import nn

from visionforge.core.custom_code_tour import custom_code_tour
from visionforge.tasks import BaseTaskConfig, TaskSpec
from visionforge.utils.messages import tr, using_lang


class _Level1(TaskSpec):
    Config = BaseTaskConfig

    def build_model(self, cfg: Any) -> nn.Module:
        return nn.Linear(3, 1)

    def build_loaders(self, cfg: Any):
        return [], [], None

    def compute_loss(self, model: nn.Module, batch: Any, cfg: Any):
        return model(batch).sum()

    def compute_metrics(self, model: nn.Module, loader: Any, cfg: Any):
        return {"mae": 0.0}


class _Level2(_Level1):
    def run(self, cfg: Any, ctx: Any):
        return {"mae": 1.0}


def _by_id(sections: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {s["id"]: s for s in sections}


def test_level_one_shows_the_four_hooks_by_step() -> None:
    sections = custom_code_tour(_Level1)
    assert [(s["id"], s["step"]) for s in sections] == [
        ("data", "data"),
        ("model", "model"),
        ("loss", "loss"),
        ("validate", "validate"),
    ]


def test_the_lines_are_the_researchers_real_source() -> None:
    model = _by_id(custom_code_tour(_Level1))["model"]
    assert model["lines"][0] == "def build_model(self, cfg: Any) -> nn.Module:"
    assert "return nn.Linear(3, 1)" in model["lines"][1]
    assert model["name"] == "_Level1.build_model"
    assert model["file"].endswith("test_custom_code_tour.py")
    assert "\\" not in model["file"]
    assert ":" not in model["file"]
    assert model["first_line"] > 1


def test_each_section_has_a_summary_and_a_note_on_its_def_line() -> None:
    for section in custom_code_tour(_Level1):
        assert section["summary"]
        assert section["notes_resolved"] == section["notes_total"] == 1
        note = section["notes"][0]
        assert section["lines"][note["first"]].lstrip().startswith("def ")
        assert note["active"] is None
        assert note["text"]


def test_a_level_two_task_adds_its_own_loop_as_the_train_step() -> None:
    sections = _by_id(custom_code_tour(_Level2))
    assert sections["train"]["step"] == "train"
    assert sections["train"]["name"] == "_Level2.run"
    assert "return {" in sections["train"]["lines"][1]
    assert "train" not in _by_id(custom_code_tour(_Level1))


def test_a_hook_without_source_becomes_a_note_not_an_error() -> None:
    class _NoSource(_Level1):
        compute_metrics: Any = len  # a builtin: inspect has no source for it

    sections = _by_id(custom_code_tour(_NoSource))
    section = sections["validate"]
    assert section["step"] == "validate"
    assert section["notes_resolved"] == 1
    note = section["notes"][0]
    assert 0 <= note["first"] <= note["last"] < len(section["lines"])
    name = f"{_NoSource.__qualname__}.compute_metrics"
    assert note["text"] == tr("code.custom.no_source", name=name)
    # The other hooks are still read.
    assert "return nn.Linear" in sections["model"]["lines"][1]


def test_a_file_that_is_gone_becomes_a_note_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(_: object) -> Any:
        raise OSError("could not get source code")

    monkeypatch.setattr(inspect, "getsourcelines", boom)
    sections = custom_code_tour(_Level1)
    assert len(sections) == 4
    expected = tr("code.custom.no_source", name="_Level1.build_model")
    assert _by_id(sections)["model"]["notes"][0]["text"] == expected


def test_the_text_follows_the_language() -> None:
    with using_lang("pt"):
        pt = custom_code_tour(_Level1)[0]["summary"]
    with using_lang("en"):
        en = custom_code_tour(_Level1)[0]["summary"]
    assert pt != en
