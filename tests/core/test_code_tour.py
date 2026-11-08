"""The code tour (ADR-173): real source, notes that still find their lines."""

from __future__ import annotations

import pytest

from visionforge.core.code_tour import (
    TOURS,
    Note,
    code_tour,
    model_family,
    note_span,
)
from visionforge.utils.messages import message

_LINES = [
    "def f(x):",
    "    if x:",
    "        a = 1",
    "    else:",
    "        a = 2",
    "    return a",
]


@pytest.mark.parametrize("task", sorted(TOURS))
def test_every_note_still_finds_its_lines(task: str) -> None:
    # The guard of the whole feature: a change to the trainer that moves or
    # rewrites a commented line fails here instead of pointing at nothing.
    for section in code_tour(task, {}):
        assert section["lines"], section["id"]
        for note in section["notes"]:
            assert 0 <= note["first"] <= note["last"] < len(section["lines"])


@pytest.mark.parametrize("task", sorted(TOURS))
def test_every_note_has_text_in_both_languages(task: str) -> None:
    for section in TOURS[task]:
        for note in section.notes:
            family = dict(note.when).get("family")
            params = {"family": family} if family else {}
            assert message("pt", note.key, **params)
            assert message("en", note.key, **params)


def test_the_files_are_inside_the_package_never_absolute() -> None:
    files = {section["file"] for section in code_tour("classification", {})}
    assert files == {
        "visionforge/core/data.py",
        "visionforge/models/factory.py",
        "visionforge/core/trainer.py",
        "visionforge/core/evaluator.py",
    }


def test_the_source_is_the_function_the_trainer_runs() -> None:
    from visionforge.core.trainer import Trainer

    train = next(s for s in code_tour("classification", {}) if s["id"] == "train")
    assert train["name"] == "Trainer._train_epoch"
    assert train["lines"][0].startswith("def _train_epoch(")
    assert train["first_line"] == Trainer._train_epoch.__code__.co_firstlineno


def test_notes_of_the_branch_the_run_takes_are_active() -> None:
    settings = {"kind": "binary", "amp": "off", "augment": "on", "family": "resnet"}
    sections = {s["id"]: s for s in code_tour("classification", settings)}
    train = sections["train"]["notes"]
    lines = sections["train"]["lines"]

    def active(text: str) -> bool | None:
        (note,) = [n for n in train if text in lines[n["first"]]]
        flag: bool | None = note["active"]
        return flag

    assert active("target = labels.float().unsqueeze(1)") is True
    assert active("preds = outputs.argmax(dim=1)") is False
    assert active("if use_amp:") is False
    assert active("model.train()") is None
    family = [n["active"] for n in sections["model"]["notes"]]
    assert family.count(True) == 1
    assert family[0] is True  # resnet is the first branch


def test_without_settings_no_branch_is_decided() -> None:
    for section in code_tour("classification", {"kind": "neither", "amp": "maybe"}):
        assert all(note["active"] is None for note in section["notes"])


def test_the_amp_note_stops_before_the_else() -> None:
    train = next(s for s in code_tour("classification", {}) if s["id"] == "train")
    lines = train["lines"]
    (amp,) = [n for n in train["notes"] if "if use_amp:" in lines[n["first"]]]
    assert "scaler.update()" in lines[amp["last"]]


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("resnet18", "resnet"),
        ("efficientnet_b0", "efficientnet"),
        ("vit_b_16", "vit"),
        ("swin_t", "swin"),
        ("convnext_tiny", "convnext"),
        ("vgg16", "vgg"),
        ("alexnet", "vgg"),
        ("mobilenet_v3_small", None),
    ],
)
def test_model_family(name: str, family: str | None) -> None:
    assert model_family(name) == family


def test_note_span_counts_occurrences_and_ends_before_the_end_text() -> None:
    assert note_span(Note("a = ", "k", nth=2), _LINES) == (4, 4)
    assert note_span(Note("if x:", "k", end="else:"), _LINES) == (1, 2)


@pytest.mark.parametrize(
    "note",
    [Note("missing", "k"), Note("a = ", "k", nth=3), Note("if x:", "k", end="never")],
)
def test_note_span_refuses_a_note_that_lost_its_line(note: Note) -> None:
    with pytest.raises(ValueError):
        note_span(note, _LINES)
