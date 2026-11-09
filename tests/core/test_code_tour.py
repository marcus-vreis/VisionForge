"""The code tour (ADR-173): real source, notes that still find their lines."""

from __future__ import annotations

import pytest

from visionforge.core.code_tour import (
    TOUR_SETTINGS,
    TOURS,
    Note,
    code_tour,
    model_family,
    note_span,
    tour_settings,
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


def test_the_regression_tour_has_its_sections_and_every_note_resolves() -> None:
    sections = code_tour("regression", {})
    assert [s["id"] for s in sections] == [
        "data",
        "model",
        "loss",
        "train",
        "validate",
        "test",
    ]
    for section in sections:
        for note in section["notes"]:
            assert 0 <= note["first"] <= note["last"] < len(section["lines"])
            assert note["text"]


def test_the_regression_tour_is_the_code_the_regression_run_calls() -> None:
    from visionforge.core.regression_trainer import RegressionTrainer
    from visionforge.core.training_loop import train_epoch

    sections = {s["id"]: s for s in code_tour("regression", {})}
    assert sections["loss"]["name"] == "RegressionTrainer._build_criterion"
    assert sections["train"]["name"] == "train_epoch"
    assert sections["train"]["first_line"] == train_epoch.__code__.co_firstlineno
    assert (
        sections["validate"]["first_line"]
        == RegressionTrainer._eval_epoch.__code__.co_firstlineno
    )
    assert sections["model"]["file"] == "visionforge/models/regression_factory.py"


def _loss_actives(task: str, settings: dict[str, str]) -> dict[str, bool | None]:
    loss = next(s for s in code_tour(task, settings) if s["id"] == "loss")
    return {loss["lines"][n["first"]].strip(): n["active"] for n in loss["notes"]}


def test_the_regression_loss_marks_the_criterion_of_this_run() -> None:
    assert _loss_actives("regression", {"loss": "huber"}) == {
        "return nn.L1Loss()": False,
        "return nn.HuberLoss(delta=cfg.huber_delta)": True,
        "return nn.MSELoss()": False,
    }
    assert set(_loss_actives("regression", {"loss": "quantile"}).values()) == {None}


def test_tour_settings_keep_only_what_the_task_takes() -> None:
    query = {"kind": "binary", "amp": "false", "model": "resnet18", "loss": "mae"}
    assert tour_settings("classification", query) == {
        "kind": "binary",
        "amp": "off",
        "family": "resnet",
    }
    assert tour_settings("regression", query) == {"loss": "mae"}
    assert tour_settings("regression", {"loss": "quantile"}) == {}


def test_every_tour_declares_its_settings() -> None:
    assert set(TOUR_SETTINGS) == set(TOURS)
    for task, sections in TOURS.items():
        for section in sections:
            for note in section.notes:
                for key, value in note.when:
                    assert value in TOUR_SETTINGS[task][key], (task, note.key)


def test_the_segmentation_tour_resolves_every_note() -> None:
    sections = code_tour("segmentation", {})
    assert [s["id"] for s in sections] == [
        "data",
        "model",
        "unet",
        "loss",
        "train",
        "validate",
        "test",
        "metrics",
    ]
    for section in sections:
        for note in section["notes"]:
            assert 0 <= note["first"] <= note["last"] < len(section["lines"])
            assert note["text"]


def test_the_segmentation_loss_marks_the_criterion_of_this_run() -> None:
    assert _loss_actives("segmentation", {"loss": "combined"}) == {
        "return _DiceCriterion(self._ignore_index)": False,
        "return _CombinedLoss(self._ignore_index)": True,
        "return nn.CrossEntropyLoss(ignore_index=self._ignore_index)": False,
    }


@pytest.mark.parametrize(
    ("name", "arch"),
    [
        ("unet", "unet"),
        ("deeplabv3_resnet50", "torchvision"),
        ("lraspp_mobilenet_v3_large", "torchvision"),
        ("resnet18", None),
    ],
)
def test_the_segmentation_model_name_picks_the_architecture(
    name: str, arch: str | None
) -> None:
    expected = {"arch": arch} if arch else {}
    assert tour_settings("segmentation", {"model": name}) == expected


def test_a_torchvision_segmenter_leaves_the_unet_code_off_its_path() -> None:
    sections = {s["id"]: s for s in code_tour("segmentation", {"arch": "torchvision"})}
    assert [n["active"] for n in sections["model"]["notes"]] == [False, True]
    assert all(n["active"] is False for n in sections["unet"]["notes"])


def test_the_anomaly_tour_resolves_every_note() -> None:
    sections = code_tour("anomaly", {})
    assert [s["id"] for s in sections] == [
        "data",
        "model",
        "train",
        "coreset",
        "score",
        "patch_score",
        "threshold",
        "test",
        "metrics",
    ]
    for section in sections:
        for note in section["notes"]:
            assert 0 <= note["first"] <= note["last"] < len(section["lines"])
            assert note["text"]


def _method_actives(method: str) -> dict[str, set[bool | None]]:
    return {
        s["id"]: {n["active"] for n in s["notes"]}
        for s in code_tour("anomaly", {"method": method})
    }


def test_patchcore_marks_its_own_code_and_leaves_the_autoencoder_off() -> None:
    actives = _method_actives("patchcore")
    assert actives["coreset"] == {True}
    assert actives["patch_score"] == {True}
    assert actives["train"] == {False}
    assert actives["model"] == {True, False}
    assert actives["threshold"] == {None}


def test_the_autoencoder_marks_its_own_code_and_leaves_patchcore_off() -> None:
    actives = _method_actives("autoencoder")
    assert actives["train"] == {True}
    assert actives["coreset"] == {False}
    assert actives["patch_score"] == {False}
    score = next(
        s for s in code_tour("anomaly", {"method": "autoencoder"}) if s["id"] == "score"
    )
    lines = score["lines"]
    by_line = {lines[n["first"]].strip(): n["active"] for n in score["notes"]}
    assert by_line == {
        "if isinstance(model, ConvAutoencoder):": True,
        "if isinstance(model, PatchCore):": False,
    }
