"""The real code of a training, cut into the sections the "Código" tab shows (ADR-173).

Nothing here is a simplified copy: each section is the source of the function
the run calls, read with ``inspect`` when asked, so it cannot drift from what
trains. A note points at lines by their text, never by number; a test fails
when a change to the code leaves a note without its line. A note with
conditions marks the branch this run's settings take.
"""

from __future__ import annotations

import inspect
import textwrap
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visionforge.utils.messages import tr

# Settings a note can depend on, and their values.
SETTINGS: dict[str, frozenset[str]] = {
    "kind": frozenset({"binary", "multiclass"}),
    "amp": frozenset({"on", "off"}),
    "augment": frozenset({"on", "off"}),
    "family": frozenset({"resnet", "efficientnet", "vit", "swin", "convnext", "vgg"}),
}


@dataclass(frozen=True)
class Note:
    """An explanation of the lines from ``start`` up to (not including) ``end``.

    ``start`` is the text of the first line, and ``nth`` picks its occurrence
    when the text repeats. ``end`` is the text of the first line after the
    span, searched after ``start``; without it the note covers one line.
    ``when`` names the settings under which this is the path the run takes.
    """

    start: str
    key: str
    end: str | None = None
    nth: int = 1
    when: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Section:
    """One function of the training, shown whole, with its notes."""

    id: str
    target: Callable[[], Callable[..., Any]]
    notes: tuple[Note, ...]


def _transforms() -> Callable[..., Any]:
    from visionforge.core.data import _build_transforms

    return _build_transforms


def _final_layer() -> Callable[..., Any]:
    from visionforge.models.factory import replace_final_layer

    return replace_final_layer


def _criterion() -> Callable[..., Any]:
    from visionforge.core.trainer import Trainer

    return Trainer._build_criterion


def _train_epoch() -> Callable[..., Any]:
    from visionforge.core.trainer import Trainer

    return Trainer._train_epoch


def _eval_epoch() -> Callable[..., Any]:
    from visionforge.core.trainer import Trainer

    return Trainer._eval_epoch


def _evaluate() -> Callable[..., Any]:
    from visionforge.core.evaluator import Evaluator

    return Evaluator.evaluate


_BINARY = (("kind", "binary"),)
_MULTICLASS = (("kind", "multiclass"),)


def _family(name: str, start: str, end: str) -> Note:
    return Note(start, "code.model.family", end=end, when=(("family", name),))


CLASSIFICATION: tuple[Section, ...] = (
    Section(
        "data",
        _transforms,
        (
            Note("T.Resize(config.image_size),", "code.data.resize", end="if is_train"),
            Note(
                "if is_train and config.augment:",
                "code.data.augment",
                end="T.ToTensor(),",
                when=(("augment", "on"),),
            ),
            Note("T.ToTensor(),", "code.data.tensor", end="return T.Compose"),
        ),
    ),
    Section(
        "model",
        _final_layer,
        (
            _family("resnet", 'if name.startswith("resnet"):', "elif"),
            _family("efficientnet", 'elif name.startswith("efficientnet"):', "elif"),
            _family("vit", 'elif name.startswith("vit"):', "elif"),
            _family("swin", 'elif name.startswith("swin"):', "elif"),
            _family("convnext", 'elif name.startswith("convnext"):', "elif"),
            Note(
                'elif name.startswith("vgg") or name == "alexnet":',
                "code.model.family",
                end="clf[6] = nn.Linear",
                when=(("family", "vgg"),),
            ),
        ),
    ),
    Section(
        "loss",
        _criterion,
        (
            Note("return nn.BCEWithLogitsLoss()", "code.loss.binary", when=_BINARY),
            Note(
                "return nn.CrossEntropyLoss()", "code.loss.multiclass", when=_MULTICLASS
            ),
        ),
    ),
    Section(
        "train",
        _train_epoch,
        (
            Note("model.train()", "code.train.mode"),
            Note(
                "for inputs, labels in loader:",
                "code.train.batch",
                end='if self._config.task == "binary":',
            ),
            Note(
                "target = labels.float().unsqueeze(1)",
                "code.train.binary_target",
                when=_BINARY,
            ),
            Note("optimizer.zero_grad(set_to_none=True)", "code.train.zero_grad"),
            Note("if use_amp:", "code.train.amp", end="else:", when=(("amp", "on"),)),
            Note(
                "outputs = model(inputs)",
                "code.train.step",
                end="total_loss += loss.item()",
                nth=2,
                when=(("amp", "off"),),
            ),
            Note("with torch.no_grad():", "code.train.count", end="n = len(loader)"),
            Note(
                "preds = (outputs.sigmoid() > 0.5).squeeze(1).long()",
                "code.predict.binary",
                when=_BINARY,
            ),
            Note(
                "preds = outputs.argmax(dim=1)",
                "code.predict.multiclass",
                when=_MULTICLASS,
            ),
        ),
    ),
    Section(
        "validate",
        _eval_epoch,
        (
            Note("model.eval()", "code.validate.mode"),
            Note("with torch.no_grad():", "code.validate.no_grad"),
            Note("total_loss += loss.item()", "code.validate.loss"),
        ),
    ),
    Section(
        "test",
        _evaluate,
        (
            Note("model.eval()", "code.test.once"),
            Note(
                "pos = outputs.sigmoid().squeeze(1)",
                "code.predict.binary",
                end="else:",
                when=_BINARY,
            ),
            Note(
                "softmax = outputs.softmax(dim=1)",
                "code.predict.multiclass",
                end="all_preds.extend",
                when=_MULTICLASS,
            ),
            Note(
                'avg = "binary" if self._config.task == "binary" else "macro"',
                "code.test.metrics",
                end="return EvalResult(",
            ),
        ),
    ),
)

TOURS: dict[str, tuple[Section, ...]] = {"classification": CLASSIFICATION}


def model_family(model_name: str) -> str | None:
    """The branch of ``replace_final_layer`` a model name takes, if any."""
    if model_name == "alexnet":
        return "vgg"
    for family in sorted(SETTINGS["family"]):
        if model_name.startswith(family):
            return family
    return None


def code_tour(task: str, settings: Mapping[str, str]) -> list[dict[str, Any]]:
    """The sections of ``task`` with their source and notes, in the request's language.

    ``settings`` are what the run uses (``kind``, ``amp``, ``augment``,
    ``family``); one missing or unknown leaves its notes undecided.

    Raises:
        KeyError: a task without a tour.
    """
    known = {
        key: value
        for key, value in settings.items()
        if key in SETTINGS and value in SETTINGS[key]
    }
    return [_section(section, known) for section in TOURS[task]]


def _section(section: Section, settings: Mapping[str, str]) -> dict[str, Any]:
    target = section.target()
    raw, first_line = inspect.getsourcelines(target)
    lines = textwrap.dedent("".join(raw)).rstrip("\n").split("\n")
    return {
        "id": section.id,
        "name": target.__qualname__,
        "file": _module_file(target),
        "first_line": first_line,
        "lines": lines,
        "notes": [_note(note, lines, settings) for note in section.notes],
    }


def _note(note: Note, lines: list[str], settings: Mapping[str, str]) -> dict[str, Any]:
    start, end = note_span(note, lines)
    family = dict(note.when).get("family")
    return {
        "first": start,
        "last": end,
        "text": tr(note.key, family=family) if family else tr(note.key),
        "active": _active(note, settings),
    }


def note_span(note: Note, lines: list[str]) -> tuple[int, int]:
    """``(first, last)`` 0-based line indexes the note covers, inclusive.

    Raises:
        ValueError: the start text is not found ``nth`` times, or the end
            text is not found after it.
    """
    hits = [i for i, line in enumerate(lines) if note.start in line]
    if len(hits) < note.nth:
        raise ValueError(f"Note start not found: {note.start!r}")
    start = hits[note.nth - 1]
    if note.end is None:
        return start, start
    for index in range(start + 1, len(lines)):
        if note.end in lines[index]:
            return start, index - 1
    raise ValueError(f"Note end not found after {note.start!r}: {note.end!r}")


def _active(note: Note, settings: Mapping[str, str]) -> bool | None:
    """True on this run's path, False off it, None when unconditional or unknown."""
    if not note.when:
        return None
    if any(key not in settings for key, _ in note.when):
        return None
    return all(settings[key] == value for key, value in note.when)


def _module_file(target: Callable[..., Any]) -> str:
    """The file inside the package, never the absolute path of this install."""
    module = inspect.getmodule(target)
    name = module.__name__ if module else ""
    return Path(*name.split(".")).with_suffix(".py").as_posix()
