"""The code tour of a researcher's own task (ADR-173, ADR-058).

The built-in tours (``code_tour.py``) point at the package's functions. A task
the researcher registered with ``@register_task`` has no such list, so this one
reads the hooks of its registered ``TaskSpec`` class with ``inspect``: the
loaders, the model, the loss, the metrics and, for a Level 2 task, its own
``run``. Nothing of theirs is executed here: the class is already imported by the
registry, and only its source text is read.

A hook whose source ``inspect`` cannot give (a builtin, a C extension, a file
deleted after the import) becomes a section holding one note that says so,
never an error: the rest of the tour is still shown.
"""

from __future__ import annotations

import inspect
import textwrap
from pathlib import Path
from typing import Any

from visionforge.core.code_tour import Note, Step, note_span
from visionforge.tasks.base import TaskSpec
from visionforge.utils.messages import tr

# (section id, hook name on ``TaskSpec``, step of the run it belongs to).
_HOOKS: tuple[tuple[str, str, Step], ...] = (
    ("data", "build_loaders", "data"),
    ("model", "build_model", "model"),
    ("loss", "compute_loss", "loss"),
    ("validate", "compute_metrics", "validate"),
)
_OWN_LOOP: tuple[str, str, Step] = ("train", "run", "train")


def custom_code_tour(spec_cls: type[TaskSpec]) -> list[dict[str, Any]]:
    """The sections of a researcher's task, in the shape ``GET /api/code-tour`` sends.

    A Level 2 task (its own ``run``) gets one more, the ``train`` step.
    """
    hooks = list(_HOOKS)
    if spec_cls.run is not TaskSpec.run:
        hooks.append(_OWN_LOOP)
    return [_section(spec_cls, *hook) for hook in hooks]


def _section(
    spec_cls: type[TaskSpec], section_id: str, hook: str, step: Step
) -> dict[str, Any]:
    qualified = f"{spec_cls.__qualname__}.{hook}"
    target = getattr(spec_cls, hook, None)
    try:
        raw, first_line = inspect.getsourcelines(target)  # type: ignore[arg-type]
    except (OSError, TypeError):
        return _without_source(section_id, step, hook, qualified)
    lines = textwrap.dedent("".join(raw)).rstrip("\n").split("\n")
    note = Note(f"def {hook}(", f"code.custom.note.{section_id}")
    try:
        first, last = note_span(note, lines)
    except ValueError:  # a hook bound to a function of another name
        first = last = 0
    return {
        "id": section_id,
        "step": step,
        "summary": tr(f"code.custom.summary.{section_id}"),
        "name": getattr(target, "__qualname__", qualified),
        "file": _file(target),
        "first_line": first_line,
        "lines": lines,
        "notes": [_note(first, last, tr(note.key))],
        "notes_resolved": 1,
        "notes_total": 1,
    }


def _without_source(
    section_id: str, step: Step, hook: str, qualified: str
) -> dict[str, Any]:
    """A section that only says its code could not be read."""
    return {
        "id": section_id,
        "step": step,
        "summary": tr(f"code.custom.summary.{section_id}"),
        "name": qualified,
        "file": "",
        "first_line": 1,
        "lines": [f"# {hook}"],
        "notes": [_note(0, 0, tr("code.custom.no_source", name=qualified))],
        "notes_resolved": 1,
        "notes_total": 1,
    }


def _note(first: int, last: int, text: str) -> dict[str, Any]:
    return {"first": first, "last": last, "text": text, "active": None}


def _file(target: Any) -> str:
    """The folder and name of the researcher's file, never an absolute path."""
    try:
        path = inspect.getsourcefile(target)
    except TypeError:
        path = None
    return Path(*Path(path).parts[-2:]).as_posix() if path else ""
