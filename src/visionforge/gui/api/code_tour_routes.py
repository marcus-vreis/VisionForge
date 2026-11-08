"""GET /api/code-tour/{task}: the real code of a training, for the "Código" tab (ADR-173)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from visionforge.core.code_tour import TOURS, code_tour, model_family
from visionforge.gui.api.http_responses import NOT_FOUND_TEXT
from visionforge.utils.messages import tr

router = APIRouter(prefix="/code-tour")


class CodeNote(BaseModel):
    """Lines ``first`` to ``last`` (0-based, inclusive) and what they do.

    ``active`` is True on this run's path, False on a branch it does not take,
    and None when the note is unconditional or the setting is unknown.
    """

    first: int
    last: int
    text: str
    active: bool | None


class CodeSection(BaseModel):
    """One function of the training, as it is in the installed package."""

    id: str
    name: str
    file: str
    first_line: int
    lines: list[str]
    notes: list[CodeNote]


@router.get("/{task}", responses={404: {"description": NOT_FOUND_TEXT}})
def get_code_tour(
    task: str,
    kind: str | None = None,
    amp: bool | None = None,
    augment: bool | None = None,
    model: str | None = None,
) -> list[CodeSection]:
    """The sections of ``task``'s training code, with notes for this run's settings.

    Every setting is optional: one left out leaves its notes undecided instead
    of guessing which branch the run takes.
    """
    if task not in TOURS:
        raise HTTPException(404, tr("code.tour_missing"))
    settings: dict[str, str] = {}
    if kind is not None:
        settings["kind"] = kind
    if amp is not None:
        settings["amp"] = "on" if amp else "off"
    if augment is not None:
        settings["augment"] = "on" if augment else "off"
    family = model_family(model) if model else None
    if family is not None:
        settings["family"] = family
    return [CodeSection.model_validate(item) for item in code_tour(task, settings)]
