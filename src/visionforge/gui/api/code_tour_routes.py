"""The real code of a training, for the "Código" tab (ADR-173).

``GET /api/code-tour`` lists the tasks with a tour and the settings each one
reads; ``GET /api/code-tour/{task}`` sends the sections of one of them.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from visionforge.core.code_tour import (
    TOUR_SETTINGS,
    TOURS,
    code_tour,
    tour_settings,
)
from visionforge.gui.api.http_responses import NOT_FOUND_TEXT, TAG_SYSTEM
from visionforge.utils.messages import tr

router = APIRouter(prefix="/code-tour", tags=[TAG_SYSTEM])


class CodeNote(BaseModel):
    """Lines ``first`` to ``last`` (0-based, inclusive) and what they do.

    ``active`` is True on this run's path, False on a branch it does not take,
    and None when the note is unconditional or the setting is unknown.
    """

    first: int
    last: int
    text: str
    active: bool | None


class CodeLibrary(BaseModel):
    """The third-party library a section's code belongs to (ADR-194)."""

    name: str
    version: str
    license: str


class CodeSection(BaseModel):
    """One function of the training, as it is in the installed package.

    ``notes_resolved`` of ``notes_total`` found their line: always all of them
    for VisionForge's code, possibly fewer in a library's other version.
    """

    id: str
    name: str
    file: str
    first_line: int
    lines: list[str]
    notes: list[CodeNote]
    notes_resolved: int
    notes_total: int
    library: CodeLibrary | None = None


class CodeTourTask(BaseModel):
    """A task with a tour, and the values of each setting its notes depend on."""

    task: str
    settings: dict[str, list[str]]


@router.get("")
def list_code_tours() -> list[CodeTourTask]:
    """The tasks whose training code the "Código" tab can show."""
    return [
        CodeTourTask(
            task=task,
            settings={
                key: sorted(values) for key, values in TOUR_SETTINGS[task].items()
            },
        )
        for task in TOURS
    ]


@router.get("/{task}", responses={404: {"description": NOT_FOUND_TEXT}})
def get_code_tour(task: str, request: Request) -> list[CodeSection]:
    """The sections of ``task``'s training code, with notes for this run's settings.

    The query carries the run's settings (``GET /api/code-tour`` lists each
    task's); ``model`` is a model name, read as the setting it decides. Every
    setting is optional: one left out, or with a value the task does not take,
    leaves its notes undecided instead of guessing which branch the run takes.

    Detection on Ultralytics shows VisionForge's layer, then the library's
    installed loop with its version and license (ADR-194).
    """
    if task not in TOURS:
        raise HTTPException(404, tr("code.tour_missing"))
    settings = tour_settings(task, request.query_params)
    return [CodeSection.model_validate(item) for item in code_tour(task, settings)]
