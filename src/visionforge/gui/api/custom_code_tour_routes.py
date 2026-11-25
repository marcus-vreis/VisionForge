"""The code of a researcher's own task, for the "Código" tab (ADR-173, ADR-058).

``GET /api/custom/{key}/code-tour`` sends the sections of a registered custom
task, read with ``inspect`` from its own class. ``custom_task_routes`` includes
this router. The route is ``off`` in server mode, like the rest of the custom-task
routes: it opens code a user brought (ADR-212).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from visionforge.core.custom_code_tour import custom_code_tour
from visionforge.gui.api.code_tour_routes import CodeSection
from visionforge.gui.api.http_responses import NOT_FOUND_TEXT, TAG_CUSTOM_TASKS
from visionforge.tasks.registry import get_task

router = APIRouter()


@router.get(
    "/custom/{key}/code-tour",
    responses={404: {"description": NOT_FOUND_TEXT}},
    tags=[TAG_CUSTOM_TASKS],
)
def custom_task_code_tour(key: str) -> list[CodeSection]:
    """The functions of a custom task, as they are in the researcher's file.

    A hook whose source cannot be read comes as a section with a note that says
    so, not as an error.
    """
    try:
        info = get_task(key)
    except KeyError as exc:
        raise HTTPException(404, exc.args[0] if exc.args else str(exc)) from exc
    if info.spec_cls is None:
        raise HTTPException(404, str(key))
    return [
        CodeSection.model_validate(item) for item in custom_code_tour(info.spec_cls)
    ]
