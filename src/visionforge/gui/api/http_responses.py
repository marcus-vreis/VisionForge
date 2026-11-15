"""OpenAPI documentation shared by the routers (ADR-132).

Owns the OpenAPI text of the status codes more than one router declares, so a
router split out of ``routes.py`` documents them without importing it, and the
domain tags every operation is grouped under (#205).

The ``*_TEXT`` strings are for routes that spell their ``responses=`` out with
literal status keys (``{400: {"description": BAD_REQUEST_TEXT}}``): a static
analyser reads the keys of such a literal, which it cannot do through a
``{**BAD_REQUEST}`` spread of an imported dict (Sonar S8415).
"""

from __future__ import annotations

from typing import Any

BAD_REQUEST_TEXT = "Invalid request"
FORBIDDEN_TEXT = "Access denied"
NOT_FOUND_TEXT = "Resource not found"
CONFLICT_TEXT = "Operation conflicts with current state"
UNPROCESSABLE_TEXT = "Validation Error"
SERVER_ERROR_TEXT = "Operation failed"

# A route that raises 422 itself declares it, and a description-only entry
# would drop FastAPI's validation-error schema from the document.
VALIDATION_ERROR_CONTENT: dict[str, Any] = {
    "application/json": {"schema": {"$ref": "#/components/schemas/HTTPValidationError"}}
}

BAD_REQUEST: dict[int | str, dict[str, Any]] = {400: {"description": BAD_REQUEST_TEXT}}
FORBIDDEN: dict[int | str, dict[str, Any]] = {403: {"description": FORBIDDEN_TEXT}}
NOT_FOUND: dict[int | str, dict[str, Any]] = {404: {"description": NOT_FOUND_TEXT}}
CONFLICT: dict[int | str, dict[str, Any]] = {409: {"description": CONFLICT_TEXT}}
SERVER_ERROR: dict[int | str, dict[str, Any]] = {
    500: {"description": SERVER_ERROR_TEXT}
}

TAG_SYSTEM = "system"
TAG_PROFILES = "profiles"
TAG_DATASETS = "datasets"
TAG_EXPLORER = "explorer"
TAG_EXPERIMENTS = "experiments"
TAG_DETECTION = "detection"
TAG_REGRESSION = "regression"
TAG_SEGMENTATION = "segmentation"
TAG_ANOMALY = "anomaly"
TAG_CUSTOM_TASKS = "custom-tasks"
TAG_RUNS = "runs"
TAG_COLAB = "colab"
TAG_INTEGRATIONS = "integrations"
TAG_SPA = "spa"

# ``openapi_tags`` of the app: the order is the order of the documentation.
OPENAPI_TAGS: list[dict[str, str]] = [
    {"name": TAG_SYSTEM, "description": "Health, hardware, updates and the code tour."},
    {"name": TAG_PROFILES, "description": "Researcher profiles."},
    {"name": TAG_DATASETS, "description": "Dataset pickers, statistics and downloads."},
    {"name": TAG_EXPLORER, "description": "Dataset explorer and derived datasets."},
    {
        "name": TAG_EXPERIMENTS,
        "description": "Classification runs, the queue and their progress.",
    },
    {"name": TAG_DETECTION, "description": "Object detection runs."},
    {"name": TAG_REGRESSION, "description": "Regression runs."},
    {"name": TAG_SEGMENTATION, "description": "Segmentation runs."},
    {"name": TAG_ANOMALY, "description": "Anomaly detection runs."},
    {"name": TAG_CUSTOM_TASKS, "description": "Tasks added by the researcher."},
    {"name": TAG_RUNS, "description": "Finished runs: history, reports and exports."},
    {"name": TAG_COLAB, "description": "Runs on Google Colab."},
    {
        "name": TAG_INTEGRATIONS,
        "description": "Hugging Face Hub, GitHub, Zenodo, W&B and DVC.",
    },
    {"name": TAG_SPA, "description": "The web interface's static files."},
]
