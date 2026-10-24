"""Run history endpoints (list, detail, delete), split out of ``routes.py`` (ADR-132).

No ``from __future__ import annotations``: ``ProfileDep`` is built inside the
builder, and FastAPI resolves string annotations against module globals only.
"""

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from loguru import logger

from visionforge.core.dataset_fingerprint import dataset_identity
from visionforge.gui.api.folder_opener import can_reveal
from visionforge.gui.api.http_responses import (
    BAD_REQUEST,
    CONFLICT,
    NOT_FOUND,
    SERVER_ERROR,
)
from visionforge.gui.api.profiles import PROFILE_HEADER, Profile
from visionforge.gui.api.run_history import (
    RUN_BAD_JSON_KEY,
    RUN_FILENAME,
    RUN_NOT_FOUND_KEY,
    load_runs,
    metric_directions,
    resume_status,
    run_span,
    run_task,
)
from visionforge.gui.api.schemas import DatasetInfo, RunDetail, RunSummary
from visionforge.utils.messages import tr


def build_runs_router(
    find_run_dir: Callable[[str, Profile], Path | None],
    resolve_profile: Callable[[str | None], Profile],
    current_run: Callable[[], dict[str, Any] | None],
    client_host: Callable[[Request], str | None],
) -> APIRouter:
    """Build the history endpoints; routes.py owns the lookups, read per request.

    ``current_run`` returns the state of the job executing now, so a delete can
    refuse that run without this module holding a copy of it.
    """
    router = APIRouter()

    def profile_dep(
        x_vf_profile: Annotated[str | None, Header(alias=PROFILE_HEADER)] = None,
    ) -> Profile:
        return resolve_profile(x_vf_profile)

    ProfileDep = Annotated[Profile, Depends(profile_dep)]

    @router.get("/runs", responses={**BAD_REQUEST, **NOT_FOUND})
    async def list_runs(profile: ProfileDep) -> list[RunSummary]:
        """Return the profile's historical runs sorted by started_at descending."""
        return load_runs(profile.models_dir.resolve())

    @router.get(
        "/runs/{run_id}", responses={**BAD_REQUEST, **NOT_FOUND, **SERVER_ERROR}
    )
    async def get_run_detail(
        run_id: str, request: Request, profile: ProfileDep
    ) -> RunDetail:
        """Return the full run.json for a specific run_id."""
        run_dir = find_run_dir(run_id, profile)
        if run_dir is None:
            raise HTTPException(404, tr(RUN_NOT_FOUND_KEY, run_id=run_id))
        run_json_path = run_dir / RUN_FILENAME
        try:
            data: dict[str, Any] = json.loads(run_json_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, tr(RUN_BAD_JSON_KEY, error=exc)) from exc

        status = data.get("status", "completed")
        started, finished = run_span(run_dir.name, data["timestamp"], status)

        dataset_name, dataset_root = dataset_identity(data)
        resumable, configured_epochs = resume_status(run_dir, data)
        fingerprint = data.get("dataset_fingerprint") or {}
        dataset = (
            DatasetInfo(
                name=dataset_name,
                root=dataset_root or "",
                n_files=fingerprint.get("n_files"),
                total_bytes=fingerprint.get("total_bytes"),
                method=fingerprint.get("method"),
                digest=fingerprint.get("digest") or None,
                note=fingerprint.get("note"),
            )
            if dataset_name
            else None
        )

        task = run_task(data)
        metrics: dict[str, Any] = data.get("metrics", {})
        history: list[dict[str, Any]] = data.get("history", [])

        return RunDetail(
            run_id=run_dir.name,
            experiment_name=data.get("experiment", run_dir.parent.name),
            status=status,
            started_at=started,
            finished_at=finished,
            device_used=data.get("device_used"),
            environment=data.get("environment", {}),
            run_dir=data.get("run_dir", str(run_dir.resolve())),
            config=data.get("config", {}),
            metrics=metrics,
            metric_cis=data.get("metric_cis", {}),
            history=history,
            artifacts=data.get("artifacts", {}),
            tests=data.get("tests", []),
            dataset=dataset,
            resumable=resumable,
            configured_epochs=configured_epochs,
            stopped=bool(data.get("stopped")),
            group_id=data.get("group_id"),
            group=data.get("group") if isinstance(data.get("group"), dict) else None,
            task=task,
            can_reveal=can_reveal(client_host(request)),
            metric_directions=metric_directions(task, metrics, history),
        )

    @router.delete("/runs/{run_id}", responses={**BAD_REQUEST, **NOT_FOUND, **CONFLICT})
    async def delete_run(run_id: str, profile: ProfileDep) -> dict[str, str]:
        """Permanently remove a run directory and all its artifacts.

        Safety:
        - Must resolve to a directory under the profile's models folder (no path
          traversal), which is ``_MODELS_DIR`` for the default profile (ADR-114).
        - Refuses to delete the run that is currently executing.
        """
        run_dir = find_run_dir(run_id, profile)
        if run_dir is None:
            raise HTTPException(404, tr(RUN_NOT_FOUND_KEY, run_id=run_id))

        run = current_run()
        # Block deleting the active run — its files are held open by the trainer.
        if (
            run is not None
            and run.get("status") == "running"
            and run.get("run_id") == run_id
        ):
            raise HTTPException(409, tr("run.cannot_delete_active"))

        resolved = run_dir.resolve()
        models_dir = profile.models_dir.resolve()
        try:
            resolved.relative_to(models_dir)
        except ValueError as exc:
            raise HTTPException(
                400, tr("run.delete_outside", models_dir=models_dir, path=resolved)
            ) from exc

        if not resolved.is_dir():
            raise HTTPException(404, tr("run.dir_missing", path=resolved))

        shutil.rmtree(resolved)
        logger.info("GUI: deleted run {} at {}", run_id, resolved)
        return {"run_id": run_id, "status": "deleted"}

    return router
