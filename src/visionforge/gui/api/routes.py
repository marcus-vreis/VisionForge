"""API endpoints for VisionForge GUI."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger

from visionforge.blocks.classification import ClassificationBlock
from visionforge.gui.api.schemas import (
    DatasetDetectRequest,
    DatasetDetectResponse,
    RunResponse,
    RunResult,
    RunStatus,
    RunSummary,
)
from visionforge.utils.config import ExperimentConfig

# Common split folder names per role. Lowercased, accent-stripped at match time.
_TRAIN_ALIASES = {"train", "training", "treino", "trains", "tr"}
_VAL_ALIASES = {
    "val",
    "valid",
    "validation",
    "validacao",
    "validação",
    "eval",
    "dev",
    "vl",
}
_TEST_ALIASES = {"test", "testing", "teste", "ts", "holdout", "hold_out"}

router = APIRouter(prefix="/api")

# Single-experiment state (MVP: one run at a time).
_current_run: dict[str, Any] | None = None

# Per-run SSE queue; None when no run is active.
# The Trainer worker thread puts dicts onto this queue via asyncio.run_coroutine_threadsafe;
# the SSE generator pulls from it and serialises each item as an SSE data line.
# A None sentinel signals end-of-stream so the generator can close cleanly.
_event_queue: asyncio.Queue[dict[str, Any] | None] | None = None

# Default location where Trainer writes run directories.
_MODELS_DIR = Path("outputs/models")

# Strong references to background tasks to prevent premature GC (asyncio doc).
_background_tasks: set[asyncio.Task[None]] = set()


@router.get("/schema")
async def get_schema() -> dict[str, Any]:
    """Return the JSON Schema for ExperimentConfig."""
    return ExperimentConfig.model_json_schema()


@router.get("/runs")
async def list_runs() -> list[RunSummary]:
    """Return all historical runs sorted by started_at descending."""
    return _load_runs(_MODELS_DIR.resolve())


@router.post("/dataset/detect")
async def detect_dataset(req: DatasetDetectRequest) -> DatasetDetectResponse:
    """Scan a base_dir for standard train/val/test split subdirectories.

    Returns a structured response describing what was found so the frontend
    can either auto-fill the splits or ask the user to map them manually.
    """
    return _detect_dataset_layout(req.base_dir)


@router.post("/experiment/run")
async def run_experiment(config: ExperimentConfig) -> RunResponse:
    """Start a training experiment in the background."""
    global _current_run, _event_queue

    if _current_run and _current_run["status"] == "running":
        raise HTTPException(409, "An experiment is already running.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{config.name}_{timestamp}"

    # Fresh queue for this run; SSE clients poll this queue.
    _event_queue = asyncio.Queue()

    _current_run = {
        "run_id": run_id,
        "status": "running",
        "error": None,
        "report": None,
        "run_dir": None,
    }

    task = asyncio.create_task(_execute_experiment(config, run_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return RunResponse(run_id=run_id)


@router.get("/experiment/events")
async def experiment_events() -> StreamingResponse:
    """Stream live training progress as Server-Sent Events.

    Yields one SSE data line per training event (start, epoch_end × N, end).
    The stream closes when the run completes, fails, or when no run is active.
    """
    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _sse_generator():  # type: ignore[return]
    """Pull events off _event_queue and yield SSE-formatted lines."""
    # Snapshot the queue at request time; if no run is active, close immediately.
    queue = _event_queue
    if queue is None:
        yield "data: {}\n\n"
        return

    while True:
        item = await queue.get()
        if item is None:
            # Sentinel — run finished or failed.
            break
        yield f"data: {json.dumps(item)}\n\n"


@router.get("/experiment/status")
async def get_status() -> RunStatus:
    """Poll the current experiment status."""
    if _current_run is None:
        return RunStatus(status="idle")

    return RunStatus(
        status=_current_run["status"],
        run_id=_current_run["run_id"],
        error=_current_run.get("error"),
    )


@router.get("/experiment/result/{run_id}")
async def get_result(run_id: str) -> RunResult:
    """Fetch the result of a completed experiment."""
    if _current_run is None or _current_run["run_id"] != run_id:
        raise HTTPException(404, "Run not found.")

    if _current_run["status"] == "running":
        raise HTTPException(409, "Experiment is still running.")

    if _current_run["status"] == "failed":
        raise HTTPException(
            500,
            f"Experiment failed: {_current_run.get('error', 'unknown error')}",
        )

    run_dir = _current_run.get("run_dir")
    run_json_path = Path(run_dir) / "run.json" if run_dir else None

    if run_json_path and run_json_path.exists():
        data = json.loads(run_json_path.read_text(encoding="utf-8"))
        return RunResult(
            run_id=run_id,
            metrics=data.get("metrics", {}),
            report=_current_run.get("report", {}),
            artifacts=data.get("artifacts", {}),
        )

    return RunResult(
        run_id=run_id,
        metrics={},
        report=_current_run.get("report", {}),
        artifacts={},
    )


@router.get("/artifacts/{file_path:path}")
async def serve_artifact(file_path: str) -> FileResponse:
    """Serve output files (plots, models) with path traversal protection."""
    resolved = Path(
        file_path
    ).resolve()  # NOSONAR pythonsecurity:S6549 - re-checked with is_relative_to below before FileResponse
    outputs_dir = Path("outputs").resolve()

    # is_relative_to is immune to the str.startswith bypass on case-insensitive FSes.
    if not resolved.is_relative_to(outputs_dir):
        raise HTTPException(403, "Access denied.")

    if not resolved.is_file():
        raise HTTPException(404, "File not found.")

    # Path is validated above to be inside outputs/ and to be a regular file.
    return FileResponse(resolved)  # NOSONAR python:S2083


def _load_runs(models_dir: Path) -> list[RunSummary]:
    """Scan models_dir recursively for run.json files and parse each into RunSummary.

    Directories without a parseable run.json are silently skipped.
    """
    if not models_dir.exists():
        return []

    summaries: list[RunSummary] = []
    for run_json_path in models_dir.rglob("run.json"):
        try:
            data: dict[str, Any] = json.loads(run_json_path.read_text(encoding="utf-8"))
            summary = _parse_run_summary(run_json_path.parent, data)
            summaries.append(summary)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping unparsable run.json at {}: {}", run_json_path, exc)

    summaries.sort(key=lambda s: s.started_at, reverse=True)
    return summaries


def _parse_run_summary(run_dir: Path, data: dict[str, Any]) -> RunSummary:
    """Build a RunSummary from a parsed run.json dict.

    Raises KeyError or ValueError if required fields are missing.
    """
    status: str = data["status"]
    started_at = datetime.fromisoformat(data["timestamp"])
    finished_at: datetime | None = started_at if status == "completed" else None

    config: dict[str, Any] = data["config"]
    metrics: dict[str, Any] = data.get("metrics", {})

    _metric_map = {
        "accuracy": "test_accuracy",
        "f1": "test_f1",
        "val_loss": "best_val_loss",
    }
    final_metrics: dict[str, float] = {
        key: float(metrics[src]) for key, src in _metric_map.items() if src in metrics
    }

    return RunSummary(
        run_id=run_dir.name,
        experiment_name=data["experiment"],
        model_arch=config["model"]["name"],
        task=config["task"],
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        epochs_completed=int(metrics["total_epochs"]),
        final_metrics=final_metrics,
    )


async def _execute_experiment(config: ExperimentConfig, run_id: str) -> None:
    """Run the experiment in a background thread."""
    global _current_run, _event_queue

    loop = asyncio.get_running_loop()
    queue = _event_queue  # capture at task-start to survive queue replacement

    def _put_event(event: dict[str, Any]) -> None:
        # Called from the Trainer worker thread; must schedule the coroutine on
        # the event loop thread-safely.
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    try:
        block = ClassificationBlock()
        block.setup(config)
        block._progress_callback = _put_event

        logger.info("GUI: Starting experiment {}", run_id)
        await asyncio.to_thread(block.run)

        report = block.report()

        # Extract run_dir from the train result stored in the block.
        run_dir: str | None = None
        if block._train_result and block._train_result.model_path:
            run_dir = str(block._train_result.model_path.parent)

        _current_run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": run_dir,
        }
        logger.success("GUI: Experiment {} completed.", run_id)

    except Exception as e:
        logger.exception("GUI: Experiment {} failed", run_id)
        cls = type(e).__name__
        msg = str(e) or "(sem mensagem)"
        _current_run = {
            "run_id": run_id,
            "status": "failed",
            "error": f"{cls}: {msg}",
            "report": None,
            "run_dir": None,
        }

    finally:
        # Always put the sentinel so the SSE generator can close cleanly.
        if queue is not None:
            await queue.put(None)


def _normalize_split_name(name: str) -> str:
    """Lowercase + strip simple separators so 'Train_Set' matches 'train'."""
    import unicodedata

    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    stripped = stripped.lower()
    for sep in ("-", "_", " ", "."):
        stripped = stripped.replace(sep, "")
    return stripped


def _match_alias(name: str, aliases: set[str]) -> bool:
    """Return True if ``name`` (or its prefix) maps to one of the alias variants."""
    normalized = _normalize_split_name(name)
    return normalized in aliases


def _detect_dataset_layout(base_dir_str: str) -> DatasetDetectResponse:
    """Inspect a directory and try to identify standard train/val/test splits."""
    if not base_dir_str or not base_dir_str.strip():
        return DatasetDetectResponse(
            base_dir=base_dir_str,
            detected=False,
            message="Informe o caminho do diretório base do dataset.",
        )

    base = Path(base_dir_str).expanduser()
    if not base.exists():
        return DatasetDetectResponse(
            base_dir=str(base),
            detected=False,
            message=f"O diretório '{base}' não foi encontrado no disco.",
        )
    if not base.is_dir():
        return DatasetDetectResponse(
            base_dir=str(base),
            detected=False,
            message=f"O caminho '{base}' existe mas não é uma pasta.",
        )

    try:
        children = [p for p in base.iterdir() if p.is_dir()]
    except PermissionError:
        return DatasetDetectResponse(
            base_dir=str(base),
            detected=False,
            message=f"Sem permissão de leitura em '{base}'.",
        )

    if not children:
        return DatasetDetectResponse(
            base_dir=str(base),
            detected=False,
            message=(
                "Nenhuma subpasta encontrada em "
                f"'{base}'. Esperado pastas separando treino, validação e teste."
            ),
        )

    train_dir: str | None = None
    val_dir: str | None = None
    test_dir: str | None = None

    for child in children:
        if train_dir is None and _match_alias(child.name, _TRAIN_ALIASES):
            train_dir = child.name
        elif val_dir is None and _match_alias(child.name, _VAL_ALIASES):
            val_dir = child.name
        elif test_dir is None and _match_alias(child.name, _TEST_ALIASES):
            test_dir = child.name

    candidates = sorted(c.name for c in children)
    found = [v for v in (train_dir, val_dir, test_dir) if v]

    if len(found) >= 2:
        missing = []
        if not train_dir:
            missing.append("treino")
        if not val_dir:
            missing.append("validação")
        if not test_dir:
            missing.append("teste")
        if missing:
            message = (
                f"Detectado parcialmente. Faltando: {', '.join(missing)}. "
                "Selecione manualmente as pastas restantes."
            )
            return DatasetDetectResponse(
                base_dir=str(base),
                detected=False,
                train_dir=train_dir,
                val_dir=val_dir,
                test_dir=test_dir,
                candidates=candidates,
                message=message,
            )
        return DatasetDetectResponse(
            base_dir=str(base),
            detected=True,
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            candidates=candidates,
            message=(
                f"Splits detectados: treino='{train_dir}', "
                f"validação='{val_dir}', teste='{test_dir}'."
            ),
        )

    return DatasetDetectResponse(
        base_dir=str(base),
        detected=False,
        candidates=candidates,
        message=(
            "Não foi possível identificar automaticamente os splits. "
            f"Subpastas encontradas: {', '.join(candidates)}. "
            "Selecione manualmente qual é treino / validação / teste."
        ),
    )


__all__ = ["router", "_load_runs", "_detect_dataset_layout", "_sse_generator"]
