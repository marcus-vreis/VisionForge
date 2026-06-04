"""API endpoints for VisionForge GUI."""

from __future__ import annotations

import asyncio
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from loguru import logger

from visionforge.blocks.anomaly import AnomalyBlock
from visionforge.blocks.batch_prediction import BatchPredictionBlock
from visionforge.blocks.classification import ClassificationBlock
from visionforge.blocks.cross_validation import CrossValidationBlock
from visionforge.blocks.detection import DetectionBlock
from visionforge.blocks.export_onnx import ExportONNXBlock
from visionforge.blocks.grid_search import GridSearchBlock
from visionforge.blocks.model_comparison import ModelComparisonBlock
from visionforge.blocks.random_search import RandomSearchBlock
from visionforge.blocks.regression import RegressionBlock
from visionforge.blocks.segmentation import SegmentationBlock
from visionforge.blocks.transfer_learning import TransferLearningBlock
from visionforge.core.data import DataModule
from visionforge.core.detection_data import resolve_yolo_split
from visionforge.core.evaluator import Evaluator
from visionforge.core.plotter import MetricsPlotter
from visionforge.gui.api.detection_export import export_detection_run
from visionforge.gui.api.detection_testing import evaluate_detection_run
from visionforge.gui.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    CheckpointPickResponse,
    DatasetDetectRequest,
    DatasetDetectResponse,
    DatasetPickResponse,
    DatasetSamplesRequest,
    DatasetSamplesResponse,
    DatasetStatsRequest,
    DatasetStatsResponse,
    DetectionDatasetStatsRequest,
    DetectionDatasetStatsResponse,
    DetectionSplitStats,
    DeviceInfoResponse,
    ExportOnnxRequest,
    ExportOnnxResponse,
    GPUInfo,
    PreprocessPreviewRequest,
    PreprocessPreviewResponse,
    PreprocessPreviewStep,
    RunDetail,
    RunResponse,
    RunResult,
    RunStatus,
    RunSummary,
    RunTestRequest,
    RunTestResponse,
    SplitStats,
    SystemInfo,
)
from visionforge.models.factory import ModelFactory
from visionforge.utils.anomaly_config import AnomalyConfig
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.cuda import check_cuda
from visionforge.utils.detection_config import DetectionConfig
from visionforge.utils.regression_config import RegressionConfig
from visionforge.utils.segmentation_config import SegmentationConfig

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

# Image extensions accepted both for dataset stats / sample probes and for the
# /dataset/file thumbnail endpoint.
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Dataset roots the user has explicitly probed this session — populated by
# /dataset/pick, /dataset/stats, /dataset/samples, /dataset/preview_preprocess
# and /runs/{id}/test. The /dataset/file endpoint refuses paths that aren't
# inside one of these roots, so a crafted request cannot read arbitrary files
# even though the server runs locally with full FS access.
_ALLOWED_DATASET_ROOTS: set[Path] = set()


def _remember_dataset_root(base_dir: str | Path) -> None:
    """Add a resolved dataset root to the allow-list for /dataset/file."""
    try:
        resolved = Path(base_dir).resolve()
        if resolved.is_dir():
            _ALLOWED_DATASET_ROOTS.add(resolved)
    except OSError:
        pass


def _is_inside_allowed_root(path: Path) -> bool:
    """True iff ``path`` resolves inside one of the remembered dataset roots."""
    for root in _ALLOWED_DATASET_ROOTS:
        try:
            if path.is_relative_to(root):
                return True
        except ValueError:
            continue
    return False


# Strong references to background tasks to prevent premature GC (asyncio doc).
_background_tasks: set[asyncio.Task[None]] = set()


@router.get("/schema")
async def get_schema() -> dict[str, Any]:
    """Return the JSON Schema for ExperimentConfig."""
    return ExperimentConfig.model_json_schema()


@router.get("/system/info")
async def get_system_info() -> SystemInfo:
    """CPU probe used to derive sensible defaults (workers, threads).

    Suggested workers follow PyTorch's common heuristic min(cpu_count, 8):
    too many workers fight the GIL on small datasets and burn RAM on the
    DataLoader pre-fetch queues without speeding things up.
    """
    import os

    cpu = os.cpu_count() or 1
    return SystemInfo(
        cpu_count=cpu,
        suggested_workers=min(cpu, 8),
        platform=platform.system(),
    )


@router.get("/device/info")
async def get_device_info() -> DeviceInfoResponse:
    """Probe the runtime and return the real list of selectable devices."""
    info = check_cuda()
    gpus = [
        GPUInfo(
            index=d.index,
            name=d.name,
            total_memory_mb=d.total_memory_mb,
            compute_capability=d.compute_capability,
        )
        for d in info.devices
    ]
    return DeviceInfoResponse(
        cuda_available=info.available,
        cuda_version=info.cuda_version,
        cpu_name=platform.processor() or platform.machine() or "CPU",
        gpus=gpus,
    )


@router.get("/runs")
async def list_runs() -> list[RunSummary]:
    """Return all historical runs sorted by started_at descending."""
    return _load_runs(_MODELS_DIR.resolve())


@router.get("/runs/{run_id}")
async def get_run_detail(run_id: str) -> RunDetail:
    """Return the full run.json for a specific run_id."""
    run_dir = _find_run_dir(run_id)
    if run_dir is None:
        raise HTTPException(404, f"Run '{run_id}' not found.")
    run_json_path = run_dir / "run.json"
    try:
        data: dict[str, Any] = json.loads(run_json_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Failed to parse run.json: {exc}") from exc

    started = datetime.fromisoformat(data["timestamp"])
    status = data.get("status", "completed")
    finished = started if status == "completed" else None

    return RunDetail(
        run_id=run_dir.name,
        experiment_name=data.get("experiment", run_dir.parent.name),
        status=status,
        started_at=started,
        finished_at=finished,
        device_used=data.get("device_used"),
        run_dir=data.get("run_dir", str(run_dir.resolve())),
        config=data.get("config", {}),
        metrics=data.get("metrics", {}),
        history=data.get("history", []),
        artifacts=data.get("artifacts", {}),
        tests=data.get("tests", []),
    )


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str) -> dict[str, str]:
    """Permanently remove a run directory and all its artifacts.

    Safety:
    - Must resolve to a directory under _MODELS_DIR (no path traversal).
    - Refuses to delete the run that is currently executing.
    """
    import shutil

    run_dir = _find_run_dir(run_id)
    if run_dir is None:
        raise HTTPException(404, f"Run '{run_id}' not found.")

    # Block deleting the active run — its files are held open by the trainer.
    if (
        _current_run is not None
        and _current_run.get("status") == "running"
        and _current_run.get("run_id") == run_id
    ):
        raise HTTPException(409, "Cannot delete a run that is currently executing.")

    resolved = run_dir.resolve()
    models_dir = _MODELS_DIR.resolve()
    try:
        resolved.relative_to(models_dir)
    except ValueError as exc:
        raise HTTPException(
            400, f"Refused to delete path outside {models_dir}: {resolved}"
        ) from exc

    if not resolved.is_dir():
        raise HTTPException(404, f"Run directory '{resolved}' does not exist.")

    shutil.rmtree(resolved)
    logger.info("GUI: deleted run {} at {}", run_id, resolved)
    return {"run_id": run_id, "status": "deleted"}


@router.get("/runs/{run_id}/export_md")
async def export_run_markdown(run_id: str) -> Response:
    """Render a model card (markdown) for the run, ready for paper inclusion."""
    run_dir = _find_run_dir(run_id)
    if run_dir is None:
        raise HTTPException(404, f"Run '{run_id}' not found.")
    run_json_path = run_dir / "run.json"
    try:
        data: dict[str, Any] = json.loads(run_json_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Failed to parse run.json: {exc}") from exc

    md = _render_run_markdown(run_dir, data)
    filename = f"{data.get('experiment', run_dir.name)}_{run_dir.name}.md"
    return Response(
        content=md,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/runs/{run_id}/test")
async def test_run_on_dataset(run_id: str, req: RunTestRequest) -> RunTestResponse:
    """Run the saved checkpoint against a new dataset and record the result.

    The test result is appended to run.json under the 'tests' array so each
    saved model accumulates its own per-dataset history.
    """
    run_dir = _find_run_dir(run_id)
    if run_dir is None:
        raise HTTPException(404, f"Run '{run_id}' not found.")

    # Run the actual evaluation in a worker thread (PyTorch + disk I/O).
    try:
        result = await asyncio.to_thread(_execute_run_test, run_dir, req)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Test on run {} failed", run_id)
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

    return result


@router.post("/runs/{run_id}/batch_predict")
async def batch_predict_run(
    run_id: str, req: BatchPredictRequest
) -> BatchPredictResponse:
    """Run inference of this run's checkpoint over a folder of images.

    Outputs a CSV with one row per image. ``output_csv`` defaults to a
    timestamped file in ``<run_dir>/predictions/`` so consecutive batch runs
    don't overwrite each other.
    """
    run_dir = _find_run_dir(run_id)
    if run_dir is None:
        raise HTTPException(404, f"Run '{run_id}' not found.")

    try:
        result = await asyncio.to_thread(_execute_batch_predict, run_dir, req)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Batch prediction on run {} failed", run_id)
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

    return result


@router.post("/runs/{run_id}/export_onnx")
async def export_run_to_onnx(run_id: str, req: ExportOnnxRequest) -> ExportOnnxResponse:
    """Export this run's saved checkpoint to ONNX, optionally validating it.

    Reuses the run's stored ``config.data.transforms.image_size`` and
    ``config.model`` to build the dummy input and the architecture for export.
    """
    run_dir = _find_run_dir(run_id)
    if run_dir is None:
        raise HTTPException(404, f"Run '{run_id}' not found.")

    try:
        result = await asyncio.to_thread(_execute_onnx_export, run_dir, req)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("ONNX export on run {} failed", run_id)
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

    return result


@router.post("/dataset/detect")
async def detect_dataset(req: DatasetDetectRequest) -> DatasetDetectResponse:
    """Scan a base_dir for standard train/val/test split subdirectories."""
    return _detect_dataset_layout(req.base_dir)


@router.post("/dataset/preview_preprocess")
async def preview_preprocess(
    req: PreprocessPreviewRequest,
) -> PreprocessPreviewResponse:
    """Generate a strip of PNG previews showing the preprocessing pipeline.

    Picks one image from the requested split (first one alphabetically inside
    ``class_name`` or the first class found) and runs every step. Each step's
    output is saved as a PNG under ``outputs/preview_cache/<run-id>/`` and
    served via the existing /api/artifacts/{path} endpoint.
    """
    return await asyncio.to_thread(_render_preprocess_preview, req)


@router.post("/dataset/samples")
async def dataset_samples(req: DatasetSamplesRequest) -> DatasetSamplesResponse:
    """Return the first N image paths per class in a chosen split.

    The frontend renders thumbnails so the researcher can verify labels are
    correct (i.e. ``class_a`` folder really does contain class A images).
    """
    return await asyncio.to_thread(_collect_dataset_samples, req)


@router.post("/dataset/stats")
async def dataset_stats(req: DatasetStatsRequest) -> DatasetStatsResponse:
    """Count images per class in train/val/test and flag imbalance.

    Walks the standard ImageFolder layout and counts files with image
    extensions. Splits whose directory is missing are reported with
    ``missing=true`` instead of raising — partial datasets are common during
    a first-time setup.
    """
    return await asyncio.to_thread(_collect_dataset_stats, req)


@router.post("/checkpoint/pick")
async def pick_checkpoint_file() -> CheckpointPickResponse:
    """Open a native file picker (filtered to .pth/.pt) and return the path.

    Used by the ModelConfig.weights_path field — researchers loading a custom
    pretrained checkpoint into the architecture instead of ImageNet weights.
    """
    return await asyncio.to_thread(_open_native_checkpoint_dialog)


@router.post("/dataset/pick")
async def pick_dataset_folder() -> DatasetPickResponse:
    """Open a native folder picker on the server and return the absolute path.

    Because VisionForge runs locally (browser + Python on the same machine),
    we can open a tkinter dialog server-side so the user gets a *real* absolute
    path — something the browser's File System Access API never provides.
    """
    return await asyncio.to_thread(_open_native_folder_dialog)


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


@router.get("/detection/schema")
async def get_detection_schema() -> dict[str, Any]:
    """Return the JSON Schema for DetectionConfig (drives the detection form)."""
    return DetectionConfig.model_json_schema()


@router.post("/detection/dataset/pick_yaml")
async def pick_detection_yaml() -> DatasetPickResponse:
    """Open a native file picker (filtered to .yaml/.yml) for a data.yaml.

    Detection datasets in the standard Ultralytics/Roboflow format ship a
    ``data.yaml`` describing splits and class names; this lets the user point
    at it directly instead of synthesizing one from a folder layout.
    """
    return await asyncio.to_thread(_open_native_yaml_dialog)


@router.post("/detection/dataset/stats")
async def detection_dataset_stats(
    req: DetectionDatasetStatsRequest,
) -> DetectionDatasetStatsResponse:
    """Per-split instance distribution for a YOLO-layout detection dataset.

    The detection analogue of /dataset/stats: instead of one folder per class,
    it counts annotated boxes per class across the ``.txt`` label files so the
    researcher can spot class imbalance and unlabeled images before training.
    """
    return await asyncio.to_thread(_collect_detection_dataset_stats, req)


@router.post("/detection/run")
async def run_detection(config: DetectionConfig) -> RunResponse:
    """Start an object-detection training run in the background.

    Detection shares the single-run state and the /experiment/{status,events,
    result} endpoints (one run at a time, one GPU) but dispatches its own
    standalone DetectionBlock (ADR-033) rather than the ExperimentConfig path.
    """
    global _current_run, _event_queue

    if _current_run and _current_run["status"] == "running":
        raise HTTPException(409, "An experiment is already running.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{config.name}_{timestamp}"

    _event_queue = asyncio.Queue()
    _current_run = {
        "run_id": run_id,
        "status": "running",
        "error": None,
        "report": None,
        "run_dir": None,
    }

    task = asyncio.create_task(_execute_detection(config, run_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return RunResponse(run_id=run_id)


@router.get("/regression/schema")
async def get_regression_schema() -> dict[str, Any]:
    """Return the JSON Schema for RegressionConfig (drives the regression form)."""
    return RegressionConfig.model_json_schema()


@router.post("/regression/run")
async def run_regression(config: RegressionConfig) -> RunResponse:
    """Start an image-regression training run in the background.

    Regression shares the single-run state and the /experiment/{status,events,
    result} endpoints (one run at a time, one GPU) but dispatches its own
    standalone RegressionBlock (ADR-036) rather than the ExperimentConfig path.
    """
    global _current_run, _event_queue

    if _current_run and _current_run["status"] == "running":
        raise HTTPException(409, "An experiment is already running.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{config.name}_{timestamp}"

    _event_queue = asyncio.Queue()
    _current_run = {
        "run_id": run_id,
        "status": "running",
        "error": None,
        "report": None,
        "run_dir": None,
    }

    task = asyncio.create_task(_execute_regression(config, run_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return RunResponse(run_id=run_id)


@router.get("/segmentation/schema")
async def get_segmentation_schema() -> dict[str, Any]:
    """Return the JSON Schema for SegmentationConfig (drives the segmentation form)."""
    return SegmentationConfig.model_json_schema()


@router.post("/segmentation/run")
async def run_segmentation(config: SegmentationConfig) -> RunResponse:
    """Start a semantic-segmentation training run in the background.

    Segmentation shares the single-run state and the /experiment/{status,events,
    result} endpoints (one run at a time, one GPU) but dispatches its own
    standalone SegmentationBlock (ADR-037) rather than the ExperimentConfig path.
    """
    global _current_run, _event_queue

    if _current_run and _current_run["status"] == "running":
        raise HTTPException(409, "An experiment is already running.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{config.name}_{timestamp}"

    _event_queue = asyncio.Queue()
    _current_run = {
        "run_id": run_id,
        "status": "running",
        "error": None,
        "report": None,
        "run_dir": None,
    }

    task = asyncio.create_task(_execute_segmentation(config, run_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return RunResponse(run_id=run_id)


@router.get("/anomaly/schema")
async def get_anomaly_schema() -> dict[str, Any]:
    """Return the JSON Schema for AnomalyConfig (drives the anomaly form)."""
    return AnomalyConfig.model_json_schema()


@router.post("/anomaly/run")
async def run_anomaly(config: AnomalyConfig) -> RunResponse:
    """Start an anomaly-detection training run in the background.

    Anomaly shares the single-run state and the /experiment/{status,events,
    result} endpoints (one run at a time, one GPU) but dispatches its own
    standalone AnomalyBlock (ADR-038) rather than the ExperimentConfig path.
    """
    global _current_run, _event_queue

    if _current_run and _current_run["status"] == "running":
        raise HTTPException(409, "An experiment is already running.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{config.name}_{timestamp}"

    _event_queue = asyncio.Queue()
    _current_run = {
        "run_id": run_id,
        "status": "running",
        "error": None,
        "report": None,
        "run_dir": None,
    }

    task = asyncio.create_task(_execute_anomaly(config, run_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return RunResponse(run_id=run_id)


@router.get("/experiment/events")
async def experiment_events() -> StreamingResponse:
    """Stream live training progress as Server-Sent Events."""
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
    queue = _event_queue
    if queue is None:
        yield "data: {}\n\n"
        return

    while True:
        item = await queue.get()
        if item is None:
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


@router.get("/dataset/file")
async def serve_dataset_file(path: str) -> FileResponse:
    """Serve a local image file for dataset preview thumbnails.

    Defense in depth — even though VisionForge runs locally, the path must:
    1. resolve to a real file
    2. carry a known image extension
    3. sit inside one of the dataset roots the user has already probed via
       /dataset/pick, /dataset/stats, /dataset/samples, etc.

    Without #3 a crafted query could read any file the server process can
    open. Restricting to the session's allow-list keeps the endpoint useful
    for the thumbnail UI without exposing arbitrary FS reads.
    """
    # Fail fast on obvious traversal before we even resolve. The allow-list
    # check below is the authoritative guard, but these two early checks make
    # the intent obvious to readers (and to taint-analysis tools).
    if ".." in path.replace("\\", "/").split("/"):
        raise HTTPException(400, "Path contains traversal segments.")
    candidate = Path(path)
    if not candidate.is_absolute():
        raise HTTPException(400, "Path must be absolute.")
    try:
        resolved = candidate.resolve(strict=True)  # NOSONAR pythonsecurity:S6549
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(404, "File not found.") from exc
    if resolved.suffix.lower() not in _IMAGE_EXTS:
        raise HTTPException(400, "Only image files are previewable.")
    if not _is_inside_allowed_root(resolved):
        raise HTTPException(
            403, "Path is not inside any dataset directory probed this session."
        )
    return FileResponse(resolved)  # NOSONAR python:S2083


@router.get("/artifacts/{file_path:path}")
async def serve_artifact(file_path: str) -> FileResponse:
    """Serve output files (plots, models) with path traversal protection."""
    resolved = Path(
        file_path
    ).resolve()  # NOSONAR pythonsecurity:S6549 - re-checked with is_relative_to below before FileResponse
    outputs_dir = Path("outputs").resolve()

    if not resolved.is_relative_to(outputs_dir):
        raise HTTPException(403, "Access denied.")

    if not resolved.is_file():
        raise HTTPException(404, "File not found.")

    return FileResponse(resolved)  # NOSONAR python:S2083


# ── private helpers ──────────────────────────────────────────────────────────


def _render_run_markdown(run_dir: Path, data: dict[str, Any]) -> str:
    """Format a run.json into a paper-ready model card."""
    lines: list[str] = []
    name = data.get("experiment", run_dir.name)
    config: dict[str, Any] = data.get("config", {})
    metrics: dict[str, Any] = data.get("metrics", {})
    history: list[dict[str, Any]] = data.get("history", [])

    lines.append(f"# {name}")
    lines.append("")
    lines.append(f"- **Run ID:** `{data.get('id', run_dir.name)}`")
    lines.append(f"- **Timestamp:** {data.get('timestamp', '?')}")
    lines.append(f"- **Status:** {data.get('status', '?')}")
    if data.get("device_used"):
        lines.append(f"- **Device used:** `{data['device_used']}`")
    lines.append(f"- **Run directory:** `{data.get('run_dir', run_dir)}`")
    artifacts = data.get("artifacts", {})
    if artifacts.get("model"):
        lines.append(f"- **Checkpoint:** `{artifacts['model']}`")
    lines.append("")

    lines.append("## Configuration")
    lines.append("")
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    lines.append(f"- **Architecture:** {model_cfg.get('name', '?')}")
    lines.append(f"- **num_classes:** {model_cfg.get('num_classes', '?')}")
    lines.append(f"- **Pretrained:** {model_cfg.get('pretrained', '?')}")
    lines.append(f"- **Task:** {config.get('task', '?')}")
    lines.append(f"- **Optimizer:** {training_cfg.get('optimizer', '?')}")
    lines.append(f"- **Learning rate:** {training_cfg.get('learning_rate', '?')}")
    lines.append(f"- **Batch size:** {training_cfg.get('batch_size', '?')}")
    lines.append(f"- **Epochs (max):** {training_cfg.get('epochs', '?')}")
    lines.append(f"- **Seed:** {training_cfg.get('seed', '?')}")
    if training_cfg.get("scheduler", {}).get("kind", "none") != "none":
        sched = training_cfg["scheduler"]
        lines.append(f"- **Scheduler:** {sched['kind']} ({sched})")
    if training_cfg.get("mixed_precision"):
        lines.append("- **Mixed precision:** enabled")
    lines.append("")

    data_cfg = config.get("data", {})
    pp_steps = data_cfg.get("preprocessing", {}).get("steps", [])
    if pp_steps:
        lines.append("## Preprocessing pipeline")
        lines.append("")
        lines.append(
            "Applied to every loaded image before augmentation, in this order:"
        )
        lines.append("")
        for i, step in enumerate(pp_steps, start=1):
            kind = step.get("kind", "?")
            params = ", ".join(f"{k}={v!r}" for k, v in step.items() if k != "kind")
            lines.append(f"{i}. **{kind}**" + (f" ({params})" if params else ""))
        lines.append("")

    tx = data_cfg.get("transforms", {})
    if tx:
        lines.append("## Augmentation / Normalization")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        for k, v in tx.items():
            lines.append(f"| {k} | `{v}` |")
        lines.append("")

    lines.append("## Final metrics")
    lines.append("")
    if metrics:
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
            else:
                lines.append(f"| {k} | {v} |")
    else:
        lines.append("_(none recorded)_")
    lines.append("")

    if history:
        lines.append(f"## Training history ({len(history)} epoch(s))")
        lines.append("")
        lines.append("| Epoch | train_loss | train_acc | val_loss | val_acc |")
        lines.append("|---|---|---|---|---|")
        for h in history:
            lines.append(
                f"| {h.get('epoch')} | {h.get('train_loss', '?'):.4f} | "
                f"{h.get('train_accuracy', '?'):.4f} | "
                f"{h.get('val_loss', '?'):.4f} | {h.get('val_accuracy', '?'):.4f} |"
            )
        lines.append("")

    graphics = artifacts.get("graphics", [])
    if graphics:
        lines.append("## Plots")
        lines.append("")
        for g in graphics:
            lines.append(f"- `{g}`")
        lines.append("")

    tests = data.get("tests", [])
    if tests:
        lines.append(f"## Test runs on additional datasets ({len(tests)})")
        lines.append("")
        for t in tests:
            lines.append(f"### {t.get('label', t.get('test_id'))}")
            lines.append(f"- **base_dir:** `{t.get('base_dir')}`")
            lines.append(f"- **timestamp:** {t.get('timestamp')}")
            m = t.get("metrics", {})
            if m:
                lines.append("| Metric | Value |")
                lines.append("|---|---|")
                for k, v in m.items():
                    val = f"{v:.4f}" if isinstance(v, float) else str(v)
                    lines.append(f"| {k} | {val} |")
            lines.append("")

    return "\n".join(lines)


def _find_run_dir(run_id: str) -> Path | None:
    """Locate a run directory by its timestamp folder name."""
    models_dir = _MODELS_DIR.resolve()
    if not models_dir.exists():
        return None
    # run_id == timestamp folder OR experiment_name+timestamp; check both forms.
    for run_json in models_dir.rglob("run.json"):
        d = run_json.parent
        if d.name == run_id:
            return d
        try:
            data = json.loads(run_json.read_text(encoding="utf-8"))
            if data.get("id") == run_id:
                return d
        except Exception:  # noqa: BLE001
            continue
    return None


_PREVIEW_CACHE_DIR = Path("outputs/preview_cache")


def _render_preprocess_preview(
    req: PreprocessPreviewRequest,
) -> PreprocessPreviewResponse:
    """Save original + per-step previews and return their artifact paths."""
    from PIL import Image

    from visionforge.core.preprocessing import (
        apply_pipeline,
        available_kinds,
    )

    _remember_dataset_root(req.base_dir)
    base = Path(req.base_dir)
    split_dir = base / req.split
    if not split_dir.is_dir():
        return PreprocessPreviewResponse(
            original="",
            steps=[],
            final="",
            source_image="",
            available_kinds=available_kinds(),
            message=f"Split '{req.split}' não encontrado.",
        )

    # Pick the first available class + first image inside it.
    class_dirs = [d for d in sorted(split_dir.iterdir()) if d.is_dir()]
    if not class_dirs:
        return PreprocessPreviewResponse(
            original="",
            steps=[],
            final="",
            source_image="",
            available_kinds=available_kinds(),
            message="Nenhuma classe encontrada no split.",
        )

    target_class = None
    if req.class_name is not None:
        target_class = next((d for d in class_dirs if d.name == req.class_name), None)
    if target_class is None:
        target_class = class_dirs[0]

    img_path = next(
        (
            p
            for p in sorted(target_class.iterdir())
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        ),
        None,
    )
    if img_path is None:
        return PreprocessPreviewResponse(
            original="",
            steps=[],
            final="",
            source_image="",
            available_kinds=available_kinds(),
            message=f"Sem imagens no diretório {target_class}.",
        )

    # Cache dir is deterministic per source image so repeated previews don't
    # accumulate hundreds of files. Each invocation overwrites prior renders.
    cache_dir = _PREVIEW_CACHE_DIR / target_class.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob("*.png"):
        try:
            old.unlink()
        except OSError:
            pass

    original_img = Image.open(img_path).convert("RGB")
    original_path = cache_dir / "00_original.png"
    original_img.save(original_path)

    final_img, intermediates = apply_pipeline(original_img, req.steps)

    step_records: list[PreprocessPreviewStep] = []
    for i, (step_cfg, img) in enumerate(zip(req.steps, intermediates, strict=False)):
        step_path = cache_dir / f"{i + 1:02d}_{step_cfg.get('kind', 'step')}.png"
        img.save(step_path)
        step_records.append(
            PreprocessPreviewStep(
                kind=str(step_cfg.get("kind", "")),
                artifact=str(step_path),
                params={k: v for k, v in step_cfg.items() if k != "kind"},
            )
        )

    if step_records:
        final_path = Path(step_records[-1].artifact)
    else:
        final_path = original_path
    if step_records:
        # Save a separate "final" copy so callers don't depend on intermediates.
        final_path = cache_dir / "99_final.png"
        final_img.save(final_path)

    return PreprocessPreviewResponse(
        original=str(original_path),
        steps=step_records,
        final=str(final_path),
        source_image=str(img_path.resolve()),
        available_kinds=available_kinds(),
    )


def _collect_dataset_samples(req: DatasetSamplesRequest) -> DatasetSamplesResponse:
    """Walk the chosen split and return up to ``per_class`` image paths per class."""
    _remember_dataset_root(req.base_dir)
    base = Path(req.base_dir)
    split_dir = base / req.split
    if not split_dir.is_dir():
        return DatasetSamplesResponse(
            base_dir=req.base_dir,
            split=req.split,
            samples={},
            message=f"Split '{req.split}' não encontrado em {base}.",
        )

    samples: dict[str, list[str]] = {}
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        picks: list[str] = []
        for p in sorted(class_dir.iterdir()):
            if len(picks) >= req.per_class:
                break
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                picks.append(str(p.resolve()))
        samples[class_dir.name] = picks

    return DatasetSamplesResponse(
        base_dir=str(base.resolve()),
        split=req.split,
        samples=samples,
    )


def _collect_dataset_stats(req: DatasetStatsRequest) -> DatasetStatsResponse:
    """Walk an ImageFolder layout and count per-class image files per split."""
    _remember_dataset_root(req.base_dir)
    base = Path(req.base_dir)
    if not base.exists() or not base.is_dir():
        return DatasetStatsResponse(
            base_dir=req.base_dir,
            splits={},
            class_names=[],
            imbalanced=False,
            message=f"Diretório base não encontrado: {req.base_dir}",
        )

    split_map = {"train": req.train_dir, "val": req.val_dir, "test": req.test_dir}
    splits: dict[str, SplitStats] = {}
    all_class_names: set[str] = set()

    for split_label, subdir in split_map.items():
        split_dir = base / subdir
        if not split_dir.is_dir():
            splits[split_label] = SplitStats(total_images=0, classes={}, missing=True)
            continue

        counts: dict[str, int] = {}
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_count = sum(
                1
                for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
            )
            counts[class_dir.name] = class_count
            all_class_names.add(class_dir.name)

        splits[split_label] = SplitStats(
            total_images=sum(counts.values()), classes=counts, missing=False
        )

    imbalanced = False
    for s in splits.values():
        non_zero = [c for c in s.classes.values() if c > 0]
        if len(non_zero) >= 2 and max(non_zero) / min(non_zero) > 2.0:
            imbalanced = True
            break

    return DatasetStatsResponse(
        base_dir=str(base.resolve()),
        splits=splits,
        class_names=sorted(all_class_names),
        imbalanced=imbalanced,
    )


def _read_yolo_class_names(base: Path) -> list[str]:
    """Class names from a ``classes.txt``/``names.txt`` at the root, else []."""
    for candidate in ("classes.txt", "names.txt"):
        path = base / candidate
        if path.is_file():
            names = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if names:
                return names
    return []


def _count_split_instances(
    images_dir: Path, labels_dir: Path
) -> tuple[int, dict[int, int], int]:
    """Return (image_count, per-class-id instance counts, unlabeled_image_count).

    An image is "unlabeled" when it has no matching label file or none of its
    lines is a valid ``class cx cy w h`` row.
    """
    images = [
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    ]
    counts: dict[int, int] = {}
    unlabeled = 0
    for img in images:
        label_path = labels_dir / f"{img.stem}.txt"
        instances = 0
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                except ValueError:
                    continue
                counts[cls] = counts.get(cls, 0) + 1
                instances += 1
        if instances == 0:
            unlabeled += 1
    return len(images), counts, unlabeled


def _collect_detection_dataset_stats(
    req: DetectionDatasetStatsRequest,
) -> DetectionDatasetStatsResponse:
    """Tally per-class annotation instances across a YOLO dataset's splits."""
    _remember_dataset_root(req.base_dir)
    base = Path(req.base_dir)
    if not base.exists() or not base.is_dir():
        return DetectionDatasetStatsResponse(
            base_dir=req.base_dir,
            splits={},
            class_names=[],
            imbalanced=False,
            message=f"Diretório base não encontrado: {req.base_dir}",
        )

    declared_names = _read_yolo_class_names(base)

    # Scan each split once, keeping raw int-id counts + a missing flag.
    raw: dict[str, tuple[int, dict[int, int], int, bool]] = {}
    max_class_id = -1
    for split in ("train", "val", "test"):
        resolved = resolve_yolo_split(base, split)
        if resolved is None:
            raw[split] = (0, {}, 0, True)
            continue
        image_count, counts, unlabeled = _count_split_instances(*resolved)
        raw[split] = (image_count, counts, unlabeled, False)
        if counts:
            max_class_id = max(max_class_id, *counts.keys())

    # Declared names win; otherwise generate up to the highest class id seen.
    num_classes = max(len(declared_names), max_class_id + 1)
    class_names = [
        declared_names[i] if i < len(declared_names) else f"class_{i}"
        for i in range(num_classes)
    ]

    def _name(cid: int) -> str:
        return class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"

    splits: dict[str, DetectionSplitStats] = {}
    aggregate: dict[str, int] = {}
    for split, (image_count, counts, unlabeled, is_missing) in raw.items():
        named_counts: dict[str, int] = {}
        for cid, n in counts.items():
            named = _name(cid)
            named_counts[named] = named_counts.get(named, 0) + n
            aggregate[named] = aggregate.get(named, 0) + n
        splits[split] = DetectionSplitStats(
            total_images=image_count,
            total_annotations=sum(counts.values()),
            class_counts=named_counts,
            unlabeled_images=unlabeled,
            missing=is_missing,
        )

    non_zero = [c for c in aggregate.values() if c > 0]
    imbalanced = len(non_zero) >= 2 and max(non_zero) / min(non_zero) > 2.0

    message = None
    if all(m for _, _, _, m in raw.values()):
        message = (
            "Nenhum split YOLO encontrado (esperado 'images/<split>' "
            "ou '<split>/images')."
        )

    return DetectionDatasetStatsResponse(
        base_dir=str(base.resolve()),
        splits=splits,
        class_names=class_names,
        imbalanced=imbalanced,
        message=message,
    )


def _open_native_checkpoint_dialog() -> CheckpointPickResponse:
    """Open a tkinter file dialog filtered to .pth/.pt and return the chosen path.

    Returns an empty path + cancelled=True when the dialog is dismissed or
    tkinter is unavailable.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return CheckpointPickResponse(
            path="",
            cancelled=True,
            message="tkinter is not available on this Python installation.",
        )

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(
            title="Selecione um checkpoint (.pth ou .pt)",
            filetypes=[
                ("PyTorch checkpoint", "*.pth *.pt"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
    except Exception as exc:  # noqa: BLE001
        return CheckpointPickResponse(
            path="",
            cancelled=True,
            message=f"Falha ao abrir o seletor: {exc}",
        )

    if not chosen:
        return CheckpointPickResponse(path="", cancelled=True, message="Cancelado.")

    return CheckpointPickResponse(path=str(Path(chosen).resolve()), cancelled=False)


def _open_native_yaml_dialog() -> DatasetPickResponse:
    """Open a tkinter file dialog filtered to .yaml/.yml and return the path.

    Returns an empty path + cancelled=True when the dialog is dismissed or
    tkinter is unavailable.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return DatasetPickResponse(
            path="",
            cancelled=True,
            message="tkinter is not available on this Python installation.",
        )

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(
            title="Selecione o data.yaml do dataset",
            filetypes=[
                ("Ultralytics data.yaml", "*.yaml *.yml"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
    except Exception as exc:  # noqa: BLE001
        return DatasetPickResponse(
            path="",
            cancelled=True,
            message=f"Falha ao abrir o seletor: {exc}",
        )

    if not chosen:
        return DatasetPickResponse(path="", cancelled=True, message="Cancelado.")

    return DatasetPickResponse(path=str(Path(chosen).resolve()), cancelled=False)


def _open_native_folder_dialog() -> DatasetPickResponse:
    """Open a tkinter folder dialog and return the chosen absolute path.

    Runs in a worker thread because tkinter blocks the calling thread. Returns
    an empty path + cancelled=True when the dialog is dismissed or unavailable.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return DatasetPickResponse(
            path="",
            cancelled=True,
            message="tkinter is not available on this Python installation.",
        )

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(title="Selecione o diretório base do dataset")
        root.destroy()
    except Exception as exc:  # noqa: BLE001
        return DatasetPickResponse(
            path="",
            cancelled=True,
            message=f"Falha ao abrir o seletor: {exc}",
        )

    if not chosen:
        return DatasetPickResponse(path="", cancelled=True, message="Cancelado.")

    resolved = Path(chosen).resolve()
    _remember_dataset_root(resolved)
    return DatasetPickResponse(path=str(resolved), cancelled=False)


def _execute_batch_predict(
    run_dir: Path, req: BatchPredictRequest
) -> BatchPredictResponse:
    """Build a one-shot ExperimentConfig for BatchPredictionBlock and run it.

    The run's stored config supplies the architecture; the request supplies
    input_dir, output_csv (default: timestamped under run_dir/predictions/),
    and the recursive flag.
    """
    run_json_path = run_dir / "run.json"
    data: dict[str, Any] = json.loads(run_json_path.read_text(encoding="utf-8"))
    _require_classification_run(data, "Inferência em lote")
    base_config_dict: dict[str, Any] = data["config"]

    checkpoint_path = data.get("artifacts", {}).get("model")
    if not checkpoint_path:
        raise FileNotFoundError(
            f"Run '{run_dir.name}' has no checkpoint recorded in artifacts.model."
        )

    input_dir = Path(req.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(
            f"input_dir does not exist or is not a directory: {input_dir}"
        )
    _remember_dataset_root(input_dir)

    if req.output_csv:
        output_csv = Path(req.output_csv)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_csv = run_dir / "predictions" / f"predictions_{ts}.csv"

    batch_config_dict = {
        **base_config_dict,
        "block": "batch_prediction",
        "batch_prediction": {
            "checkpoint_path": checkpoint_path,
            "input_dir": str(input_dir.resolve()),
            "output_csv": str(output_csv.resolve()),
            "recursive": req.recursive,
            "class_names": req.class_names,
        },
    }
    config = ExperimentConfig.model_validate(batch_config_dict)

    block = BatchPredictionBlock()
    block.setup(config)
    block.run()
    report = block.report()

    failed: list[str] = list(report.get("failed_files", []))
    return BatchPredictResponse(
        output_csv=str(output_csv.resolve()),
        total_processed=int(report.get("total_processed", 0)),
        failed_count=len(failed),
        failed_files=failed,
    )


def _execute_onnx_export(run_dir: Path, req: ExportOnnxRequest) -> ExportOnnxResponse:
    """Build a one-shot ExperimentConfig for ExportONNXBlock and run it.

    The run's stored config supplies the architecture and image size; the
    request supplies the export-specific knobs. The exported file defaults
    to ``<run_dir>/best_model.onnx`` next to the checkpoint.
    """
    run_json_path = run_dir / "run.json"
    data: dict[str, Any] = json.loads(run_json_path.read_text(encoding="utf-8"))
    # Detection exports via Ultralytics' own ONNX exporter; classification below.
    if data.get("config", {}).get("task") == "detection":
        return export_detection_run(run_dir, req, data)
    base_config_dict: dict[str, Any] = data["config"]

    checkpoint_path = data.get("artifacts", {}).get("model")
    if not checkpoint_path:
        raise FileNotFoundError(
            f"Run '{run_dir.name}' has no checkpoint recorded in artifacts.model."
        )

    output_onnx = req.output_onnx or str((run_dir / "best_model.onnx").resolve())

    export_config_dict = {
        **base_config_dict,
        "block": "export_onnx",
        "export_onnx": {
            "checkpoint_path": checkpoint_path,
            "output_onnx": output_onnx,
            "opset_version": req.opset_version,
            "dynamic_axes": req.dynamic_axes,
            "validate": req.run_validate,
            "benchmark": req.benchmark,
            "benchmark_runs": req.benchmark_runs,
        },
    }
    config = ExperimentConfig.model_validate(export_config_dict)

    block = ExportONNXBlock()
    block.setup(config)
    block.run()
    report = block.report()

    file_size = int(report.get("file_size_bytes", 0))
    return ExportOnnxResponse(
        output_onnx=output_onnx,
        file_size_bytes=file_size,
        validation=report.get("validation"),
        benchmark=report.get("benchmark"),
    )


def _require_classification_run(data: dict[str, Any], action: str) -> None:
    """Reject post-training actions that only support classification runs.

    Classification runs carry task='binary'/'multiclass'; detection runs carry
    task='detection' and have no ModelFactory/Evaluator path (their equivalents
    are tracked in PHASE7_DETECTION_PLAN brick D/F). The GUI already hides these
    actions for detection — this is the backend's defense in depth.

    Raises:
        ValueError: when the run is a detection run.
    """
    task = data.get("config", {}).get("task", "")
    if task == "detection":
        raise ValueError(f"{action} não é suportado para runs de '{task}' ainda.")


def _execute_run_test(run_dir: Path, req: RunTestRequest) -> RunTestResponse:
    """Reload a saved checkpoint and evaluate it on a new dataset path.

    Updates run.json so each model accumulates its own test history.
    """
    run_json_path = run_dir / "run.json"
    data: dict[str, Any] = json.loads(run_json_path.read_text(encoding="utf-8"))
    # Detection runs evaluate via their own mAP path; classification continues below.
    if data.get("config", {}).get("task") == "detection":
        return evaluate_detection_run(run_dir, req, data)
    base_config_dict: dict[str, Any] = data["config"]

    # Build an ExperimentConfig clone with the new dataset paths + evaluate mode.
    test_config_dict = {
        **base_config_dict,
        "classification": {
            "mode": "evaluate",
            "checkpoint_path": data["artifacts"]["model"],
        },
        "data": {
            **base_config_dict.get("data", {}),
            "base_dir": req.base_dir,
            "train_dir": req.train_dir,
            "val_dir": req.val_dir,
            "test_dir": req.test_dir,
        },
    }
    config = ExperimentConfig.model_validate(test_config_dict)

    model = ModelFactory.create(config.model)
    import torch  # local import — keeps top-level lightweight on cold start

    state_dict = torch.load(
        str(config.classification.checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)  # type: ignore[arg-type]

    data_module = DataModule(config)
    eval_result = Evaluator(config).evaluate(model, data_module.test_loader())

    timestamp = datetime.now()
    test_id = f"test_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
    tests_dir = run_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    # Render the full evaluation-time plot set for the test dataset so a
    # per-model test history mirrors what the original training produced.
    artifacts: dict[str, Any] = {}
    cm_path = tests_dir / f"{test_id}_confusion_matrix.png"
    MetricsPlotter.confusion_matrix_plot(
        eval_result.confusion_matrix, data_module.class_names, cm_path
    )
    artifacts["confusion_matrix"] = str(cm_path)

    cm_norm_path = tests_dir / f"{test_id}_confusion_matrix_normalized.png"
    MetricsPlotter.confusion_matrix_normalized(
        eval_result.confusion_matrix, data_module.class_names, cm_norm_path
    )
    artifacts["confusion_matrix_normalized"] = str(cm_norm_path)

    roc_path = tests_dir / f"{test_id}_roc_curve.png"
    if MetricsPlotter.roc_curve_plot(
        eval_result.y_true,
        eval_result.y_proba_full,
        data_module.class_names,
        roc_path,
    ):
        artifacts["roc_curve"] = str(roc_path)

    pr_path = tests_dir / f"{test_id}_precision_recall_curve.png"
    if MetricsPlotter.precision_recall_curve_plot(
        eval_result.y_true,
        eval_result.y_proba_full,
        data_module.class_names,
        pr_path,
    ):
        artifacts["precision_recall_curve"] = str(pr_path)

    metrics = {
        "accuracy": eval_result.accuracy,
        "f1": eval_result.f1,
        "precision": eval_result.precision,
        "recall": eval_result.recall,
        "auc_roc": eval_result.auc_roc,
    }

    label = req.label or Path(req.base_dir).name or "test"
    base_dir_str = str(Path(req.base_dir).resolve())
    record: dict[str, Any] = {
        "test_id": test_id,
        "label": label,
        "base_dir": base_dir_str,
        "timestamp": timestamp.isoformat(),
        "metrics": metrics,
        "artifacts": artifacts,
    }
    data.setdefault("tests", []).append(record)
    run_json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return RunTestResponse(
        test_id=test_id,
        run_id=run_dir.name,
        label=label,
        base_dir=base_dir_str,
        timestamp=timestamp,
        metrics=metrics,
        artifacts=artifacts,
    )


def _load_runs(models_dir: Path) -> list[RunSummary]:
    """Scan models_dir recursively for run.json files and parse each into RunSummary."""
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


# Per-task projection of run.json metrics onto the compact set the history
# card shows. Adding a task is one row; the frontend RunCard mirrors these keys.
# Unknown tasks fall back to the classification row.
_SUMMARY_METRIC_KEYS: dict[str, dict[str, str]] = {
    "classification": {
        "accuracy": "test_accuracy",
        "f1": "test_f1",
        "val_loss": "best_val_loss",
    },
    "detection": {
        "map50": "map50",
        "map50_95": "map50_95",
    },
}


def _summary_metrics(task: str, metrics: dict[str, Any]) -> dict[str, float]:
    """Project run.json metrics onto the task-specific history-card metric set.

    Missing or non-numeric values are skipped so a partially-written run.json
    (Ultralytics omits mAP early in training) never raises.
    """
    key_map = _SUMMARY_METRIC_KEYS.get(task, _SUMMARY_METRIC_KEYS["classification"])
    out: dict[str, float] = {}
    for label, src in key_map.items():
        value = metrics.get(src)
        if value is None:
            continue
        try:
            out[label] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _parse_run_summary(run_dir: Path, data: dict[str, Any]) -> RunSummary:
    """Build a RunSummary from a parsed run.json dict."""
    status: str = data["status"]
    started_at = datetime.fromisoformat(data["timestamp"])
    finished_at: datetime | None = started_at if status == "completed" else None

    config: dict[str, Any] = data["config"]
    metrics: dict[str, Any] = data.get("metrics", {})
    task: str = config["task"]

    final_metrics = _summary_metrics(task, metrics)

    data_cfg = config.get("data") or {}
    pp_cfg = data_cfg.get("preprocessing") or {}
    pp_steps = pp_cfg.get("steps") or []
    preprocessing_count = len(pp_steps) if isinstance(pp_steps, list) else 0

    # The block field was added in 2026-05; older run.json files don't have
    # it, so fall back to the config dict (also written by every block) and
    # finally infer from the task (detection writes no block marker) before
    # defaulting to classification.
    block = (
        data.get("block")
        or config.get("block")
        or ("detection" if task == "detection" else "classification")
    )

    return RunSummary(
        run_id=run_dir.name,
        experiment_name=data["experiment"],
        model_arch=config["model"]["name"],
        task=task,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        epochs_completed=int(metrics["total_epochs"]),
        final_metrics=final_metrics,
        preprocessing_count=preprocessing_count,
        block=block,
    )


async def _execute_experiment(config: ExperimentConfig, run_id: str) -> None:
    """Run the experiment in a background thread."""
    global _current_run, _event_queue

    loop = asyncio.get_running_loop()
    queue = _event_queue  # capture at task-start to survive queue replacement

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    try:
        # Dispatch to the right block based on config.block. The classification
        # path stays default; cross_validation reuses the same /experiment/run
        # endpoint so the frontend doesn't need a separate route. Adding more
        # blocks here is a one-liner — keep them limited to ones whose GUI
        # surface is actually wired up.
        block: (
            ClassificationBlock
            | CrossValidationBlock
            | TransferLearningBlock
            | ModelComparisonBlock
            | GridSearchBlock
            | RandomSearchBlock
        )
        if config.block == "cross_validation":
            block = CrossValidationBlock()
        elif config.block == "transfer_learning":
            block = TransferLearningBlock()
        elif config.block == "model_comparison":
            block = ModelComparisonBlock()
        elif config.block == "grid_search":
            block = GridSearchBlock()
        elif config.block == "random_search":
            block = RandomSearchBlock()
        else:
            block = ClassificationBlock()
        block.setup(config)
        # Blocks that stream live epoch progress over SSE get the event pump
        # wired here. Grid/random search forward each inner trial's epochs
        # (annotated with trial context) plus trial_start/trial_end banners.
        # CV and TransferLearning don't take a progress_callback yet.
        if isinstance(block, ClassificationBlock | GridSearchBlock | RandomSearchBlock):
            block._progress_callback = _put_event

        logger.info("GUI: Starting experiment {} (block={})", run_id, config.block)
        await asyncio.to_thread(block.run)

        report = block.report()

        # Classification and TransferLearning produce a TrainResult with a
        # model_path so the run_dir is the checkpoint's parent folder. CV
        # writes per-fold run.jsons in subdirs and has no single run_dir.
        run_dir: str | None = None
        if isinstance(block, ClassificationBlock | TransferLearningBlock):
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
        if queue is not None:
            await queue.put(None)


async def _execute_detection(config: DetectionConfig, run_id: str) -> None:
    """Run an Ultralytics detection training in a background thread."""
    global _current_run, _event_queue

    loop = asyncio.get_running_loop()
    queue = _event_queue

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    try:
        block = DetectionBlock()
        block.setup(config)
        block._progress_callback = _put_event

        logger.info("GUI: Starting detection {} ({})", run_id, config.model.name)
        await asyncio.to_thread(block.run)

        report = block.report()
        run_dir = report.get("detection", {}).get("run_dir")
        _current_run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": run_dir,
        }
        logger.success("GUI: Detection {} completed.", run_id)

    except Exception as e:
        logger.exception("GUI: Detection {} failed", run_id)
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
        if queue is not None:
            await queue.put(None)


async def _execute_regression(config: RegressionConfig, run_id: str) -> None:
    """Run an image-regression training in a background thread."""
    global _current_run, _event_queue

    loop = asyncio.get_running_loop()
    queue = _event_queue

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    try:
        block = RegressionBlock()
        block.setup(config)
        block._progress_callback = _put_event

        logger.info("GUI: Starting regression {} ({})", run_id, config.model.name)
        await asyncio.to_thread(block.run)

        report = block.report()
        run_dir = report.get("train", {}).get("run_dir")
        _current_run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": run_dir,
        }
        logger.success("GUI: Regression {} completed.", run_id)

    except Exception as e:
        logger.exception("GUI: Regression {} failed", run_id)
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
        if queue is not None:
            await queue.put(None)


async def _execute_segmentation(config: SegmentationConfig, run_id: str) -> None:
    """Run a semantic-segmentation training in a background thread."""
    global _current_run, _event_queue

    loop = asyncio.get_running_loop()
    queue = _event_queue

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    try:
        block = SegmentationBlock()
        block.setup(config)
        block._progress_callback = _put_event

        logger.info("GUI: Starting segmentation {} ({})", run_id, config.model.name)
        await asyncio.to_thread(block.run)

        report = block.report()
        run_dir = report.get("train", {}).get("run_dir")
        _current_run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": run_dir,
        }
        logger.success("GUI: Segmentation {} completed.", run_id)

    except Exception as e:
        logger.exception("GUI: Segmentation {} failed", run_id)
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
        if queue is not None:
            await queue.put(None)


async def _execute_anomaly(config: AnomalyConfig, run_id: str) -> None:
    """Run an anomaly-detection training in a background thread."""
    global _current_run, _event_queue

    loop = asyncio.get_running_loop()
    queue = _event_queue

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    try:
        block = AnomalyBlock()
        block.setup(config)
        block._progress_callback = _put_event

        logger.info("GUI: Starting anomaly {} ({})", run_id, config.model.name)
        await asyncio.to_thread(block.run)

        report = block.report()
        run_dir = report.get("train", {}).get("run_dir")
        _current_run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": run_dir,
        }
        logger.success("GUI: Anomaly {} completed.", run_id)

    except Exception as e:
        logger.exception("GUI: Anomaly {} failed", run_id)
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
