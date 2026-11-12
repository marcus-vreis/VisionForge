"""Starting one training run: classification and the standalone tasks (#204, ADR-196).

``POST /api/experiment/run`` and, per standalone task (detection, regression,
segmentation, anomaly), its form schema and ``run`` endpoint. Each run checks
its dataset at submit time, goes to the queue through ``submission._submit_job``
and runs in this module's executor, which writes only its own job's state
(ADR-185).

``routes.py`` includes the two routers where these routes used to be defined, so
the OpenAPI document and the route order do not change. It owns the profile
resolution and the tracker factories and passes them in, read per request. A
test that replaces an executor (``_execute_detection``...) patches this module:
the run endpoints and ``routes.resume_run`` look it up here when the job starts.

No ``from __future__ import annotations``: ``ProfileDep`` is built inside the
builders, and FastAPI resolves string annotations against module globals only.
The blocks are imported inside the executors, so importing this module stays
light (#260).
"""

import asyncio
import csv
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from visionforge.core.detection_data import resolve_yolo_split
from visionforge.core.run_trackers import TrackerFactory
from visionforge.gui.api import execution, submission
from visionforge.gui.api.datasets_routes import SCAN_BASE_MISSING_KEY, _image_stems
from visionforge.gui.api.execution import RUN_NO_MESSAGE_KEY, _job_state
from visionforge.gui.api.http_responses import (
    BAD_REQUEST_TEXT,
    NOT_FOUND_TEXT,
    TAG_ANOMALY,
    TAG_DETECTION,
    TAG_EXPERIMENTS,
    TAG_REGRESSION,
    TAG_SEGMENTATION,
    UNPROCESSABLE_TEXT,
    VALIDATION_ERROR_CONTENT,
)
from visionforge.gui.api.profiles import Profile
from visionforge.gui.api.run_queue import StopPoint
from visionforge.gui.api.schemas import RunResponse
from visionforge.gui.api.submission import (
    _GPU_UNAVAILABLE_TEXT,
    ProfileResolver,
    profile_dependency,
)
from visionforge.gui.api.tracker_exports import exported
from visionforge.utils.anomaly_config import AnomalyConfig
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.detection_config import DetectionConfig
from visionforge.utils.messages import tr
from visionforge.utils.regression_config import RegressionConfig
from visionforge.utils.segmentation_config import SegmentationConfig

# The task keys: the queue's task label, the executors' log wording and, for
# detection, the report section that names the run folder.
_DETECTION = "detection"
_REGRESSION = "regression"
_SEGMENTATION = "segmentation"
_ANOMALY = "anomaly"

# What a run endpoint documents: a bad or missing dataset, an invalid config
# and a broken GPU context.
_RUN_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"description": BAD_REQUEST_TEXT},
    404: {"description": NOT_FOUND_TEXT},
    422: {"description": UNPROCESSABLE_TEXT, "content": VALIDATION_ERROR_CONTENT},
    503: {"description": _GPU_UNAVAILABLE_TEXT},
}

# The exporters a profile turned on for a classification run (``routes.py``).
TrackersOf = Callable[[ExperimentConfig, Profile], tuple[TrackerFactory, ...]]
# The exporters a profile turned on, for any other task (ADR-197).
ProfileTrackers = Callable[[Profile], tuple[TrackerFactory, ...]]


def _new_run_id(name: str) -> str:
    """A run id from the config name and the submit time, to the microsecond."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{name}_{timestamp}"


def build_experiment_run_router(
    resolve_profile: ProfileResolver, trackers_of: TrackersOf
) -> APIRouter:
    """Mount ``POST /experiment/run``; ``routes.py`` owns both lookups."""
    router = APIRouter()
    ProfileDep = Annotated[Profile, Depends(profile_dependency(resolve_profile))]

    @router.post("/experiment/run", responses=_RUN_RESPONSES, tags=[TAG_EXPERIMENTS])
    async def run_experiment(
        config: ExperimentConfig, profile: ProfileDep
    ) -> RunResponse:
        """Start a training experiment, or queue it if one is already running."""
        config = profile.scope_model(config)
        trackers = trackers_of(config, profile)
        await asyncio.to_thread(_check_classification_classes, config)
        run_id = _new_run_id(config.name)

        return submission._submit_job(
            run_id,
            config.name,
            "classification",
            config.block,
            lambda: (
                execution._execute_experiment(config, run_id, trackers=trackers)
                if trackers
                else execution._execute_experiment(config, run_id)
            ),
            profile=profile,
        )

    return router


def _submit_task_run(
    task: str,
    config: Any,
    execute: Callable[..., Awaitable[None]],
    profile: Profile,
    trackers: tuple[TrackerFactory, ...],
    stop_at: StopPoint | Literal["auto"] = "auto",
) -> RunResponse:
    """Queue one standalone-task run; its run.json goes to ``trackers`` at the end."""
    run_id = _new_run_id(config.name)
    return submission._submit_job(
        run_id,
        config.name,
        task,
        "simple",
        lambda: exported(execute(config, run_id), run_id, trackers),
        stop_at=stop_at,
        profile=profile,
    )


def build_task_run_router(
    resolve_profile: ProfileResolver, trackers_for: ProfileTrackers
) -> APIRouter:
    """Mount each standalone task's schema and run endpoints, in their old order."""
    router = APIRouter()
    ProfileDep = Annotated[Profile, Depends(profile_dependency(resolve_profile))]

    @router.get("/detection/schema", tags=[TAG_DETECTION])
    async def get_detection_schema() -> dict[str, Any]:
        """Return the JSON Schema for DetectionConfig (drives the detection form)."""
        return DetectionConfig.model_json_schema()

    @router.post("/detection/run", responses=_RUN_RESPONSES, tags=[TAG_DETECTION])
    async def run_detection(
        config: DetectionConfig, profile: ProfileDep
    ) -> RunResponse:
        """Start an object-detection training run in the background.

        Detection shares the run queue and the /experiment/{status,events,result}
        endpoints (one run at a time, one GPU) but dispatches its own standalone
        DetectionBlock (ADR-033) rather than the ExperimentConfig path.
        """
        config = profile.scope_model(config)
        await asyncio.to_thread(_check_detection_dataset, config)
        trackers = trackers_for(profile)
        return _submit_task_run(
            _DETECTION, config, _execute_detection, profile, trackers
        )

    @router.get("/regression/schema", tags=[TAG_REGRESSION])
    async def get_regression_schema() -> dict[str, Any]:
        """Return the JSON Schema for RegressionConfig (drives the regression form)."""
        return RegressionConfig.model_json_schema()

    @router.post("/regression/run", responses=_RUN_RESPONSES, tags=[TAG_REGRESSION])
    async def run_regression(
        config: RegressionConfig, profile: ProfileDep
    ) -> RunResponse:
        """Start an image-regression training run in the background.

        Regression shares the run queue and the /experiment/{status,events,result}
        endpoints (one run at a time, one GPU) but dispatches its own standalone
        RegressionBlock (ADR-036) rather than the ExperimentConfig path.
        """
        config = profile.scope_model(config)
        await asyncio.to_thread(_check_regression_dataset, config)
        trackers = trackers_for(profile)
        return _submit_task_run(
            _REGRESSION, config, _execute_regression, profile, trackers
        )

    @router.get("/segmentation/schema", tags=[TAG_SEGMENTATION])
    async def get_segmentation_schema() -> dict[str, Any]:
        """Return the JSON Schema for SegmentationConfig (drives the segmentation form)."""
        return SegmentationConfig.model_json_schema()

    @router.post("/segmentation/run", responses=_RUN_RESPONSES, tags=[TAG_SEGMENTATION])
    async def run_segmentation(
        config: SegmentationConfig, profile: ProfileDep
    ) -> RunResponse:
        """Start a semantic-segmentation training run in the background.

        Segmentation shares the run queue and the /experiment/{status,events,result}
        endpoints (one run at a time, one GPU) but dispatches its own standalone
        SegmentationBlock (ADR-037) rather than the ExperimentConfig path.
        """
        config = profile.scope_model(config)
        await asyncio.to_thread(_check_segmentation_dataset, config)
        trackers = trackers_for(profile)
        return _submit_task_run(
            _SEGMENTATION, config, _execute_segmentation, profile, trackers
        )

    @router.get("/anomaly/schema", tags=[TAG_ANOMALY])
    async def get_anomaly_schema() -> dict[str, Any]:
        """Return the JSON Schema for AnomalyConfig (drives the anomaly form)."""
        return AnomalyConfig.model_json_schema()

    @router.post(
        "/anomaly/run",
        responses={
            400: {"description": BAD_REQUEST_TEXT},
            404: {"description": NOT_FOUND_TEXT},
            503: {"description": _GPU_UNAVAILABLE_TEXT},
        },
        tags=[TAG_ANOMALY],
    )
    async def run_anomaly(config: AnomalyConfig, profile: ProfileDep) -> RunResponse:
        """Start an anomaly-detection training run in the background.

        Anomaly shares the run queue and the /experiment/{status,events,result}
        endpoints (one run at a time, one GPU) but dispatches its own standalone
        AnomalyBlock (ADR-038) rather than the ExperimentConfig path.
        """
        config = profile.scope_model(config)
        return _submit_task_run(
            _ANOMALY,
            config,
            _execute_anomaly,
            profile,
            trackers_for(profile),
            # PatchCore has no epochs; it stops between its phases (ADR-111).
            stop_at="phase" if config.model.name == "patchcore" else "epoch",
        )

    return router


# ── dataset checks at submit time ───────────────────────────────────────────
# The run queue accepts any config that validates, so an empty or mismatched
# dataset used to enter the queue and die on the first batch with a raw English
# exception. These checks list directories and read one CSV row at most (the
# /dataset/stats walks count every file); they reuse the stats probes' split
# resolution and messages so the form and the submit agree on what is usable.


def _refuse(message: str) -> NoReturn:
    raise HTTPException(422, message)


def _check_detection_dataset(config: DetectionConfig) -> None:
    """Refuse a YOLO ``base_dir`` with no split or no train images."""
    data = config.data
    if data.data_yaml is not None or data.base_dir is None:
        return  # an explicit data.yaml names its own splits; Ultralytics owns them
    base = Path(data.base_dir)
    if not base.is_dir():
        _refuse(tr(SCAN_BASE_MISSING_KEY, path=base))
    splits = {s: resolve_yolo_split(base, s) for s in ("train", "val", "test")}
    if all(resolved is None for resolved in splits.values()):
        _refuse(tr("scan.no_yolo_split"))
    train = splits["train"]
    if train is None or not _has_detection_image(train[0]):
        _refuse(tr("submit.detection_empty_train"))


def _has_detection_image(directory: Path) -> bool:
    """Any image Ultralytics would read under ``directory``, subfolders included.

    Stricter than this would refuse datasets the trainer accepts: Ultralytics
    walks subfolders and reads formats (``.tif``, ``.avif``...) the stats probe
    does not count.
    """
    from visionforge.utils.third_party_privacy import apply_opt_outs

    apply_opt_outs()  # before ultralytics' first import in a `visionforge gui`
    from ultralytics.data.utils import IMG_FORMATS

    return any(
        p.is_file() and p.suffix.lower().lstrip(".") in IMG_FORMATS
        for p in directory.rglob("*")
    )


def _check_segmentation_dataset(config: SegmentationConfig) -> None:
    """Refuse a base_dir with no split or no image-mask pair in train."""
    data = config.data
    base = Path(data.base_dir)
    split_dirs = (data.train_dir, data.val_dir, data.test_dir)
    if not any((base / d).is_dir() for d in split_dirs):
        _refuse(tr("scan.no_seg_split"))
    train = base / data.train_dir
    if not train.is_dir():
        _refuse(tr("submit.segmentation_no_train_dir", path=train))
    # The dataset's own extension lists: the trainer pairs .tif images and masks.
    from visionforge.core.segmentation_data import IMAGE_EXTS, MASK_EXTS

    images = _image_stems(train / data.images_subdir, IMAGE_EXTS)
    masks = _image_stems(train / data.masks_subdir, MASK_EXTS)
    if not images & masks:
        _refuse(
            tr(
                "submit.segmentation_no_pairs",
                images=len(images),
                masks=len(masks),
            )
        )


def _check_regression_dataset(config: RegressionConfig) -> None:
    """Refuse a missing train CSV, absent columns or a manifest with no rows."""
    data = config.data
    base = Path(data.base_dir)
    expected = [data.image_column, *data.target_columns]
    train_csv = base / data.train_csv
    if not train_csv.is_file():
        _refuse(tr("submit.regression_no_train_csv", path=train_csv))
    for name in (data.train_csv, data.val_csv, data.test_csv):
        path = base / name
        if not path.is_file():
            continue  # val/test are optional here; the trainer reports what it needs
        try:
            with path.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames or []
                has_rows = next(reader, None) is not None
        except (OSError, UnicodeDecodeError, csv.Error):
            continue  # an unreadable file fails with its own message at load time
        missing = [c for c in expected if c not in header]
        if missing:
            _refuse(
                tr(
                    "submit.regression_missing_columns",
                    csv=name,
                    missing=", ".join(missing),
                    found=", ".join(header) or "-",
                )
            )
        if path == train_csv and not has_rows:
            _refuse(tr("submit.regression_empty_manifest", csv=name))


def _check_classification_classes(config: ExperimentConfig) -> None:
    """Refuse ``num_classes`` that disagrees with the train class folders.

    ``binary`` has one output for two folders (ADR on binary heads), so it is
    checked against 2 folders, not against ``num_classes``.
    """
    train = Path(config.data.base_dir) / config.data.train_dir
    if not train.is_dir():
        return
    # ImageFolder counts every sub-directory as a class, empty or hidden included.
    names = sorted(p.name for p in train.iterdir() if p.is_dir())
    if not names:
        return
    shown = ", ".join(names[:10]) + (", ..." if len(names) > 10 else "")
    if config.task == "binary":
        if len(names) != 2:
            _refuse(tr("submit.binary_folders", found=len(names), names=shown))
        return
    if len(names) != config.model.num_classes:
        _refuse(
            tr(
                "submit.num_classes_mismatch",
                num_classes=config.model.num_classes,
                found=len(names),
                names=shown,
            )
        )


# ── executors ───────────────────────────────────────────────────────────────


async def _execute_block(
    block_cls: type[Any],
    task: str,
    config: Any,
    run_id: str,
    resume_dir: Path | None,
    report_section: str = "train",
) -> None:
    """Run a standalone task's block in a background thread and record how it ended.

    ``report_section`` is the part of the block's report that names the run folder.
    """
    state = _job_state(run_id)

    loop = asyncio.get_running_loop()
    queue = state.log

    def _put_event(event: dict[str, Any]) -> None:
        if queue is not None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    try:
        block = block_cls()
        block.setup(config)
        block._progress_callback = _put_event
        block._cancel_token = state.token
        block._resume_dir = resume_dir

        logger.info("GUI: Starting {} {} ({})", task, run_id, config.model.name)
        await asyncio.to_thread(block.run)

        report = block.report()
        run_dir = report.get(report_section, {}).get("run_dir")
        state.run = {
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "report": report,
            "run_dir": run_dir,
        }
        logger.success("GUI: {} {} completed.", task.capitalize(), run_id)

    except Exception as e:
        logger.exception("GUI: {} {} failed", task.capitalize(), run_id)
        cls = type(e).__name__
        msg = str(e) or tr(RUN_NO_MESSAGE_KEY)
        state.run = {
            "run_id": run_id,
            "status": "failed",
            "error": f"{cls}: {msg}",
            "report": None,
            "run_dir": None,
        }

    finally:
        if queue is not None:
            await queue.put(None)


async def _execute_detection(
    config: DetectionConfig, run_id: str, resume_dir: Path | None = None
) -> None:
    """Run an Ultralytics detection training in a background thread."""
    from visionforge.blocks.detection import DetectionBlock

    await _execute_block(
        DetectionBlock, _DETECTION, config, run_id, resume_dir, _DETECTION
    )


async def _execute_regression(
    config: RegressionConfig, run_id: str, resume_dir: Path | None = None
) -> None:
    """Run an image-regression training in a background thread."""
    from visionforge.blocks.regression import RegressionBlock

    await _execute_block(RegressionBlock, _REGRESSION, config, run_id, resume_dir)


async def _execute_segmentation(
    config: SegmentationConfig, run_id: str, resume_dir: Path | None = None
) -> None:
    """Run a semantic-segmentation training in a background thread."""
    from visionforge.blocks.segmentation import SegmentationBlock

    await _execute_block(SegmentationBlock, _SEGMENTATION, config, run_id, resume_dir)


async def _execute_anomaly(
    config: AnomalyConfig, run_id: str, resume_dir: Path | None = None
) -> None:
    """Run an anomaly-detection training in a background thread."""
    from visionforge.blocks.anomaly import AnomalyBlock

    await _execute_block(AnomalyBlock, _ANOMALY, config, run_id, resume_dir)
