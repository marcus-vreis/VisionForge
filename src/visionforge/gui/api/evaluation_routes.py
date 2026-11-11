"""Per-run evaluation endpoints, split out of ``routes.py`` (#204, ADR-132).

Testing a saved checkpoint on another dataset, Grad-CAM overlays, batch
prediction over a folder and ONNX export: each one reloads a finished run's
checkpoint and works in a thread. ``routes.py`` owns the run lookup and the
profile resolution and passes them in, read per request, so tests that repoint
``routes._MODELS_DIR`` or ``routes._find_run_dir`` still reach these routes.

No ``from __future__ import annotations``: ``ProfileDep`` is built inside the
builder, and FastAPI resolves string annotations against module globals only.
Torch, the blocks and the plotter are imported inside the executors, so
importing this module stays light (#260).
"""

import asyncio
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException
from loguru import logger

from visionforge.gui.api.datasets_routes import _remember_dataset_root
from visionforge.gui.api.detection_export import export_detection_run
from visionforge.gui.api.detection_testing import evaluate_detection_run
from visionforge.gui.api.http_responses import (
    BAD_REQUEST_TEXT,
    NOT_FOUND_TEXT,
    SERVER_ERROR_TEXT,
    TAG_RUNS,
)
from visionforge.gui.api.profiles import PROFILE_HEADER, Profile
from visionforge.gui.api.run_history import (
    CUSTOM_TASK_PREFIX,
    RUN_FILENAME,
    RUN_NOT_FOUND_KEY,
)
from visionforge.gui.api.run_history import run_task as _run_task
from visionforge.gui.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ExportOnnxRequest,
    ExportOnnxResponse,
    GradCamItem,
    GradCamRequest,
    GradCamResponse,
    RunTestRequest,
    RunTestResponse,
)
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.messages import tr


def build_evaluation_router(
    find_run_dir: Callable[[str, Profile], Path | None],
    resolve_profile: Callable[[str | None], Profile],
) -> APIRouter:
    """Mount test, Grad-CAM, batch prediction and ONNX export of a saved run.

    ``routes.py`` owns both lookups; they are called per request.
    """
    router = APIRouter(tags=[TAG_RUNS])

    def profile_dep(
        x_vf_profile: Annotated[str | None, Header(alias=PROFILE_HEADER)] = None,
    ) -> Profile:
        return resolve_profile(x_vf_profile)

    ProfileDep = Annotated[Profile, Depends(profile_dep)]

    @router.post(
        "/runs/{run_id}/test",
        responses={
            400: {"description": BAD_REQUEST_TEXT},
            404: {"description": NOT_FOUND_TEXT},
            500: {"description": SERVER_ERROR_TEXT},
        },
    )
    async def test_run_on_dataset(
        run_id: str, req: RunTestRequest, profile: ProfileDep
    ) -> RunTestResponse:
        """Run the saved checkpoint against a new dataset and record the result.

        The test result is appended to run.json under the 'tests' array so each
        saved model accumulates its own per-dataset history.
        """
        run_dir = find_run_dir(run_id, profile)
        if run_dir is None:
            raise HTTPException(404, tr(RUN_NOT_FOUND_KEY, run_id=run_id))

        _reject_group_action(run_dir)
        _reject_custom_task_action(run_dir, "test")

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

    @router.post(
        "/runs/{run_id}/gradcam",
        responses={
            400: {"description": BAD_REQUEST_TEXT},
            404: {"description": NOT_FOUND_TEXT},
            500: {"description": SERVER_ERROR_TEXT},
        },
    )
    async def gradcam_run(
        run_id: str, req: GradCamRequest, profile: ProfileDep
    ) -> GradCamResponse:
        """Generate Grad-CAM overlays for sample images using this run's checkpoint.

        Classification-only (detection/regression/segmentation/anomaly have no
        softmax-classifier CAM path). Overlays are written to ``<run_dir>/gradcam/``.
        """
        run_dir = find_run_dir(run_id, profile)
        if run_dir is None:
            raise HTTPException(404, tr(RUN_NOT_FOUND_KEY, run_id=run_id))
        _reject_group_action(run_dir)
        _reject_custom_task_action(run_dir, "gradcam")
        try:
            return await asyncio.to_thread(_execute_run_gradcam, run_dir, req)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Grad-CAM on run {} failed", run_id)
            raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

    @router.post(
        "/runs/{run_id}/batch_predict",
        responses={
            400: {"description": BAD_REQUEST_TEXT},
            404: {"description": NOT_FOUND_TEXT},
            500: {"description": SERVER_ERROR_TEXT},
        },
    )
    async def batch_predict_run(
        run_id: str, req: BatchPredictRequest, profile: ProfileDep
    ) -> BatchPredictResponse:
        """Run inference of this run's checkpoint over a folder of images.

        Outputs a CSV with one row per image. ``output_csv`` defaults to a
        timestamped file in ``<run_dir>/predictions/`` so consecutive batch runs
        don't overwrite each other.
        """
        run_dir = find_run_dir(run_id, profile)
        if run_dir is None:
            raise HTTPException(404, tr(RUN_NOT_FOUND_KEY, run_id=run_id))

        _reject_group_action(run_dir)
        _reject_custom_task_action(run_dir, "batch_predict")
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

    @router.post(
        "/runs/{run_id}/export_onnx",
        responses={
            400: {"description": BAD_REQUEST_TEXT},
            404: {"description": NOT_FOUND_TEXT},
            500: {"description": SERVER_ERROR_TEXT},
        },
    )
    async def export_run_to_onnx(
        run_id: str, req: ExportOnnxRequest, profile: ProfileDep
    ) -> ExportOnnxResponse:
        """Export this run's saved checkpoint to ONNX, optionally validating it.

        Reuses the run's stored ``config.data.transforms.image_size`` and
        ``config.model`` to build the dummy input and the architecture for export.
        """
        run_dir = find_run_dir(run_id, profile)
        if run_dir is None:
            raise HTTPException(404, tr(RUN_NOT_FOUND_KEY, run_id=run_id))

        _reject_group_action(run_dir)
        _reject_custom_task_action(run_dir, "export_onnx")
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

    return router


def _reject_group_action(run_dir: Path) -> None:
    """Refuse a checkpoint-based action on a replicate group (ADR-113).

    A group trained nothing itself: its ``run.json`` has no checkpoint, so
    continuing, testing, batch prediction, ONNX export and Grad-CAM have
    nothing to act on. They used to fail by accident (a 409 on resume, a "no
    checkpoint" error from deep inside the others); the researcher now gets the
    reason and the way out. Deleting the group (removes only its summary) and
    revealing or exporting it stay allowed, so they do not call this.

    Raises:
        HTTPException: 400 pointing at the seeds, which are ordinary runs.
    """
    try:
        data = json.loads((run_dir / RUN_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(data, dict) and isinstance(data.get("group"), dict):
        raise HTTPException(400, tr("run.group_action"))


def _reject_custom_task_action(run_dir: Path, action: str) -> None:
    """Refuse a per-run action on a researcher-defined task's run.

    These four actions rebuild the model from ``config.model`` and assume one of
    the five built-in shapes. A custom run.json has no ``config.task`` and no
    ``config.model`` — the task owns those — so the dispatchers fell through to
    the classification default and died inside ``load_state_dict`` with a raw
    ResNet key mismatch. The task's own key lives at the top level of run.json
    (ADR-013), which is what makes an honest refusal possible here.

    Raises:
        HTTPException: 400 naming the task, so the GUI can hide the action.
    """
    run_json = run_dir / RUN_FILENAME
    if not run_json.exists():
        return
    try:
        task = json.loads(run_json.read_text(encoding="utf-8")).get("task", "")
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(task, str) and task.startswith(CUSTOM_TASK_PREFIX):
        raise HTTPException(
            400,
            tr(
                "run.custom_action", action=action, task=task[len(CUSTOM_TASK_PREFIX) :]
            ),
        )


def _execute_batch_predict(
    run_dir: Path, req: BatchPredictRequest
) -> BatchPredictResponse:
    """Dispatch folder batch-prediction to the run's task path.

    Classification runs go through ``BatchPredictionBlock`` (softmax/sigmoid CSV);
    regression and anomaly runs reuse the shared ``core.batch_predict`` helper
    (continuous targets / anomaly score + decision → CSV, ADR-041 slice 3).
    Detection/segmentation are not yet supported and are rejected. The request
    supplies ``input_dir``, ``output_csv`` (default: timestamped under
    ``run_dir/predictions/``) and the recursive flag.
    """
    from visionforge.blocks.batch_prediction import BatchPredictionBlock
    from visionforge.gui.api.torch_batch_predict import (
        batch_predict_anomaly_run,
        batch_predict_regression_run,
    )

    run_json_path = run_dir / RUN_FILENAME
    data: dict[str, Any] = json.loads(run_json_path.read_text(encoding="utf-8"))
    task = str(data.get("config", {}).get("task", ""))

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

    if task in ("regression", "anomaly"):
        runner = (
            batch_predict_regression_run
            if task == "regression"
            else batch_predict_anomaly_run
        )
        summary = runner(
            run_name=run_dir.name,
            data=data,
            input_dir=input_dir,
            output_csv=output_csv.resolve(),
            recursive=req.recursive,
        )
        return BatchPredictResponse(
            output_csv=summary.output_csv,
            total_processed=summary.total_processed,
            failed_count=len(summary.failed_files),
            failed_files=summary.failed_files,
        )

    if task in ("detection", "segmentation"):
        raise ValueError(tr("testrun.batch_unsupported", task=task))

    base_config_dict: dict[str, Any] = data["config"]

    checkpoint_path = data.get("artifacts", {}).get("model")
    if not checkpoint_path:
        raise FileNotFoundError(
            f"Run '{run_dir.name}' has no checkpoint recorded in artifacts.model."
        )

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
    from visionforge.blocks.export_onnx import ExportONNXBlock
    from visionforge.gui.api.torch_onnx_export import (
        export_regression_run,
        export_segmentation_run,
    )

    run_json_path = run_dir / RUN_FILENAME
    data: dict[str, Any] = json.loads(run_json_path.read_text(encoding="utf-8"))
    # Detection exports via Ultralytics; regression/segmentation via the shared
    # core helper (their own factory builds the model); classification below.
    task = data.get("config", {}).get("task")
    if task == "detection":
        return export_detection_run(run_dir, req, data)
    if task == "regression":
        return export_regression_run(run_dir, req, data)
    if task == "segmentation":
        return export_segmentation_run(run_dir, req, data)
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


def _evaluate_standalone_run(
    run_dir: Path, req: RunTestRequest, data: dict[str, Any], task: str
) -> RunTestResponse:
    """Score a regression / segmentation / anomaly checkpoint on one folder.

    Each task's trainer already knows how to evaluate; all that was missing was
    a way to point its *test* split at the folder the researcher chose. For the
    folder-shaped tasks that is ``base_dir=<parent>`` + ``test_dir=<name>``;
    regression's manifest model makes the chosen path a ``.csv`` instead, so its
    parent becomes the base and the file becomes ``test_csv``.

    Raises:
        FileNotFoundError: the run has no checkpoint on disk.
        ValueError: the chosen path does not exist, or an anomaly run has no
            normal training split to calibrate its threshold on.
    """
    import torch

    # What the researcher just chose is validated first: a complaint about the
    # run's checkpoint is confusing when the actual problem is the path they
    # typed a second ago.
    chosen = Path(req.data_dir)
    if not chosen.exists():
        raise ValueError(tr("testrun.path_not_found", path=chosen))
    if task == "regression" and chosen.is_dir():
        raise ValueError(tr("testrun.regression_needs_csv"))

    checkpoint = (data.get("artifacts") or {}).get("model")
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError(
            tr("testrun.no_checkpoint", run=run_dir.name, model=repr(checkpoint))
        )

    config_dict = dict(data["config"])
    data_dict = dict(config_dict.get("data", {}))
    if task == "regression":
        data_dict |= {"base_dir": str(chosen.parent), "test_csv": chosen.name}
    else:
        data_dict |= {"base_dir": str(chosen.parent), "test_dir": chosen.name}
    config_dict["data"] = data_dict

    if task == "regression":
        from visionforge.core.regression_data import RegressionDataModule
        from visionforge.core.regression_trainer import RegressionTrainer
        from visionforge.models.regression_factory import RegressionModelFactory
        from visionforge.utils.regression_config import RegressionConfig

        reg_config = RegressionConfig.model_validate(config_dict)
        reg_model = RegressionModelFactory.create(reg_config.model)
        reg_model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        # The chosen manifest is the only split: a held-out set rarely has
        # train.csv / val.csv beside it.
        reg_loader = RegressionDataModule(
            reg_config, evaluation_only=True
        ).test_loader()
        if reg_loader is None:
            raise ValueError(tr("testrun.no_rows", path=chosen))
        mse, rmse, mae, r2 = RegressionTrainer(reg_config).evaluate(
            reg_model, reg_loader
        )
        metrics: dict[str, Any] = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    elif task == "segmentation":
        from visionforge.core.segmentation_data import SegmentationDataModule
        from visionforge.core.segmentation_trainer import SegmentationTrainer
        from visionforge.models.segmentation_factory import SegmentationModelFactory
        from visionforge.utils.segmentation_config import SegmentationConfig

        seg_config = SegmentationConfig.model_validate(config_dict)
        # Without this the pairing step walks into the missing folder and the
        # operating system's own message (in its own language) reaches the screen.
        test_split = seg_config.data.base_dir / seg_config.data.test_dir
        masks_dir = test_split / seg_config.data.masks_subdir
        if (test_split / seg_config.data.images_subdir).is_dir() and (
            not masks_dir.is_dir()
        ):
            raise ValueError(
                tr(
                    "testrun.masks_missing",
                    split=test_split,
                    images=seg_config.data.images_subdir,
                    masks=seg_config.data.masks_subdir,
                    masks_path=masks_dir,
                )
            )
        seg_model = SegmentationModelFactory.create(seg_config.model)
        seg_model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        seg_loader = SegmentationDataModule(
            seg_config, evaluation_only=True
        ).test_loader()
        if seg_loader is None:
            raise ValueError(tr("testrun.no_pairs", path=chosen))
        miou, dice, pixel_acc = SegmentationTrainer(seg_config).evaluate(
            seg_model, seg_loader
        )
        metrics = {"miou": miou, "dice": dice, "pixel_acc": pixel_acc}

    else:  # anomaly
        from visionforge.core.anomaly_data import AnomalyDataModule
        from visionforge.core.anomaly_trainer import AnomalyTrainer
        from visionforge.models.anomaly_factory import AnomalyModelFactory
        from visionforge.utils.anomaly_config import AnomalyConfig

        anom_config = AnomalyConfig.model_validate(config_dict)
        # Checked before the checkpoint is loaded: the data module's own message
        # ("no normal training images") does not tell someone who just picked a
        # test folder why a training folder is being asked for.
        train_normal = (
            anom_config.data.base_dir
            / anom_config.data.train_dir
            / anom_config.data.normal_dir
        )
        if not train_normal.is_dir():
            raise ValueError(
                tr(
                    "testrun.anomaly_train_missing",
                    path=train_normal,
                    train_dir=anom_config.data.train_dir,
                    normal_dir=anom_config.data.normal_dir,
                )
            )
        anom_model = AnomalyModelFactory.create(anom_config.model)
        anom_model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        modules = AnomalyDataModule(anom_config)
        # The threshold comes from the *normal training* distribution, so that
        # split is still needed even though only the chosen folder is being
        # scored (ADR-076) — read without augmentation, which is the
        # distribution the model is applied to (ADR-098).
        auroc, threshold, image_f1 = AnomalyTrainer(anom_config).evaluate(
            anom_model, modules.calibration_loader(), modules.test_loader()
        )
        metrics = {"auroc": auroc, "threshold": threshold, "image_f1": image_f1}

    timestamp = datetime.now()
    test_id = f"test_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
    label = req.label or chosen.name or "test"
    resolved = str(chosen.resolve())
    record: dict[str, Any] = {
        "test_id": test_id,
        "label": label,
        "base_dir": resolved,
        "timestamp": timestamp.isoformat(),
        "metrics": metrics,
        "artifacts": {},
    }
    data.setdefault("tests", []).append(record)
    (run_dir / RUN_FILENAME).write_text(json.dumps(data, indent=2), encoding="utf-8")

    return RunTestResponse(
        test_id=test_id,
        run_id=run_dir.name,
        label=label,
        base_dir=resolved,
        timestamp=timestamp,
        metrics=metrics,
        artifacts={},
    )


def _class_root_or_raise(folder: Path) -> Path:
    """The folder a classification test reads: one sub-folder per class.

    A class folder (images, no sub-folders) is the likeliest wrong pick, and
    ImageFolder would only say it found no class there.
    """
    if not folder.is_dir():
        raise FileNotFoundError(tr("testrun.path_not_found", path=str(folder)))
    if not any(p.is_dir() for p in folder.iterdir()):
        raise ValueError(tr("testrun.not_class_root", path=str(folder)))
    return folder


def _execute_run_test(run_dir: Path, req: RunTestRequest) -> RunTestResponse:
    """Reload a saved checkpoint and evaluate it on a new dataset path.

    Updates run.json so each model accumulates its own test history.
    """
    from visionforge.core.data import class_folder_test_loader
    from visionforge.core.evaluator import Evaluator
    from visionforge.core.plotter import MetricsPlotter
    from visionforge.models.factory import ModelFactory

    run_json_path = run_dir / RUN_FILENAME
    data: dict[str, Any] = json.loads(run_json_path.read_text(encoding="utf-8"))
    task = _run_task(data)

    # Each family evaluates through its own trainer (ADR-080). Detection had a
    # native path already; the other three used to fall into the classification
    # branch below and surface a raw ExperimentConfig validation error.
    if task == "detection":
        return evaluate_detection_run(run_dir, req, data)
    if task in ("regression", "segmentation", "anomaly"):
        return _evaluate_standalone_run(run_dir, req, data, task)

    base_config_dict: dict[str, Any] = data["config"]

    # The same guard as the other families: a run stopped before its first
    # epoch records no checkpoint (ADR-111), and evaluating "None" would fail
    # with an error about a path instead of saying what is missing.
    checkpoint = (data.get("artifacts") or {}).get("model")
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError(
            tr("testrun.no_checkpoint", run=run_dir.name, model=repr(checkpoint))
        )

    # The chosen folder is the evaluation set on its own (ADR-080): it is not
    # a split of a dataset root, so nothing beside it is required to exist.
    chosen = _class_root_or_raise(Path(req.data_dir))
    config = ExperimentConfig.model_validate(
        {
            **base_config_dict,
            "classification": {"mode": "evaluate", "checkpoint_path": checkpoint},
            # The training folder may have moved since; config validation only
            # needs an existing base_dir, and nothing here reads from it.
            "data": {**base_config_dict.get("data", {}), "base_dir": str(chosen)},
        }
    )
    class_names = _gradcam_class_names(data) or sorted(
        p.name for p in chosen.iterdir() if p.is_dir()
    )

    model = ModelFactory.create(config.model)
    import torch  # local import — keeps top-level lightweight on cold start

    state_dict = torch.load(
        str(config.classification.checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)  # type: ignore[arg-type]

    loader = class_folder_test_loader(config, chosen, class_names)
    eval_result = Evaluator(config).evaluate(model, loader)

    timestamp = datetime.now()
    test_id = f"test_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
    tests_dir = run_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    # Render the full evaluation-time plot set for the test dataset so a
    # per-model test history mirrors what the original training produced.
    artifacts: dict[str, Any] = {}
    cm_path = tests_dir / f"{test_id}_confusion_matrix.png"
    MetricsPlotter.confusion_matrix_plot(
        eval_result.confusion_matrix, class_names, cm_path
    )
    artifacts["confusion_matrix"] = str(cm_path)

    cm_norm_path = tests_dir / f"{test_id}_confusion_matrix_normalized.png"
    MetricsPlotter.confusion_matrix_normalized(
        eval_result.confusion_matrix, class_names, cm_norm_path
    )
    artifacts["confusion_matrix_normalized"] = str(cm_norm_path)

    roc_path = tests_dir / f"{test_id}_roc_curve.png"
    if MetricsPlotter.roc_curve_plot(
        eval_result.y_true,
        eval_result.y_proba_full,
        class_names,
        roc_path,
    ):
        artifacts["roc_curve"] = str(roc_path)

    pr_path = tests_dir / f"{test_id}_precision_recall_curve.png"
    if MetricsPlotter.precision_recall_curve_plot(
        eval_result.y_true,
        eval_result.y_proba_full,
        class_names,
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

    label = req.label or Path(req.data_dir).name or "test"
    base_dir_str = str(Path(req.data_dir).resolve())
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


_GRADCAM_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _gradcam_class_names(data: dict[str, Any]) -> list[str]:
    """Class names for a classification run, in the model's own index order.

    Recovered from the training folder rather than run.json, which never stored
    them: ``ImageFolder`` assigns indices by sorting the class sub-directories,
    so the sorted folder names *are* the index→name mapping the checkpoint was
    trained with. Returns an empty list when the folder is gone or the task is
    not classification — the caller then reports indices, as before.
    """
    config = data.get("config", {})
    if config.get("task") not in ("binary", "multiclass"):
        return []
    # Runs written from 0.7.1 on record the mapping, which is the only form that
    # survives the dataset being renamed or moved.
    recorded = data.get("class_names")
    if isinstance(recorded, list) and recorded:
        return [str(c) for c in recorded]
    data_config = config.get("data", {})
    base_dir = data_config.get("base_dir")
    if not base_dir:
        return []
    train_dir = Path(base_dir) / str(data_config.get("train_dir", "train"))
    if not train_dir.is_dir():
        return []
    names = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    # A binary run trained with a single output unit maps 1 -> the second class;
    # anything else that does not line up is safer left unnamed.
    expected = config.get("model", {}).get("num_classes")
    if expected and len(names) != expected and not (expected == 1 and len(names) == 2):
        return []
    return names


def _execute_run_gradcam(run_dir: Path, req: GradCamRequest) -> GradCamResponse:
    """Rebuild this run's model and write Grad-CAM overlays for sample images.

    Dispatches by task (classification / regression / segmentation) via
    ``build_gradcam``; detection and anomaly are unsupported. Model + transform are
    read leniently from run.json so a moved dataset path does not block it; the
    checkpoint supplies the weights (architecture rebuilt with ``pretrained=False``,
    no download).
    """
    import torch
    from PIL import Image

    from visionforge.core.gradcam import GradCAM, overlay_cam, resolve_target_layer
    from visionforge.gui.api.torch_gradcam import build_gradcam

    data: dict[str, Any] = json.loads(
        (run_dir / RUN_FILENAME).read_text(encoding="utf-8")
    )
    checkpoint_path = data.get("artifacts", {}).get("model")
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"Run '{run_dir.name}' has no usable checkpoint at artifacts.model."
        )

    setup = build_gradcam(
        data, target_index=req.target_class, checkpoint=checkpoint_path
    )

    input_dir = Path(req.input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"input_dir is not a directory: {input_dir}")
    walker = input_dir.rglob("*") if req.recursive else input_dir.iterdir()
    images = sorted(
        p for p in walker if p.is_file() and p.suffix.lower() in _GRADCAM_IMAGE_EXTS
    )[: req.num_samples]
    if not images:
        raise ValueError(f"No image files found under {input_dir}.")

    target_layer = resolve_target_layer(setup.model)
    cam = GradCAM(setup.model, target_layer)
    out_dir = run_dir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = _gradcam_class_names(data)

    items: list[GradCamItem] = []
    try:
        for img_path in images:
            with Image.open(img_path) as im:
                tensor = setup.transform(im.convert("RGB")).unsqueeze(0)
            heat = cam(tensor, target_fn=setup.target_fn)
            with torch.no_grad():
                output = setup.model(tensor)
            predicted_class, prediction = setup.describe(output)
            overlay = overlay_cam(tensor[0], heat, alpha=req.alpha)
            overlay_path = out_dir / f"{img_path.stem}_gradcam.png"
            overlay.save(overlay_path)

            predicted_label = (
                class_names[predicted_class]
                if class_names
                and predicted_class is not None
                and 0 <= predicted_class < len(class_names)
                else None
            )
            # Only a folder named after one of this run's classes counts as
            # ground truth. A flat folder of unlabeled images has none, and
            # inventing one would be worse than showing nothing.
            parent = img_path.parent.name
            true_class = parent if parent in class_names else None
            items.append(
                GradCamItem(
                    source=str(img_path),
                    overlay=str(overlay_path),
                    predicted_class=predicted_class,
                    prediction=prediction,
                    predicted_label=predicted_label,
                    true_class=true_class,
                    correct=(
                        predicted_label == true_class
                        if true_class is not None and predicted_label is not None
                        else None
                    ),
                )
            )
    finally:
        cam.remove()

    return GradCamResponse(
        run_id=run_dir.name,
        count=len(items),
        target_layer=type(target_layer).__name__,
        items=items,
    )
