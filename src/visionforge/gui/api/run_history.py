"""Run history parsing, split out of ``routes.py`` (ADR-132).

Pure functions over ``run.json``: no FastAPI import, no module state, so the
list, the detail and the resume rules are testable without HTTP.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from loguru import logger

from visionforge.core.dataset_fingerprint import dataset_identity
from visionforge.core.resume import can_resume
from visionforge.core.task_runner import runner_metric_direction
from visionforge.gui.api.schemas import RunGroupBrief, RunSummary
from visionforge.tasks.registry import get_task

CUSTOM_TASK_PREFIX = "custom:"
RUN_BAD_JSON_KEY = "run.bad_json"
RUN_NOT_FOUND_KEY = "run.not_found"
RUN_FILENAME = "run.json"


def run_task(data: dict[str, Any]) -> str:
    """The task family a run belongs to, from run.json.

    Custom runs carry ``custom:<key>`` at the top level; the built-ins record
    their own name in ``config.task``, except classification which records the
    *problem* type (binary/multiclass/multilabel) there.
    """
    top = data.get("task")
    if isinstance(top, str) and top.startswith(CUSTOM_TASK_PREFIX):
        return top
    task = data.get("config", {}).get("task")
    if task in ("regression", "segmentation", "anomaly", "detection"):
        return str(task)
    return "classification"


# Blocks whose run directory is one training. A sweep or a K-fold writes its
# real trainings into sub-directories with their own run.json, so offering
# "continue" on the parent would continue nothing.
_RESUMABLE_BLOCKS = {"classification", "transfer_learning"}


def resume_status(run_dir: Path, data: dict[str, Any]) -> tuple[bool, int | None]:
    """Whether this run can be continued, and how many epochs it was going to run.

    Derived from what is on disk rather than from ``status``: ADR-092 made the
    resume file's presence the answer, so no stored flag can disagree with it.
    Ultralytics keeps its own state instead (ADR-093), so a YOLO run is judged
    by its ``last.pt`` and how far its history got.
    """
    config: dict[str, Any] = data.get("config") or {}
    epochs = (config.get("training") or {}).get("epochs")
    if not isinstance(epochs, int):
        return False, None
    # A replicate group trained nothing itself: its seeds are the runs one
    # continues, one at a time (ADR-113).
    if data.get("group"):
        return False, epochs

    task = run_task(data)
    if task == "classification":
        block = config.get("block") or data.get("block") or "classification"
        if block not in _RESUMABLE_BLOCKS:
            return False, epochs
    elif task not in ("regression", "segmentation", "anomaly", "detection"):
        return False, epochs

    if task == "detection" and (config.get("model") or {}).get("backend") != (
        "torchvision"
    ):
        done = (data.get("metrics") or {}).get("total_epochs") or 0
        return (run_dir / "weights" / "last.pt").is_file() and done < epochs, epochs

    return can_resume(run_dir, epochs), epochs


def metric_directions(
    task: str, metrics: dict[str, Any], history: list[dict[str, Any]]
) -> dict[str, Literal["higher", "lower"]]:
    """Which way each metric of a run improves, by the rule the rankings use.

    Covers the run's ``metrics`` keys and every history series, so the
    comparison table and its epoch curves read the same answer. A researcher's
    task (``custom:<key>``) is believed over the name: its declared directions
    apply to the metric and to the ``val_<metric>`` series the engine streams.
    A task no longer registered has nothing declared and is judged by its names.
    """
    declared: dict[str, str] = {}
    if task.startswith(CUSTOM_TASK_PREFIX):
        try:
            info = get_task(task.removeprefix(CUSTOM_TASK_PREFIX))
        except KeyError:
            info = None
        if info is not None:
            for name, direction in info.metrics.items():
                declared[name] = direction
                declared[f"val_{name}"] = direction
    names = set(metrics)
    for record in history:
        names.update(record)
    names.discard("epoch")
    holder = SimpleNamespace(metric_directions=declared)
    return {name: runner_metric_direction(holder, name) for name in sorted(names)}


def load_runs(models_dir: Path) -> list[RunSummary]:
    """Scan models_dir recursively for run.json files and parse each into RunSummary."""
    if not models_dir.exists():
        return []

    summaries: list[RunSummary] = []
    for run_json_path in models_dir.rglob(RUN_FILENAME):
        try:
            data: dict[str, Any] = json.loads(run_json_path.read_text(encoding="utf-8"))
            summary = parse_run_summary(run_json_path.parent, data)
            summaries.append(summary)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping unparsable run.json at {}: {}", run_json_path, exc)

    summaries.sort(key=lambda s: s.started_at, reverse=True)
    return summaries


# Per-task projection of run.json metrics onto the compact set the history
# card shows. Adding a task is one row; the frontend RunCard mirrors these keys.
# Unknown tasks fall back to the classification row.
# Tasks whose config tree is their own (ADR-033/036/037/038).
_STANDALONE_TASKS = frozenset({"detection", "regression", "segmentation", "anomaly"})

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
    # Without these three the history fell back to the classification keys,
    # which these tasks never write: a segmentation run showed an empty metric
    # cell and a regression run showed its loss instead of its R². The headline
    # numbers are on disk either way — the list just could not find them.
    "regression": {
        "r2": "test_r2",
        "mae": "test_mae",
        "rmse": "test_rmse",
    },
    "segmentation": {
        "miou": "test_miou",
        "dice": "test_dice",
        "pixel_acc": "test_pixel_acc",
    },
    "anomaly": {
        "auroc": "test_auroc",
        "f1": "test_image_f1",
    },
}


def summary_metrics(task: str, metrics: dict[str, Any]) -> dict[str, float]:
    """Project run.json metrics onto the task-specific history-card metric set.

    Missing or non-numeric values are skipped so a partially-written run.json
    (Ultralytics omits mAP early in training) never raises.

    A headline read from a ``test_*`` key that the run never wrote (no test
    split was scored) falls back to the validation value under the bare name and
    is projected as ``val_<label>``, so the card is not empty and says which
    split the number is from.

    A researcher-defined task (``custom:<key>``) declares its own metric names,
    so there is no fixed map: the first three numeric metrics are surfaced
    under their real names, which is what the generic history card renders.
    """
    if task.startswith(CUSTOM_TASK_PREFIX):
        declared: dict[str, float] = {}
        for name, value in metrics.items():
            if name in ("total_epochs", "best_epoch") or len(declared) >= 3:
                continue
            try:
                declared[name] = float(value)
            except (TypeError, ValueError):
                continue
        return declared

    key_map = _SUMMARY_METRIC_KEYS.get(task, _SUMMARY_METRIC_KEYS["classification"])
    out: dict[str, float] = {}
    for label, src in key_map.items():
        value = _finite_float(metrics.get(src))
        if value is not None:
            out[label] = value
            continue
        # No held-out score for this headline: the run never evaluated a test
        # split. Every trainer writes the validation score at the best epoch
        # under the bare name, so send that one, under a name that says which
        # split it is (the card prints it as "(val)").
        if src.startswith("test_"):
            bare = src.removeprefix("test_")
            # Newer runs name the split (val_miou); older ones kept the bare name.
            value = _finite_float(metrics.get(f"val_{bare}"))
            if value is None:
                value = _finite_float(metrics.get(bare))
            if value is not None:
                out[f"val_{label}"] = value
    return out


def _finite_float(value: Any) -> float | None:
    """``value`` as a float, or None when it is missing or not a number."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_run_timestamp(value: str) -> datetime:
    """Parse a run.json timestamp as naive local time.

    Writers historically disagreed (CV wrote timezone-aware UTC, everything
    else naive local); a mixed list cannot be sorted. Aware values are
    converted to local time and stripped so every run compares.
    """
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


_RUN_DIR_START = re.compile(r"^(\d{8}_\d{6})(?:_\d+)?$")


def run_span(
    dir_name: str, timestamp: str, status: str
) -> tuple[datetime, datetime | None]:
    """``(started_at, finished_at)`` of a run, from what the files really say.

    ``run.json`` is written when the run ends, so its ``timestamp`` is the *end*;
    the folder is named when the run starts. Reading the timestamp as both made
    ``finished_at == started_at`` for every run. When the folder name is not a
    start time (groups, hand-made folders) or lies after the timestamp, nothing
    trustworthy says when it began and the old reading stands: no duration is
    invented. A resumed run keeps its original folder, so its span includes the
    time it sat stopped.
    """
    ended = _parse_run_timestamp(timestamp)
    finished = ended if status == "completed" else None
    match = _RUN_DIR_START.match(dir_name)
    if match is None:
        return ended, finished
    try:
        started = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return ended, finished
    if started > ended:
        return ended, finished
    return started, finished


def parse_run_summary(run_dir: Path, data: dict[str, Any]) -> RunSummary:
    """Build a RunSummary from a parsed run.json dict."""
    status: str = data["status"]
    started_at, finished_at = run_span(run_dir.name, data["timestamp"], status)

    config: dict[str, Any] = data["config"]
    metrics: dict[str, Any] = data.get("metrics", {})
    # A researcher-defined task (ADR-058) has neither `config.task` nor a
    # `model` block — its identity is the top-level `task: "custom:<key>"`
    # the engine stamps. Reading them unguarded made every custom run get
    # skipped as unparsable, so they never reached History.
    task: str = config.get("task") or data.get("task", "classification")

    group = data.get("group") if isinstance(data.get("group"), dict) else None
    # A group's metrics are the means of its seeds under the keys a single run
    # of the task writes (run_groups), so the projection reads them as any run's.
    final_metrics = summary_metrics(task, metrics)

    data_cfg = config.get("data") or {}
    pp_cfg = data_cfg.get("preprocessing") or {}
    pp_steps = pp_cfg.get("steps") or []
    preprocessing_count = len(pp_steps) if isinstance(pp_steps, list) else 0

    # The block field was added in 2026-05; older run.json files don't have
    # it, so fall back to the config dict (also written by every block) and
    # finally infer from the task (detection writes no block marker) before
    # defaulting to classification.
    # Standalone tasks (ADR-033/036/037/038) have no `block` field in their
    # config, so the old fallback labelled every one of them "classification" —
    # a segmentation run filed under the wrong block in the very filter meant to
    # separate them. Their task *is* their block.
    block = (
        data.get("block")
        or config.get("block")
        or (
            task
            if task in _STANDALONE_TASKS or task.startswith(CUSTOM_TASK_PREFIX)
            else "classification"
        )
    )

    # Custom tasks declare no architecture; the researcher's label is the most
    # useful thing to show in that column.
    model_arch = (config.get("model") or {}).get("name") or data.get("task_label", task)

    dataset_name, dataset_root = dataset_identity(data)
    resumable, configured_epochs = resume_status(run_dir, data)
    # Only a digest that proves something is sent: an ``unavailable`` fingerprint
    # has an empty one, and a method without a digest identifies nothing.
    fingerprint = data.get("dataset_fingerprint") or {}
    dataset_digest = fingerprint.get("digest") or None
    dataset_method = (fingerprint.get("method") or None) if dataset_digest else None

    return RunSummary(
        run_id=run_dir.name,
        experiment_name=data["experiment"],
        model_arch=str(model_arch),
        task=task,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        # Required: a run.json without it is incomplete and must be skipped
        # (the caller catches the KeyError), not listed with a fake 0.
        epochs_completed=int(metrics["total_epochs"]),
        final_metrics=final_metrics,
        preprocessing_count=preprocessing_count,
        block=block,
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        dataset_digest=dataset_digest,
        dataset_method=dataset_method,
        stopped=bool(data.get("stopped")),
        metric_directions=metric_directions(task, final_metrics, []),
        resumable=resumable,
        configured_epochs=configured_epochs,
        group_id=data.get("group_id"),
        group=_group_brief(group, task, final_metrics) if group else None,
    )


def _group_final_aggregates(
    task: str, group: dict[str, Any], final_metrics: dict[str, float]
) -> dict[str, Any]:
    """The aggregate behind each number of ``final_metrics``, under the same label.

    A card prints a group's mean next to the interval it came with. The report
    keys its aggregates by the metric's own name; the history projection keys
    them by label, through the run.json key the mean was stored under
    (``group.metric_keys``), so the correspondence is read, not inferred.
    """
    aggregates: dict[str, Any] = group.get("aggregates") or {}
    metric_keys: dict[str, str] = group.get("metric_keys") or {}
    if task.startswith(CUSTOM_TASK_PREFIX):
        sources = {label: label for label in final_metrics}
    else:
        sources = _SUMMARY_METRIC_KEYS.get(task, _SUMMARY_METRIC_KEYS["classification"])
    out: dict[str, Any] = {}
    for label, src in sources.items():
        name = metric_keys.get(src)
        if label in final_metrics and name in aggregates:
            out[label] = aggregates[name]
    return out


def _group_brief(
    group: dict[str, Any], task: str, final_metrics: dict[str, float]
) -> RunGroupBrief:
    """The list view of a group's ``run.json`` section."""
    kind = group["kind"]
    variants: dict[str, Any] = group.get("variants") or {}
    if kind in ("replicates", "kfold", "sweep"):
        children = group.get("children") or []
    else:
        children = [c for v in variants.values() for c in v.get("children") or []]
    if kind == "kfold":
        # The means sit under val_<name> (ADR-123), which is how the list labels
        # them; each label gets the aggregate it came from.
        aggregates = group.get("aggregates") or {}
        final_aggregates = {
            f"val_{name}": agg
            for name, agg in aggregates.items()
            if f"val_{name}" in final_metrics
        }
    elif kind == "replicates":
        final_aggregates = _group_final_aggregates(task, group, final_metrics)
    else:
        final_aggregates = {}
    return RunGroupBrief(
        kind=kind,
        metric=group.get("metric"),
        seeds=group.get("seeds") or [],
        n_requested=int(group.get("n_requested") or 0),
        n_finished=int(group.get("n_finished") or 0),
        stopped=bool(group.get("stopped")),
        child_ids=[c["run_id"] for c in children if c.get("run_id")],
        final_aggregates=final_aggregates,
        variants=list(variants),
        best_by_mean=group.get("best_by_mean"),
        mode=group.get("mode"),
        best_overrides=next(
            (
                c.get("overrides")
                for c in children
                if kind == "sweep"
                and c.get("trial_index") == group.get("best_index")
                and group.get("best_index") is not None
            ),
            None,
        ),
    )
