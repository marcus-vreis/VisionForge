from __future__ import annotations

import csv
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from loguru import logger

from visionforge.blocks.classification import ClassificationBlock
from visionforge.core.cancellation import STOPPED, STOPPED_NOTE, CancellationToken
from visionforge.core.run_groups import write_run_group
from visionforge.core.task_runner import trial_progress_wrapper
from visionforge.core.trial_name import trial_name
from visionforge.utils.config import ExperimentConfig


def set_nested(d: dict[str, Any], dot_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dot-notation key.

    Raises:
        ValueError: if any intermediate key does not exist in the dict.
    """
    keys = dot_key.split(".")
    node = d
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise ValueError(
                f"Unknown hyperparameter path: '{dot_key}' (failed at '{k}')"
            )
        node = node[k]
    if keys[-1] not in node:
        raise ValueError(
            f"Unknown hyperparameter path: '{dot_key}' (failed at '{keys[-1]}')"
        )
    node[keys[-1]] = value


def validate_dot_keys(base_raw: dict[str, Any], search_space: dict[str, Any]) -> None:
    """Raise ValueError for any dot-key that doesn't exist in the base config dict."""
    for dot_key in search_space:
        probe = dict(base_raw.items())
        set_nested(probe, dot_key, None)


def best_trial(trials: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the successful trial with the lowest best_val_loss, or None."""
    successful = [t for t in trials if t["status"] == "success"]
    if not successful:
        return None
    return min(
        successful,
        key=lambda t: (t["best_val_loss"] is None, t["best_val_loss"] or float("inf")),
    )


def write_trials_csv(csv_path: Path, trials: list[dict[str, Any]]) -> None:
    """Write a list of trial records to a CSV file."""
    if not trials:
        return
    # Every key any trial has, in first-seen order: a trial that failed early
    # lacks keys a finished one records (its run folder), and taking the first
    # row's keys alone made the writer refuse the rest.
    fieldnames = list(dict.fromkeys(k for t in trials for k in t))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(trials)


def write_best_config_yaml(
    yaml_path: Path,
    base_config: ExperimentConfig,
    best_idx: int,
    best_overrides: dict[str, Any],
    seed: int | None = None,
) -> None:
    """Write the best trial's config as a YAML file.

    ``seed`` is the one the best trial trained on. A sweep that moves the seed
    with the trial index (random search) leaves it unset and gets
    ``base seed + best_idx``; one that trains every trial on the base seed
    (grid search) passes it.
    """
    best_raw: dict[str, Any] = base_config.model_dump(mode="json")
    best_raw["training"]["seed"] = (
        base_config.training.seed + best_idx if seed is None else seed
    )
    for dot_key, val in best_overrides.items():
        set_nested(best_raw, dot_key, val)
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(best_raw, f, default_flow_style=False, allow_unicode=True)


# Lives in core so the task-agnostic comparison can use it too; the sweeps and
# the K-fold keep importing it by this name.
make_trial_progress_wrapper = trial_progress_wrapper


def run_trial(
    base_config: ExperimentConfig,
    trial_idx: int,
    total_trials: int,
    trial_seed: int,
    trial_overrides: dict[str, Any],
    trial_record: dict[str, Any],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_token: CancellationToken | None = None,
) -> None:
    """Execute one trial and mutate trial_record in place with results or error.

    Always calls torch.cuda.empty_cache() after the trial regardless of outcome.

    Args:
        base_config: the base experiment config to derive the trial config from.
        trial_idx: 0-based trial index (used for logging).
        total_trials: total number of trials in the sweep (used for logging).
        trial_seed: seed to set for this trial's training config.
        trial_overrides: dot-notation key→value pairs to apply on top of base_config.
        trial_record: mutable dict that receives status, metrics, and error fields.
        progress_callback: injected onto the inner block so the trial's Trainer
            streams epoch progress to the GUI; None disables streaming.
        cancel_token: injected the same way, so "stop this run" reaches the
            trial that is training rather than only the sweep around it.
    """
    base_raw: dict[str, Any] = base_config.model_dump(mode="json")
    base_raw["training"]["seed"] = trial_seed
    for dot_key, val in trial_overrides.items():
        set_nested(base_raw, dot_key, val)
    base_raw["name"] = trial_name(base_raw["name"], trial_overrides)

    block = ClassificationBlock()
    try:
        trial_config = ExperimentConfig.model_validate(base_raw)
        block.setup(trial_config)
        block._progress_callback = progress_callback
        block._cancel_token = cancel_token
        block.run()
        report = block.report()

        trial_record["status"] = "success"
        # The trial's own run folder, so History folds it under the sweep (ADR-124).
        train = getattr(block, "_train_result", None)
        if train is not None and train.model_path:
            trial_record["run_dir"] = str(train.model_path.parent.resolve())
        trial_record["best_val_loss"] = report.get("train", {}).get("best_val_loss")
        trial_record["test_accuracy"] = report.get("eval", {}).get("accuracy")
        trial_record["test_f1"] = report.get("eval", {}).get("f1")
        # A trial cut short by the stop must not be ranked against trials that
        # ran to the end, nor written out as the best config (ADR-111). Only
        # the trainer knows whether it was cut: a stop that landed in the last
        # epoch, or during the test evaluation, leaves a finished trial.
        if report.get("train", {}).get("stopped"):
            trial_record["status"] = STOPPED
            trial_record["error"] = STOPPED_NOTE

        logger.info(
            "Trial {}/{} succeeded: val_loss={} accuracy={}",
            trial_idx + 1,
            total_trials,
            trial_record["best_val_loss"],
            trial_record["test_accuracy"],
        )

    except Exception as exc:  # noqa: BLE001
        trial_record["error"] = str(exc)
        logger.warning("Trial {}/{} failed: {}", trial_idx + 1, total_trials, exc)

    finally:
        del block
        torch.cuda.empty_cache()


# The per-trial numbers a classification sweep records, in the row a History
# group lists for each trial.
_TRIAL_METRIC_KEYS = (
    "test_accuracy",
    "test_f1",
    "best_val_loss",
    "accuracy",
    "f1",
    "auc_roc",
)


def write_sweep_group(
    config: ExperimentConfig,
    trials: list[dict[str, Any]],
    *,
    mode: str,
    overrides_of: Callable[[dict[str, Any]], dict[str, Any]],
    metric: str,
    direction: str,
    best_index: int | None,
    planned: int,
    stopped: bool,
) -> None:
    """Give a classification sweep or model comparison its History entry (ADR-124).

    The last step of a job whose trials and report are done, so a failure is
    logged, not raised: the researcher keeps the report and loses only the entry.
    """
    rows = [
        {
            "trial_index": t.get("trial_index", i),
            "overrides": overrides_of(t),
            "status": t["status"],
            "run_dir": t.get("run_dir"),
            "metrics": {k: t[k] for k in _TRIAL_METRIC_KEYS if t.get(k) is not None},
            "error": t.get("error") or "",
        }
        for i, t in enumerate(trials)
    ]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    try:
        write_run_group(
            kind="sweep",
            group_id=f"{config.name}_sweep_{stamp}",
            name=config.name,
            config=config.model_dump(mode="json"),
            report={
                "metric": metric,
                "metric_direction": direction,
                "mode": mode,
                "n_planned": planned,
                "trials": rows,
                "best_index": best_index,
                "stopped": stopped,
                "report_dir": str(config.output.reports_dir / config.name),
            },
            report_json=None,
            models_dir=config.output.models_dir,
        )
    except Exception:  # noqa: BLE001 - the report is on disk; do not fail the job
        logger.exception("Could not write the History entry of sweep {}", config.name)


__all__ = [
    "write_sweep_group",
    "set_nested",
    "validate_dot_keys",
    "best_trial",
    "write_trials_csv",
    "write_best_config_yaml",
    "make_trial_progress_wrapper",
    "run_trial",
]
