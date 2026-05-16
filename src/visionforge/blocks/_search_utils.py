from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
import yaml
from loguru import logger

from visionforge.blocks.classification import ClassificationBlock
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
    fieldnames = list(trials[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trials)


def write_best_config_yaml(
    yaml_path: Path,
    base_config: ExperimentConfig,
    best_idx: int,
    best_overrides: dict[str, Any],
) -> None:
    """Write the best trial's config as a YAML file."""
    best_raw: dict[str, Any] = base_config.model_dump(mode="json")
    best_raw["training"]["seed"] = base_config.training.seed + best_idx
    for dot_key, val in best_overrides.items():
        set_nested(best_raw, dot_key, val)
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(best_raw, f, default_flow_style=False, allow_unicode=True)


def run_trial(
    base_config: ExperimentConfig,
    trial_idx: int,
    total_trials: int,
    trial_seed: int,
    trial_overrides: dict[str, Any],
    trial_record: dict[str, Any],
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
    """
    base_raw: dict[str, Any] = base_config.model_dump(mode="json")
    base_raw["training"]["seed"] = trial_seed
    for dot_key, val in trial_overrides.items():
        set_nested(base_raw, dot_key, val)

    block = ClassificationBlock()
    try:
        trial_config = ExperimentConfig.model_validate(base_raw)
        block.setup(trial_config)
        block.run()
        report = block.report()

        trial_record["status"] = "success"
        trial_record["best_val_loss"] = report.get("train", {}).get("best_val_loss")
        trial_record["test_accuracy"] = report.get("eval", {}).get("accuracy")
        trial_record["test_f1"] = report.get("eval", {}).get("f1")

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


__all__ = [
    "set_nested",
    "validate_dot_keys",
    "best_trial",
    "write_trials_csv",
    "write_best_config_yaml",
    "run_trial",
]
