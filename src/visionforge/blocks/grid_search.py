from __future__ import annotations

import csv
import itertools
from typing import Any

import torch
import yaml
from loguru import logger

from visionforge.blocks.base import ExperimentBlock
from visionforge.blocks.classification import ClassificationBlock
from visionforge.utils.config import ExperimentConfig


def _set_nested(d: dict[str, Any], dot_key: str, value: Any) -> None:
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


def _validate_search_space(
    base_raw: dict[str, Any], hyperparameters: dict[str, list[Any]]
) -> None:
    """Raise ValueError for any dot-key that doesn't exist in the base config dict."""
    for dot_key in hyperparameters:
        probe = dict(base_raw.items())
        _set_nested(probe, dot_key, None)


class GridSearchBlock(ExperimentBlock):
    """Exhaustive hyperparameter sweep via Cartesian product of a declared search space."""

    def setup(self, config: ExperimentConfig) -> None:
        """Validate the search space against the base config and store state.

        Raises:
            ValueError: if grid_search config is missing or any dot-key is unknown.
        """
        if config.grid_search is None:
            raise ValueError(
                "GridSearchBlock requires grid_search to be set in ExperimentConfig."
            )

        base_raw: dict[str, Any] = config.model_dump(mode="json")
        _validate_search_space(base_raw, config.grid_search.hyperparameters)

        self._config = config
        self._trials: list[dict[str, Any]] = []

    def run(self) -> None:
        """Execute all trials in the Cartesian product of the search space."""
        hp = self._config.grid_search.hyperparameters  # type: ignore[union-attr]
        base_seed = self._config.training.seed

        if hp:
            keys = list(hp.keys())
            value_lists = [hp[k] for k in keys]
            combos: list[tuple[Any, ...]] = list(itertools.product(*value_lists))
        else:
            keys = []
            combos = [()]

        for trial_idx, combo in enumerate(combos):
            trial_seed = base_seed + trial_idx
            trial_overrides = dict(zip(keys, combo, strict=True))

            base_raw: dict[str, Any] = self._config.model_dump(mode="json")
            base_raw["training"]["seed"] = trial_seed
            for dot_key, val in trial_overrides.items():
                _set_nested(base_raw, dot_key, val)

            trial_record: dict[str, Any] = {
                "trial_index": trial_idx,
                "seed": trial_seed,
                **trial_overrides,
                "status": "failed",
                "error": "",
                "best_val_loss": None,
                "test_accuracy": None,
                "test_f1": None,
            }

            try:
                trial_config = ExperimentConfig.model_validate(base_raw)
                block = ClassificationBlock()
                block.setup(trial_config)
                block.run()
                report = block.report()

                trial_record["status"] = "success"
                trial_record["best_val_loss"] = report.get("train", {}).get(
                    "best_val_loss"
                )
                trial_record["test_accuracy"] = report.get("eval", {}).get("accuracy")
                trial_record["test_f1"] = report.get("eval", {}).get("f1")

                logger.info(
                    "Trial {}/{} succeeded: val_loss={} accuracy={}",
                    trial_idx + 1,
                    len(combos),
                    trial_record["best_val_loss"],
                    trial_record["test_accuracy"],
                )

                del block
                torch.cuda.empty_cache()

            except Exception as exc:  # noqa: BLE001
                trial_record["error"] = str(exc)
                logger.warning(
                    "Trial {}/{} failed: {}", trial_idx + 1, len(combos), exc
                )
                torch.cuda.empty_cache()

            self._trials.append(trial_record)

        self._write_artifacts()

    def report(self) -> dict[str, Any]:
        """Return best trial metrics.

        Raises:
            RuntimeError: if no trials succeeded.
        """
        successful = [t for t in self._trials if t["status"] == "success"]
        if not successful:
            raise RuntimeError(
                "GridSearchBlock: all trials failed — no best trial available."
            )

        best = min(
            successful,
            key=lambda t: (
                t["best_val_loss"] is None,
                t["best_val_loss"] or float("inf"),
            ),
        )
        return {
            "best_trial": best,
            "total_trials": len(self._trials),
            "successful_trials": len(successful),
        }

    # ── private ───────────────────────────────────────────────────────────────

    def _write_artifacts(self) -> None:
        """Write grid_search_summary.csv and best_config.yaml to reports_dir."""
        out_dir = self._config.output.reports_dir / self._config.name
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "grid_search_summary.csv"
        if self._trials:
            fieldnames = list(self._trials[0].keys())
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._trials)

        successful = [t for t in self._trials if t["status"] == "success"]
        if not successful:
            return

        best = min(
            successful,
            key=lambda t: (
                t["best_val_loss"] is None,
                t["best_val_loss"] or float("inf"),
            ),
        )
        best_idx = best["trial_index"]

        # Reconstruct best trial config raw dict.
        hp = self._config.grid_search.hyperparameters  # type: ignore[union-attr]
        if hp:
            keys = list(hp.keys())
            value_lists = [hp[k] for k in keys]
            combos = list(itertools.product(*value_lists))
            best_combo = combos[best_idx]
            best_overrides = dict(zip(keys, best_combo, strict=True))
        else:
            best_overrides = {}

        best_raw: dict[str, Any] = self._config.model_dump(mode="json")
        best_raw["training"]["seed"] = self._config.training.seed + best_idx
        for dot_key, val in best_overrides.items():
            _set_nested(best_raw, dot_key, val)

        yaml_path = out_dir / "best_config.yaml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.dump(best_raw, f, default_flow_style=False, allow_unicode=True)


__all__ = ["GridSearchBlock"]
