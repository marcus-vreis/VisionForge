from __future__ import annotations

import itertools
from typing import Any

from visionforge.blocks._search_utils import (
    best_trial,
    run_trial,
    validate_dot_keys,
    write_best_config_yaml,
    write_trials_csv,
)
from visionforge.blocks.base import ExperimentBlock
from visionforge.utils.config import ExperimentConfig


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
        validate_dot_keys(base_raw, config.grid_search.hyperparameters)

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

            run_trial(
                self._config,
                trial_idx,
                len(combos),
                trial_seed,
                trial_overrides,
                trial_record,
            )
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

        best = best_trial(self._trials)
        assert best is not None
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

        write_trials_csv(out_dir / "grid_search_summary.csv", self._trials)

        best = best_trial(self._trials)
        if best is None:
            return

        best_idx = best["trial_index"]
        hp = self._config.grid_search.hyperparameters  # type: ignore[union-attr]
        if hp:
            keys = list(hp.keys())
            value_lists = [hp[k] for k in keys]
            combos = list(itertools.product(*value_lists))
            best_overrides = dict(zip(keys, combos[best_idx], strict=True))
        else:
            best_overrides = {}

        write_best_config_yaml(
            out_dir / "best_config.yaml",
            self._config,
            best_idx,
            best_overrides,
        )


__all__ = ["GridSearchBlock"]
