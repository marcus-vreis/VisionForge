from __future__ import annotations

import csv
import math
import random
from typing import Any, Literal

import torch
import yaml
from loguru import logger
from pydantic import BaseModel, Field

from visionforge.blocks.base import ExperimentBlock
from visionforge.blocks.classification import ClassificationBlock
from visionforge.utils.config import ExperimentConfig

# ── search param types (private, flat within this module) ─────────────────────


class _UniformParam(BaseModel):
    """Sample a float uniformly from [low, high]."""

    type: Literal["uniform"]
    low: float
    high: float

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)


class _LogUniformParam(BaseModel):
    """Sample a float log-uniformly from [low, high] (both must be > 0)."""

    type: Literal["log_uniform"]
    low: float = Field(gt=0.0)
    high: float = Field(gt=0.0)

    def sample(self, rng: random.Random) -> float:
        return math.exp(rng.uniform(math.log(self.low), math.log(self.high)))


class _ChoiceParam(BaseModel):
    """Sample one element uniformly from a list of options."""

    type: Literal["choice"]
    options: list[Any] = Field(min_length=1)

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.options)


_KNOWN_PARAM_TYPES = ("uniform", "log_uniform", "choice")


def _parse_param(
    name: str, raw: Any
) -> _UniformParam | _LogUniformParam | _ChoiceParam:
    """Parse a raw search-space dict into a typed param object.

    Raises:
        ValueError: if 'type' is missing or unknown.
    """
    if not isinstance(raw, dict) or "type" not in raw:
        raise ValueError(
            f"Search param '{name}' must be a dict with a 'type' key, got: {raw!r}"
        )
    param_type = raw["type"]
    if param_type == "uniform":
        return _UniformParam.model_validate(raw)
    elif param_type == "log_uniform":
        return _LogUniformParam.model_validate(raw)
    elif param_type == "choice":
        return _ChoiceParam.model_validate(raw)
    else:
        raise ValueError(
            f"Unknown param type '{param_type}' for '{name}'. "
            f"Must be one of: {list(_KNOWN_PARAM_TYPES)}"
        )


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
    base_raw: dict[str, Any], search_space: dict[str, Any]
) -> None:
    """Raise ValueError for any dot-key that doesn't exist in the base config dict."""
    for dot_key in search_space:
        probe = dict(base_raw.items())
        _set_nested(probe, dot_key, None)


class RandomSearchBlock(ExperimentBlock):
    """Random hyperparameter search: samples n_trials configs and ranks them by val_loss."""

    def setup(self, config: ExperimentConfig) -> None:
        """Validate the search space against the base config and store state.

        Raises:
            ValueError: if random_search config is missing or any dot-key is unknown.
        """
        if config.random_search is None:
            raise ValueError(
                "RandomSearchBlock requires random_search to be set in ExperimentConfig."
            )

        rs = config.random_search
        base_raw: dict[str, Any] = config.model_dump(mode="json")

        # Parse and validate all param definitions up front.
        self._params: dict[str, _UniformParam | _LogUniformParam | _ChoiceParam] = {
            name: _parse_param(name, raw) for name, raw in rs.search_space.items()
        }
        _validate_search_space(base_raw, rs.search_space)

        self._config = config
        self._trials: list[dict[str, Any]] = []

    def run(self) -> None:
        """Execute n_trials random samples from the search space."""
        rs = self._config.random_search
        assert rs is not None, "random_search must be set on config"
        rng = random.Random(rs.seed)
        base_seed = self._config.training.seed
        n_trials = rs.n_trials

        for trial_idx in range(n_trials):
            trial_seed = base_seed + trial_idx
            trial_overrides = {
                name: param.sample(rng) for name, param in self._params.items()
            }

            base_raw: dict[str, Any] = self._config.model_dump(mode="json")
            base_raw["training"]["seed"] = trial_seed
            for dot_key, val in trial_overrides.items():
                _set_nested(base_raw, dot_key, val)

            trial_record: dict[str, Any] = {
                "trial_index": trial_idx,
                "trial_seed": trial_seed,
                **trial_overrides,
                "status": "failed",
                "error": "",
                "best_val_loss": None,
                "test_accuracy": None,
                "test_f1": None,
            }

            inner_block = ClassificationBlock()
            try:
                trial_config = ExperimentConfig.model_validate(base_raw)
                inner_block.setup(trial_config)
                inner_block.run()
                report = inner_block.report()

                trial_record["status"] = "success"
                trial_record["best_val_loss"] = report.get("train", {}).get(
                    "best_val_loss"
                )
                trial_record["test_accuracy"] = report.get("eval", {}).get("accuracy")
                trial_record["test_f1"] = report.get("eval", {}).get("f1")

                logger.info(
                    "Trial {}/{} succeeded: val_loss={} accuracy={}",
                    trial_idx + 1,
                    n_trials,
                    trial_record["best_val_loss"],
                    trial_record["test_accuracy"],
                )

            except Exception as exc:  # noqa: BLE001
                trial_record["error"] = str(exc)
                logger.warning(
                    "Trial {}/{} failed (params={}): {}",
                    trial_idx + 1,
                    n_trials,
                    trial_overrides,
                    exc,
                )

            finally:
                del inner_block
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
                "RandomSearchBlock: all trials failed — no best trial available."
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
        """Write random_search_summary.csv and best_config.yaml to reports_dir."""
        out_dir = self._config.output.reports_dir / self._config.name
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "random_search_summary.csv"
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

        rng = random.Random(self._config.random_search.seed)  # type: ignore[union-attr]
        best_overrides: dict[str, Any] = {}
        for i in range(best_idx + 1):
            sample = {name: param.sample(rng) for name, param in self._params.items()}
            if i == best_idx:
                best_overrides = sample

        best_raw: dict[str, Any] = self._config.model_dump(mode="json")
        best_raw["training"]["seed"] = self._config.training.seed + best_idx
        for dot_key, val in best_overrides.items():
            _set_nested(best_raw, dot_key, val)

        yaml_path = out_dir / "best_config.yaml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.dump(best_raw, f, default_flow_style=False, allow_unicode=True)


__all__ = ["RandomSearchBlock"]
