from __future__ import annotations

import math
import random
from typing import Any, Literal

from pydantic import BaseModel, Field

from visionforge.blocks._search_utils import (
    best_trial,
    run_trial,
    validate_dot_keys,
    write_best_config_yaml,
    write_trials_csv,
)
from visionforge.blocks.base import ExperimentBlock
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

        self._params: dict[str, _UniformParam | _LogUniformParam | _ChoiceParam] = {
            name: _parse_param(name, raw) for name, raw in rs.search_space.items()
        }
        validate_dot_keys(base_raw, rs.search_space)

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

            run_trial(
                self._config,
                trial_idx,
                n_trials,
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
                "RandomSearchBlock: all trials failed — no best trial available."
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
        """Write random_search_summary.csv and best_config.yaml to reports_dir."""
        out_dir = self._config.output.reports_dir / self._config.name
        out_dir.mkdir(parents=True, exist_ok=True)

        write_trials_csv(out_dir / "random_search_summary.csv", self._trials)

        best = best_trial(self._trials)
        if best is None:
            return

        best_idx = best["trial_index"]

        # Replay the RNG to recover the best trial's sampled overrides.
        rng = random.Random(self._config.random_search.seed)  # type: ignore[union-attr]
        best_overrides: dict[str, Any] = {}
        for i in range(best_idx + 1):
            sample = {name: param.sample(rng) for name, param in self._params.items()}
            if i == best_idx:
                best_overrides = sample

        write_best_config_yaml(
            out_dir / "best_config.yaml",
            self._config,
            best_idx,
            best_overrides,
        )


__all__ = ["RandomSearchBlock"]
