"""Task-agnostic model comparison over the `TaskRunner` handle (ADR-041, ADR-044).

Trains the same dataset under N architectures and ranks them by a chosen metric.
It knows nothing about a task's internals — only the uniform handle — so the same
function serves classification, regression, segmentation, and detection. Per-trial
config is built by overriding `model.name`, which every task config shares.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from visionforge.core.task_runner import RunResult, TaskRunner

try:  # torch is the heavy hardware extra; the cache flush is best-effort.
    import torch
except ImportError:  # pragma: no cover - torch is always present in practice
    torch = None  # type: ignore[assignment]


@dataclass
class ComparisonTrial:
    """One architecture's result inside a model comparison."""

    model_arch: str
    status: str = "failed"
    metrics: dict[str, float] = field(default_factory=dict)
    training_time_s: float | None = None
    error: str = ""


def run_model_comparison(
    runner: TaskRunner,
    base_config_dict: dict[str, Any],
    model_names: list[str],
    metric: str,
) -> list[ComparisonTrial]:
    """Train each architecture via ``runner`` and return trials ranked by ``metric``.

    Successful trials come first, sorted by ``metric`` descending (higher is
    better — the convention for every task's primary metric: accuracy, r2, miou,
    map50); failed trials keep their order at the end. GPU memory is released
    between architectures so a long sweep doesn't accumulate VRAM.
    """
    trials: list[ComparisonTrial] = []

    for arch in model_names:
        trial = ComparisonTrial(model_arch=arch)
        try:
            trial_dict = {
                **base_config_dict,
                "model": {**base_config_dict.get("model", {}), "name": arch},
            }
            cfg = runner.config_type.model_validate(trial_dict)
            result: RunResult = runner.run(cfg)
            trial.status = result.status
            trial.metrics = runner.metrics(result)
            trial.training_time_s = result.training_time_s
            trial.error = result.error
            if result.status == "success":
                logger.info(
                    "Comparison: {} succeeded — {}={}",
                    arch,
                    metric,
                    trial.metrics.get(metric),
                )
            else:
                logger.warning("Comparison: {} failed — {}", arch, result.error)
        except Exception as exc:  # noqa: BLE001
            trial.error = str(exc)
            logger.warning("Comparison: {} failed — {}", arch, exc)
        finally:
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        trials.append(trial)

    successful = [t for t in trials if t.status == "success"]
    failed = [t for t in trials if t.status != "success"]
    successful.sort(key=lambda t: t.metrics.get(metric) or 0.0, reverse=True)
    return successful + failed


__all__ = ["ComparisonTrial", "run_model_comparison"]
