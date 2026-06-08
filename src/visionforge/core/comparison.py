"""Task-agnostic model comparison over the `TaskRunner` handle (ADR-041).

`GenericComparisonRunner` runs N configs through any `TaskRunner` adapter and
ranks them by a chosen metric, writing the same ``comparison_summary.json`` +
``ranking.csv`` artifacts the classification `ModelComparisonBlock` produces. It
knows nothing about a specific task — the metric column set is injected and the
ranking metric is read from the caller — so regression, segmentation (and later
tasks) reuse it without duplicating the orchestration logic.
"""

from __future__ import annotations

import csv
import gc
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from visionforge.core.task_runner import TaskRunner


@dataclass
class ComparisonReport:
    """Ranked outcome of a comparison run (successful trials first, failures last)."""

    trials: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Return top-3 architectures plus total/failed counts.

        Raises:
            RuntimeError: if every architecture failed (no ranking available).
        """
        successful = [t for t in self.trials if t["status"] == "success"]
        if not successful:
            raise RuntimeError(
                "GenericComparisonRunner: all architectures failed — no ranking available."
            )
        return {
            "top_3": successful[:3],
            "total_ran": len(self.trials),
            "failed_count": len(self.trials) - len(successful),
        }


class GenericComparisonRunner:
    """Rank N architectures by running each through a `TaskRunner` and sorting.

    The ranking metric must be one of ``metric_names`` and is assumed to be
    higher-is-better (each task picks metrics where that holds — e.g. r2, miou).
    GPU cleanup between trials lives here, not in the adapter, so the adapters
    stay free of torch (matching `ClassificationRunner`'s contract).
    """

    def __init__(self, runner: TaskRunner, metric_names: Sequence[str]) -> None:
        self._runner = runner
        self._metric_names = list(metric_names)

    def compare(
        self,
        trials: Sequence[tuple[str, Any]],
        rank_by: str,
        out_dir: Path,
    ) -> ComparisonReport:
        """Run each (arch_name, config) trial, rank by ``rank_by``, write artifacts.

        Raises:
            ValueError: if ``rank_by`` is not one of the configured metric names.
        """
        if rank_by not in self._metric_names:
            raise ValueError(
                f"rank_by '{rank_by}' must be one of {self._metric_names}."
            )

        unsorted: list[dict[str, Any]] = []
        for arch, cfg in trials:
            record: dict[str, Any] = {
                "model_arch": arch,
                "status": "failed",
                "error": "",
                **dict.fromkeys(self._metric_names),
                "training_time_s": None,
            }
            try:
                result = self._runner.run(cfg)
                if result.status == "success":
                    metrics = self._runner.metrics(result)
                    record["status"] = "success"
                    for m in self._metric_names:
                        record[m] = metrics.get(m)
                    record["training_time_s"] = result.training_time_s
                    logger.info("Comparison: {} succeeded — {}", arch, metrics)
                else:
                    record["error"] = result.error
                    logger.warning("Comparison: {} failed — {}", arch, result.error)
            except Exception as exc:  # noqa: BLE001
                record["error"] = str(exc)
                logger.warning("Comparison: {} failed — {}", arch, exc)
            finally:
                # Release references the runner/block held before the next arch
                # loads its weights; avoids OOM on VRAM-constrained GPUs.
                gc.collect()
                torch.cuda.empty_cache()
            unsorted.append(record)

        successful = [t for t in unsorted if t["status"] == "success"]
        failed = [t for t in unsorted if t["status"] != "success"]
        successful.sort(key=lambda t: t[rank_by] or 0.0, reverse=True)
        ranked = successful + failed

        self._write_artifacts(ranked, out_dir)
        return ComparisonReport(trials=ranked)

    def _write_artifacts(self, trials: list[dict[str, Any]], out_dir: Path) -> None:
        """Write comparison_summary.json (all trials) and ranking.csv (successful)."""
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "comparison_summary.json").write_text(
            json.dumps(trials, indent=2), encoding="utf-8"
        )

        successful = [t for t in trials if t["status"] == "success"]
        fieldnames = ["rank", "model_arch", *self._metric_names, "training_time_s"]
        with (out_dir / "ranking.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rank, trial in enumerate(successful, start=1):
                row: dict[str, Any] = {
                    "rank": rank,
                    "model_arch": trial["model_arch"],
                    "training_time_s": trial["training_time_s"],
                }
                for m in self._metric_names:
                    row[m] = trial[m]
                writer.writerow(row)


__all__ = ["ComparisonReport", "GenericComparisonRunner"]
