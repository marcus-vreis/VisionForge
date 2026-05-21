from __future__ import annotations

import csv
import gc
import json
import time
from typing import Any

import torch
from loguru import logger

from visionforge.blocks.base import ExperimentBlock
from visionforge.blocks.classification import ClassificationBlock
from visionforge.utils.config import ExperimentConfig


class ModelComparisonBlock(ExperimentBlock):
    """Train N architectures on the same dataset and rank them by a chosen metric."""

    def setup(self, config: ExperimentConfig) -> None:
        """Validate config and initialise trial state.

        Raises:
            ValueError: if model_comparison config is missing.
        """
        if config.model_comparison is None:
            raise ValueError(
                "ModelComparisonBlock requires model_comparison to be set in ExperimentConfig."
            )
        self._config = config
        # Populated by run(); sorted descending by metric after all trials complete.
        self._trials: list[dict[str, Any]] = []

    def run(self) -> None:
        """Execute one ClassificationBlock per architecture and collect metrics."""
        mc = self._config.model_comparison
        assert mc is not None

        raw_base: dict[str, Any] = self._config.model_dump(mode="json")
        raw_base["block"] = "classification"
        raw_base["model_comparison"] = None

        unsorted: list[dict[str, Any]] = []

        for arch in mc.model_names:
            trial_record: dict[str, Any] = {
                "model_arch": arch,
                "status": "failed",
                "error": "",
                "accuracy": None,
                "f1": None,
                "auc_roc": None,
                "training_time_s": None,
            }

            block = ClassificationBlock()
            try:
                trial_raw = dict(raw_base)
                trial_raw["model"] = dict(raw_base["model"])
                trial_raw["model"]["name"] = arch

                trial_config = ExperimentConfig.model_validate(trial_raw)
                block.setup(trial_config)

                t0 = time.monotonic()
                block.run()
                elapsed = time.monotonic() - t0

                report = block.report()
                eval_metrics = report.get("eval", {})

                trial_record["status"] = "success"
                trial_record["accuracy"] = eval_metrics.get("accuracy")
                trial_record["f1"] = eval_metrics.get("f1")
                trial_record["auc_roc"] = eval_metrics.get("auc_roc")
                trial_record["training_time_s"] = elapsed

                logger.info(
                    "ModelComparison: {} succeeded — accuracy={} f1={} auc_roc={}",
                    arch,
                    trial_record["accuracy"],
                    trial_record["f1"],
                    trial_record["auc_roc"],
                )

            except Exception as exc:  # noqa: BLE001
                trial_record["error"] = str(exc)
                logger.warning("ModelComparison: {} failed — {}", arch, exc)

            finally:
                del block
                gc.collect()
                torch.cuda.empty_cache()

            unsorted.append(trial_record)

        # Sort successful trials by the chosen metric descending; failures go last.
        metric = mc.metric
        successful = [t for t in unsorted if t["status"] == "success"]
        failed = [t for t in unsorted if t["status"] != "success"]
        successful.sort(key=lambda t: t[metric] or 0.0, reverse=True)

        self._trials = successful + failed
        self._write_artifacts()

    def report(self) -> dict[str, Any]:
        """Return top-3 architectures plus total/failed counts.

        Raises:
            RuntimeError: if all architectures failed.
        """
        successful = [t for t in self._trials if t["status"] == "success"]
        if not successful:
            raise RuntimeError(
                "ModelComparisonBlock: all architectures failed — no ranking available."
            )

        return {
            "top_3": successful[:3],
            "total_ran": len(self._trials),
            "failed_count": len(self._trials) - len(successful),
        }

    # ── private ───────────────────────────────────────────────────────────────

    def _write_artifacts(self) -> None:
        """Write comparison_summary.json and ranking.csv to reports_dir / name."""
        out_dir = self._config.output.reports_dir / self._config.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Full dump — all trials including failures, preserving insertion order.
        (out_dir / "comparison_summary.json").write_text(
            json.dumps(self._trials, indent=2), encoding="utf-8"
        )

        successful = [t for t in self._trials if t["status"] == "success"]
        csv_path = out_dir / "ranking.csv"
        fieldnames = [
            "rank",
            "model_arch",
            "accuracy",
            "f1",
            "auc_roc",
            "training_time_s",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rank, trial in enumerate(successful, start=1):
                writer.writerow(
                    {
                        "rank": rank,
                        "model_arch": trial["model_arch"],
                        "accuracy": trial["accuracy"],
                        "f1": trial["f1"],
                        "auc_roc": trial["auc_roc"],
                        "training_time_s": trial["training_time_s"],
                    }
                )


__all__ = ["ModelComparisonBlock"]
