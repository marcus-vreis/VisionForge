from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from visionforge.utils.config import ExperimentConfig

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Minimal ImageFolder structure for model-comparison block tests."""
    for split in ["train", "val", "test"]:
        for cls in ["class_a", "class_b"]:
            folder = tmp_path / split / cls
            folder.mkdir(parents=True)
            img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
            img.save(folder / "image.png")
    return tmp_path


def _base_raw(tmp_path: Path, dataset_root: Path) -> dict[str, Any]:
    return {
        "name": "mc_test",
        "task": "binary",
        "block": "model_comparison",
        "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
        "training": {
            "learning_rate": 0.001,
            "epochs": 1,
            "batch_size": 2,
            "early_stopping_patience": 1,
            "seed": 0,
        },
        "data": {
            "base_dir": str(dataset_root),
            "num_workers": 0,
            "pin_memory": False,
            "transforms": {"image_size": 32},
        },
        "output": {
            "models_dir": str(tmp_path / "models"),
            "graphics_dir": str(tmp_path / "graphics"),
            "logs_dir": str(tmp_path / "logs"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "model_comparison": {
            "model_names": ["resnet18", "resnet34"],
            "metric": "f1",
        },
    }


@pytest.fixture
def mc_config(tmp_path: Path, dataset_root: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(_base_raw(tmp_path, dataset_root))


def _mock_block_report(accuracy: float, f1: float, auc_roc: float) -> dict[str, Any]:
    return {
        "train": {"best_epoch": 1, "best_val_loss": 0.4, "total_epochs": 1},
        "eval": {
            "accuracy": accuracy,
            "f1": f1,
            "precision": 0.8,
            "recall": 0.7,
            "auc_roc": auc_roc,
        },
    }


# ── config validation ─────────────────────────────────────────────────────────


class TestModelComparisonConfig:
    def test_valid_config_parses(self, tmp_path: Path, dataset_root: Path) -> None:
        """ModelComparisonConfig with valid model_names and metric must parse."""
        config = ExperimentConfig.model_validate(_base_raw(tmp_path, dataset_root))
        assert config.model_comparison is not None
        assert config.model_comparison.model_names == ["resnet18", "resnet34"]
        assert config.model_comparison.metric == "f1"

    def test_auc_roc_rejected_for_multiclass(
        self, tmp_path: Path, dataset_root: Path
    ) -> None:
        """model_validator must reject auc_roc metric when task is multiclass."""
        raw = _base_raw(tmp_path, dataset_root)
        raw["task"] = "multiclass"
        raw["model"]["num_classes"] = 3
        raw["model_comparison"]["metric"] = "auc_roc"
        with pytest.raises(Exception, match="auc_roc"):
            ExperimentConfig.model_validate(raw)

    def test_auc_roc_allowed_for_binary(
        self, tmp_path: Path, dataset_root: Path
    ) -> None:
        """auc_roc metric must be accepted when task is binary."""
        raw = _base_raw(tmp_path, dataset_root)
        raw["model_comparison"]["metric"] = "auc_roc"
        config = ExperimentConfig.model_validate(raw)
        assert config.model_comparison is not None
        assert config.model_comparison.metric == "auc_roc"

    def test_model_names_requires_min_two(
        self, tmp_path: Path, dataset_root: Path
    ) -> None:
        """model_names with fewer than 2 entries must raise."""
        raw = _base_raw(tmp_path, dataset_root)
        raw["model_comparison"]["model_names"] = ["resnet18"]
        with pytest.raises(Exception, match="at least 2"):
            ExperimentConfig.model_validate(raw)

    def test_default_metric_is_f1(self, tmp_path: Path, dataset_root: Path) -> None:
        """Default metric must be 'f1' when not specified."""
        raw = _base_raw(tmp_path, dataset_root)
        del raw["model_comparison"]["metric"]
        config = ExperimentConfig.model_validate(raw)
        assert config.model_comparison is not None
        assert config.model_comparison.metric == "f1"


# ── setup ─────────────────────────────────────────────────────────────────────


class TestSetup:
    def test_raises_without_model_comparison_config(
        self, tmp_path: Path, dataset_root: Path
    ) -> None:
        """setup() must raise ValueError when model_comparison is None."""
        from visionforge.blocks.model_comparison import ModelComparisonBlock

        raw = _base_raw(tmp_path, dataset_root)
        raw["block"] = "classification"
        raw["model_comparison"] = None
        config = ExperimentConfig.model_validate(raw)
        block = ModelComparisonBlock()
        with pytest.raises(ValueError, match="model_comparison"):
            block.setup(config)

    def test_valid_setup_does_not_raise(self, mc_config: ExperimentConfig) -> None:
        """setup() with valid config must not raise."""
        from visionforge.blocks.model_comparison import ModelComparisonBlock

        block = ModelComparisonBlock()
        block.setup(mc_config)


# ── happy path ────────────────────────────────────────────────────────────────


class TestHappyPath:
    def _run_mocked(
        self,
        mc_config: ExperimentConfig,
        reports: list[dict[str, Any]],
    ) -> Any:
        """Run block with ClassificationBlock.run() and report() mocked per trial."""
        from visionforge.blocks.model_comparison import ModelComparisonBlock

        call_count: list[int] = [0]

        def mock_run(self: Any) -> None:  # noqa: ANN001
            pass

        def mock_report(self: Any) -> dict[str, Any]:  # noqa: ANN001
            idx = call_count[0]
            call_count[0] += 1
            return reports[idx]

        block = ModelComparisonBlock()
        block.setup(mc_config)
        with (
            patch(
                "visionforge.blocks.model_comparison.ClassificationBlock.run", mock_run
            ),
            patch(
                "visionforge.blocks.model_comparison.ClassificationBlock.report",
                mock_report,
            ),
        ):
            block.run()
        return block

    def test_ranking_has_two_rows(self, mc_config: ExperimentConfig) -> None:
        """Ranking must contain one row per successful architecture."""
        reports = [
            _mock_block_report(0.8, 0.75, 0.85),
            _mock_block_report(0.9, 0.88, 0.92),
        ]
        block = self._run_mocked(mc_config, reports)
        successful = [t for t in block._trials if t["status"] == "success"]
        assert len(successful) == 2

    def test_top_ranked_is_best_f1(self, mc_config: ExperimentConfig) -> None:
        """First ranked trial must have the higher f1 when metric='f1'."""
        reports = [
            _mock_block_report(0.8, 0.75, 0.85),  # resnet18 — lower f1
            _mock_block_report(0.9, 0.88, 0.92),  # resnet34 — higher f1
        ]
        block = self._run_mocked(mc_config, reports)
        # _trials is ordered by rank after run()
        assert block._trials[0]["model_arch"] == "resnet34"
        assert block._trials[0]["f1"] > block._trials[1]["f1"]

    def test_top_ranked_by_accuracy(self, tmp_path: Path, dataset_root: Path) -> None:
        """When metric='accuracy', top-ranked must have the highest accuracy."""
        from visionforge.blocks.model_comparison import ModelComparisonBlock

        raw = _base_raw(tmp_path, dataset_root)
        raw["model_comparison"]["metric"] = "accuracy"
        config = ExperimentConfig.model_validate(raw)

        call_count: list[int] = [0]
        reports = [
            _mock_block_report(0.6, 0.55, 0.70),  # resnet18 — lower accuracy
            _mock_block_report(0.95, 0.90, 0.95),  # resnet34 — higher accuracy
        ]

        def mock_run(self: Any) -> None:  # noqa: ANN001
            pass

        def mock_report(self: Any) -> dict[str, Any]:  # noqa: ANN001
            idx = call_count[0]
            call_count[0] += 1
            return reports[idx]

        block = ModelComparisonBlock()
        block.setup(config)
        with (
            patch(
                "visionforge.blocks.model_comparison.ClassificationBlock.run", mock_run
            ),
            patch(
                "visionforge.blocks.model_comparison.ClassificationBlock.report",
                mock_report,
            ),
        ):
            block.run()
        assert block._trials[0]["model_arch"] == "resnet34"


# ── artifact files ────────────────────────────────────────────────────────────


class TestArtifacts:
    def _run_block(self, mc_config: ExperimentConfig) -> Any:
        from visionforge.blocks.model_comparison import ModelComparisonBlock

        call_count: list[int] = [0]
        reports = [
            _mock_block_report(0.8, 0.75, 0.85),
            _mock_block_report(0.9, 0.88, 0.92),
        ]

        def mock_run(self: Any) -> None:  # noqa: ANN001
            pass

        def mock_report(self: Any) -> dict[str, Any]:  # noqa: ANN001
            idx = call_count[0]
            call_count[0] += 1
            return reports[idx]

        block = ModelComparisonBlock()
        block.setup(mc_config)
        with (
            patch(
                "visionforge.blocks.model_comparison.ClassificationBlock.run", mock_run
            ),
            patch(
                "visionforge.blocks.model_comparison.ClassificationBlock.report",
                mock_report,
            ),
        ):
            block.run()
        return block

    def test_comparison_summary_json_written(self, mc_config: ExperimentConfig) -> None:
        """run() must write comparison_summary.json with one entry per trial."""
        self._run_block(mc_config)
        path = mc_config.output.reports_dir / mc_config.name / "comparison_summary.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        # 2 architectures → 2 entries
        assert len(data) == 2

    def test_ranking_csv_written(self, mc_config: ExperimentConfig) -> None:
        """run() must write ranking.csv with only successful rows."""
        self._run_block(mc_config)
        path = mc_config.output.reports_dir / mc_config.name / "ranking.csv"
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert "rank" in rows[0]
        assert "model_arch" in rows[0]
        assert "accuracy" in rows[0]
        assert "f1" in rows[0]
        assert "auc_roc" in rows[0]
        assert "training_time_s" in rows[0]

    def test_ranking_csv_rank_column_is_ordered(
        self, mc_config: ExperimentConfig
    ) -> None:
        """ranking.csv rank column must start at 1 and increment."""
        self._run_block(mc_config)
        path = mc_config.output.reports_dir / mc_config.name / "ranking.csv"
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ranks = [int(r["rank"]) for r in rows]
        assert ranks == list(range(1, len(rows) + 1))


# ── failure handling ──────────────────────────────────────────────────────────


class TestFailureHandling:
    def test_failed_trial_skipped_in_ranking(self, mc_config: ExperimentConfig) -> None:
        """A failing architecture must be excluded from ranking.csv but present in summary."""
        from visionforge.blocks.model_comparison import ModelComparisonBlock

        call_count: list[int] = [0]

        def mock_run(self: Any) -> None:  # noqa: ANN001
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                raise RuntimeError("injected failure for resnet18")

        def mock_report(self: Any) -> dict[str, Any]:  # noqa: ANN001
            return _mock_block_report(0.9, 0.88, 0.92)

        block = ModelComparisonBlock()
        block.setup(mc_config)
        with (
            patch(
                "visionforge.blocks.model_comparison.ClassificationBlock.run", mock_run
            ),
            patch(
                "visionforge.blocks.model_comparison.ClassificationBlock.report",
                mock_report,
            ),
        ):
            block.run()

        # One failed, one succeeded
        statuses = [t["status"] for t in block._trials]
        assert "failed" in statuses
        assert "success" in statuses

        # ranking.csv has only the successful one
        csv_path = mc_config.output.reports_dir / mc_config.name / "ranking.csv"
        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

        # summary has both
        json_path = (
            mc_config.output.reports_dir / mc_config.name / "comparison_summary.json"
        )
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert len(data) == 2

    def test_report_returns_counts(self, mc_config: ExperimentConfig) -> None:
        """report() must include total_ran and failed_count."""
        from visionforge.blocks.model_comparison import ModelComparisonBlock

        call_count: list[int] = [0]
        reports = [
            _mock_block_report(0.8, 0.75, 0.85),
            _mock_block_report(0.9, 0.88, 0.92),
        ]

        def mock_run(self: Any) -> None:  # noqa: ANN001
            pass

        def mock_report(self: Any) -> dict[str, Any]:  # noqa: ANN001
            idx = call_count[0]
            call_count[0] += 1
            return reports[idx]

        block = ModelComparisonBlock()
        block.setup(mc_config)
        with (
            patch(
                "visionforge.blocks.model_comparison.ClassificationBlock.run", mock_run
            ),
            patch(
                "visionforge.blocks.model_comparison.ClassificationBlock.report",
                mock_report,
            ),
        ):
            block.run()

        result = block.report()
        assert result["total_ran"] == 2
        assert result["failed_count"] == 0
        assert len(result["top_3"]) <= 3

    def test_report_top_3_limit(self, tmp_path: Path, dataset_root: Path) -> None:
        """report() top_3 must contain at most 3 entries even with more architectures."""
        from visionforge.blocks.model_comparison import ModelComparisonBlock

        raw = _base_raw(tmp_path, dataset_root)
        raw["model_comparison"]["model_names"] = [
            "resnet18",
            "resnet34",
            "resnet50",
            "alexnet",
        ]
        config = ExperimentConfig.model_validate(raw)

        call_count: list[int] = [0]
        metrics = [
            (0.70, 0.68, 0.80),
            (0.75, 0.73, 0.82),
            (0.80, 0.78, 0.85),
            (0.85, 0.83, 0.90),
        ]

        def mock_run(self: Any) -> None:  # noqa: ANN001
            pass

        def mock_report(self: Any) -> dict[str, Any]:  # noqa: ANN001
            idx = call_count[0]
            call_count[0] += 1
            acc, f1, auc = metrics[idx]
            return _mock_block_report(acc, f1, auc)

        block = ModelComparisonBlock()
        block.setup(config)
        with (
            patch(
                "visionforge.blocks.model_comparison.ClassificationBlock.run", mock_run
            ),
            patch(
                "visionforge.blocks.model_comparison.ClassificationBlock.report",
                mock_report,
            ),
        ):
            block.run()

        result = block.report()
        assert len(result["top_3"]) == 3

    def test_all_failed_raises_in_report(self, mc_config: ExperimentConfig) -> None:
        """report() must raise RuntimeError when all architectures failed."""
        from visionforge.blocks.model_comparison import ModelComparisonBlock

        def mock_run(self: Any) -> None:  # noqa: ANN001
            raise RuntimeError("always fails")

        block = ModelComparisonBlock()
        block.setup(mc_config)
        with patch(
            "visionforge.blocks.model_comparison.ClassificationBlock.run", mock_run
        ):
            block.run()

        with pytest.raises(RuntimeError, match="all architectures failed"):
            block.report()


# ── VRAM hygiene ──────────────────────────────────────────────────────────────


class TestVramHygiene:
    def test_empty_cache_called_between_trials(
        self, mc_config: ExperimentConfig
    ) -> None:
        """torch.cuda.empty_cache must be called once per architecture trial."""
        from visionforge.blocks.model_comparison import ModelComparisonBlock

        call_count: list[int] = [0]
        reports = [
            _mock_block_report(0.8, 0.75, 0.85),
            _mock_block_report(0.9, 0.88, 0.92),
        ]

        def mock_run(self: Any) -> None:  # noqa: ANN001
            pass

        def mock_report(self: Any) -> dict[str, Any]:  # noqa: ANN001
            idx = call_count[0]
            call_count[0] += 1
            return reports[idx]

        block = ModelComparisonBlock()
        block.setup(mc_config)
        with (
            patch(
                "visionforge.blocks.model_comparison.ClassificationBlock.run", mock_run
            ),
            patch(
                "visionforge.blocks.model_comparison.ClassificationBlock.report",
                mock_report,
            ),
            patch(
                "visionforge.blocks.model_comparison.torch.cuda.empty_cache"
            ) as mock_cache,
        ):
            block.run()

        # One call per architecture (2 total)
        assert mock_cache.call_count == 2

    def test_empty_cache_called_on_failure(self, mc_config: ExperimentConfig) -> None:
        """torch.cuda.empty_cache must be called even when a trial fails."""
        from visionforge.blocks.model_comparison import ModelComparisonBlock

        def mock_run(self: Any) -> None:  # noqa: ANN001
            raise RuntimeError("fail")

        block = ModelComparisonBlock()
        block.setup(mc_config)
        with (
            patch(
                "visionforge.blocks.model_comparison.ClassificationBlock.run", mock_run
            ),
            patch(
                "visionforge.blocks.model_comparison.torch.cuda.empty_cache"
            ) as mock_cache,
        ):
            block.run()

        assert mock_cache.call_count == 2


# ── gc.collect integration ────────────────────────────────────────────────────


class TestGcCollect:
    def test_gc_collect_called_between_trials(
        self, mc_config: ExperimentConfig
    ) -> None:
        """gc.collect must be called once per trial for explicit memory reclaim."""
        from visionforge.blocks.model_comparison import ModelComparisonBlock

        call_count: list[int] = [0]
        reports = [
            _mock_block_report(0.8, 0.75, 0.85),
            _mock_block_report(0.9, 0.88, 0.92),
        ]

        def mock_run(self: Any) -> None:  # noqa: ANN001
            pass

        def mock_report(self: Any) -> dict[str, Any]:  # noqa: ANN001
            idx = call_count[0]
            call_count[0] += 1
            return reports[idx]

        block = ModelComparisonBlock()
        block.setup(mc_config)
        with (
            patch(
                "visionforge.blocks.model_comparison.ClassificationBlock.run", mock_run
            ),
            patch(
                "visionforge.blocks.model_comparison.ClassificationBlock.report",
                mock_report,
            ),
            patch("visionforge.blocks.model_comparison.gc.collect") as mock_gc,
            patch("visionforge.blocks.model_comparison.torch.cuda.empty_cache"),
        ):
            block.run()

        assert mock_gc.call_count == 2


# ── registry ──────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_registry_discovers_model_comparison_block(self) -> None:
        """BlockRegistry must include ModelComparisonBlock after import."""
        import visionforge.blocks.model_comparison  # noqa: F401
        from visionforge.blocks.registry import BlockRegistry

        registry = BlockRegistry.discover()
        assert "ModelComparisonBlock" in registry
