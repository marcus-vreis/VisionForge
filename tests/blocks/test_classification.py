import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from visionforge.blocks.classification import ClassificationBlock
from visionforge.utils.config import ExperimentConfig


class TinyBinaryModel(nn.Module):
    """Minimal model for integration tests — avoids loading a full ResNet."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Minimal ImageFolder structure for classification block tests."""
    from PIL import Image

    for split in ["train", "val", "test"]:
        for cls in ["class_a", "class_b"]:
            folder = tmp_path / split / cls
            folder.mkdir(parents=True)
            img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
            img.save(folder / "image.png")
    return tmp_path


@pytest.fixture
def train_config(dataset_root: Path, tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "block_test",
            "task": "binary",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {
                "learning_rate": 0.1,
                "epochs": 2,
                "batch_size": 2,
                "early_stopping_patience": 2,
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
            "classification": {"mode": "train"},
        }
    )


class TestClassificationBlock:
    def test_setup_and_report_before_run(self, train_config: ExperimentConfig) -> None:
        """report() before run() must return an empty dict."""
        block = ClassificationBlock()
        block.setup(train_config)
        assert block.report() == {}

    @patch(
        "visionforge.blocks.classification.ModelFactory.create",
        return_value=TinyBinaryModel(),
    )
    def test_train_mode_completes(
        self, _mock: Any, train_config: ExperimentConfig
    ) -> None:
        """run() in train mode must complete without raising."""
        block = ClassificationBlock()
        block.setup(train_config)
        block.run()

    @patch(
        "visionforge.blocks.classification.ModelFactory.create",
        return_value=TinyBinaryModel(),
    )
    def test_train_mode_writes_run_json(
        self, _mock: Any, train_config: ExperimentConfig
    ) -> None:
        """run() in train mode must update run.json with test metrics."""
        block = ClassificationBlock()
        block.setup(train_config)
        block.run()

        run_files = list(
            (train_config.output.models_dir / train_config.name).glob("*/run.json")
        )
        assert len(run_files) == 1

        data: dict[str, Any] = json.loads(run_files[0].read_text(encoding="utf-8"))
        assert "test_accuracy" in data["metrics"]
        assert "test_f1" in data["metrics"]
        # Block now renders the full plot set: loss, accuracy, confusion matrix
        # (raw + normalized), plus ROC and PR when both classes are present.
        # Minimum 4 (no ROC/PR) when only one class survives; up to 6 when both
        # appear. The fake binary fixture has both, so 6 is the expected count.
        graphics = data["artifacts"]["graphics"]
        assert len(graphics) >= 4
        filenames = [Path(p).name for p in graphics]
        assert "loss.png" in filenames
        assert "accuracy.png" in filenames
        assert "confusion_matrix.png" in filenames

    @patch(
        "visionforge.blocks.classification.ModelFactory.create",
        return_value=TinyBinaryModel(),
    )
    def test_loss_and_cm_pngs_created(
        self, _mock: Any, train_config: ExperimentConfig
    ) -> None:
        """run() in train mode must create loss.png and confusion_matrix.png."""
        block = ClassificationBlock()
        block.setup(train_config)
        block.run()

        run_dirs = list((train_config.output.models_dir / train_config.name).glob("*/"))
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "loss.png").exists()
        assert (run_dirs[0] / "confusion_matrix.png").exists()

    @patch(
        "visionforge.blocks.classification.ModelFactory.create",
        return_value=TinyBinaryModel(),
    )
    def test_report_contains_train_and_eval(
        self, _mock: Any, train_config: ExperimentConfig
    ) -> None:
        """report() after run() must contain both train and eval sections."""
        block = ClassificationBlock()
        block.setup(train_config)
        block.run()
        report = block.report()

        assert "train" in report
        assert "eval" in report
        assert "accuracy" in report["eval"]

    def test_infer_mode_raises_not_implemented(
        self, train_config: ExperimentConfig
    ) -> None:
        """run() with mode='infer' must raise NotImplementedError."""
        config = ExperimentConfig.model_validate(
            {
                **train_config.model_dump(mode="json"),
                "classification": {"mode": "infer"},
            }
        )
        block = ClassificationBlock()
        block.setup(config)
        with pytest.raises(NotImplementedError):
            block.run()

    def test_evaluate_mode_requires_checkpoint(
        self, train_config: ExperimentConfig
    ) -> None:
        """run() with mode='evaluate' and no checkpoint_path must raise ValueError."""
        config = ExperimentConfig.model_validate(
            {
                **train_config.model_dump(mode="json"),
                "classification": {"mode": "evaluate", "checkpoint_path": None},
            }
        )
        block = ClassificationBlock()
        block.setup(config)
        with pytest.raises(ValueError, match="checkpoint_path"):
            block.run()

    @patch(
        "visionforge.blocks.classification.ModelFactory.create",
        return_value=TinyBinaryModel(),
    )
    def test_evaluate_mode_with_checkpoint(
        self, _mock: Any, train_config: ExperimentConfig
    ) -> None:
        """run() in evaluate mode with a valid checkpoint must complete."""
        # First, train to produce a checkpoint.
        train_block = ClassificationBlock()
        train_block.setup(train_config)
        train_block.run()

        run_dirs = list(
            (train_config.output.models_dir / train_config.name).glob(
                "*/best_model.pth"
            )
        )
        assert len(run_dirs) == 1
        checkpoint = run_dirs[0]

        eval_config = ExperimentConfig.model_validate(
            {
                **train_config.model_dump(mode="json"),
                "classification": {
                    "mode": "evaluate",
                    "checkpoint_path": str(checkpoint),
                },
            }
        )
        eval_block = ClassificationBlock()
        eval_block.setup(eval_config)
        eval_block.run()

        report = eval_block.report()
        assert "eval" in report
        assert "train" not in report


@pytest.fixture
def wide_dataset_root(tmp_path: Path) -> Path:
    """ImageFolder with a test split big enough for a bootstrap interval.

    The CI floor is 20 samples, and the two-image fixture above sits under it on
    purpose — both sides of that boundary need a test.
    """
    from PIL import Image

    rng = np.random.default_rng(0)
    counts = {"train": 8, "val": 8, "test": 16}  # 16 per class -> 32 test images
    for split, per_class in counts.items():
        for cls in ["class_a", "class_b"]:
            folder = tmp_path / split / cls
            folder.mkdir(parents=True)
            for i in range(per_class):
                pixels = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"image_{i}.png")
    return tmp_path


@pytest.fixture
def wide_config(wide_dataset_root: Path, tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "ci_block_test",
            "task": "binary",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {
                "learning_rate": 0.1,
                "epochs": 1,
                "batch_size": 4,
                "seed": 7,
            },
            "data": {
                "base_dir": str(wide_dataset_root),
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
            "classification": {"mode": "train"},
        }
    )


class TestBootstrapConfidenceIntervals:
    """ADR-074: every classification test metric ships an interval."""

    @patch(
        "visionforge.blocks.classification.ModelFactory.create",
        return_value=TinyBinaryModel(),
    )
    def test_run_json_carries_metric_cis(
        self, _mock: Any, wide_config: ExperimentConfig
    ) -> None:
        block = ClassificationBlock()
        block.setup(wide_config)
        block.run()

        run_files = list(
            (wide_config.output.models_dir / wide_config.name).glob("*/run.json")
        )
        data: dict[str, Any] = json.loads(run_files[0].read_text(encoding="utf-8"))

        cis = data["metric_cis"]
        assert {"accuracy", "f1", "precision", "recall"} <= set(cis)
        accuracy = cis["accuracy"]
        assert accuracy["n_samples"] == 32
        assert (
            accuracy["ci_low"]
            <= data["metrics"]["test_accuracy"]
            <= accuracy["ci_high"]
        )

    @patch(
        "visionforge.blocks.classification.ModelFactory.create",
        return_value=TinyBinaryModel(),
    )
    def test_metrics_block_stays_a_flat_number_map(
        self, _mock: Any, wide_config: ExperimentConfig
    ) -> None:
        """The intervals are a sibling of metrics, not nested inside it.

        Several readers (the history projection, the markdown table) treat
        metrics as name -> number; a dict value there would be dropped or
        printed raw.
        """
        block = ClassificationBlock()
        block.setup(wide_config)
        block.run()

        run_files = list(
            (wide_config.output.models_dir / wide_config.name).glob("*/run.json")
        )
        data: dict[str, Any] = json.loads(run_files[0].read_text(encoding="utf-8"))

        assert all(not isinstance(v, dict) for v in data["metrics"].values())

    @patch(
        "visionforge.blocks.classification.ModelFactory.create",
        return_value=TinyBinaryModel(),
    )
    def test_report_exposes_the_intervals(
        self, _mock: Any, wide_config: ExperimentConfig
    ) -> None:
        block = ClassificationBlock()
        block.setup(wide_config)
        block.run()

        intervals = block.report()["eval"]["confidence_intervals"]

        assert intervals["accuracy"]["confidence"] == 0.95
        assert intervals["accuracy"]["n_resamples"] == 1000

    @patch(
        "visionforge.blocks.classification.ModelFactory.create",
        return_value=TinyBinaryModel(),
    )
    def test_tiny_test_split_gets_no_interval(
        self, _mock: Any, train_config: ExperimentConfig
    ) -> None:
        """Two test images cannot support an interval, so none is written."""
        block = ClassificationBlock()
        block.setup(train_config)
        block.run()

        run_files = list(
            (train_config.output.models_dir / train_config.name).glob("*/run.json")
        )
        data: dict[str, Any] = json.loads(run_files[0].read_text(encoding="utf-8"))

        assert "metric_cis" not in data
        assert "confidence_intervals" not in block.report()["eval"]

    @patch(
        "visionforge.blocks.classification.ModelFactory.create",
        return_value=TinyBinaryModel(),
    )
    def test_seed_makes_the_interval_reproducible(
        self, _mock: Any, wide_config: ExperimentConfig
    ) -> None:
        """The same run re-evaluated must not report a different interval."""
        first = ClassificationBlock()
        first.setup(wide_config)
        first.run()

        checkpoint = next(
            (wide_config.output.models_dir / wide_config.name).glob("*/best_model.pth")
        )
        eval_config = ExperimentConfig.model_validate(
            {
                **wide_config.model_dump(mode="json"),
                "classification": {
                    "mode": "evaluate",
                    "checkpoint_path": str(checkpoint),
                },
            }
        )
        second = ClassificationBlock()
        second.setup(eval_config)
        second.run()

        assert (
            second.report()["eval"]["confidence_intervals"]
            == first.report()["eval"]["confidence_intervals"]
        )
