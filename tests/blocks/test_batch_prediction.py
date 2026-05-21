from __future__ import annotations

import csv
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from visionforge.utils.config import ExperimentConfig


class TinyMulticlassModel(nn.Module):
    """Minimal model for batch prediction tests — 3-class output."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))  # type: ignore[no-any-return]


class TinyBinaryModel(nn.Module):
    """Minimal model for batch prediction tests — 1-class (binary sigmoid) output."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))  # type: ignore[no-any-return]


# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def checkpoint_file(tmp_path: Path) -> Path:
    """Write a dummy .pth file so field_validator passes."""
    p = tmp_path / "model.pth"
    torch.save({}, p)
    return p


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    """Flat folder with three PNG images."""
    from PIL import Image

    d = tmp_path / "images"
    d.mkdir()
    for i in range(3):
        img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        img.save(d / f"img_{i}.png")
    return d


@pytest.fixture
def nested_image_dir(tmp_path: Path) -> Path:
    """Two-level folder: root has one image, sub/ has two more."""
    from PIL import Image

    d = tmp_path / "nested"
    d.mkdir()
    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    img.save(d / "root.png")
    sub = d / "sub"
    sub.mkdir()
    img.save(sub / "a.png")
    img.save(sub / "b.png")
    return d


@pytest.fixture
def base_config_dict(
    checkpoint_file: Path,
    image_dir: Path,
    tmp_path: Path,
) -> dict[str, Any]:
    return {
        "name": "bp_test",
        "task": "multiclass",
        "block": "batch_prediction",
        "model": {"name": "resnet18", "num_classes": 3, "pretrained": False},
        "training": {
            "learning_rate": 0.01,
            "epochs": 1,
            "batch_size": 2,
            "seed": 0,
        },
        "data": {
            "base_dir": str(tmp_path),
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
        "batch_prediction": {
            "checkpoint_path": str(checkpoint_file),
            "input_dir": str(image_dir),
            "output_csv": str(tmp_path / "predictions.csv"),
            "recursive": False,
        },
    }


@pytest.fixture
def multiclass_config(base_config_dict: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig.model_validate(base_config_dict)


@pytest.fixture
def binary_config(
    checkpoint_file: Path,
    image_dir: Path,
    tmp_path: Path,
) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "bp_binary",
            "task": "binary",
            "block": "batch_prediction",
            "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
            "training": {
                "learning_rate": 0.01,
                "epochs": 1,
                "batch_size": 2,
                "seed": 0,
            },
            "data": {
                "base_dir": str(tmp_path),
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
            "batch_prediction": {
                "checkpoint_path": str(checkpoint_file),
                "input_dir": str(image_dir),
                "output_csv": str(tmp_path / "predictions_binary.csv"),
                "recursive": False,
            },
        }
    )


# ── BatchPredictionConfig validation ──────────────────────────────────────────


class TestBatchPredictionConfig:
    def test_valid_config_builds(self, checkpoint_file: Path, image_dir: Path) -> None:
        """Valid paths must build without raising."""
        from visionforge.utils.config import BatchPredictionConfig

        cfg = BatchPredictionConfig(
            checkpoint_path=checkpoint_file,
            input_dir=image_dir,
            output_csv=Path("/tmp/out.csv"),
        )
        assert cfg.checkpoint_path == checkpoint_file

    def test_checkpoint_missing_raises(self, image_dir: Path) -> None:
        """checkpoint_path must point to an existing file."""
        from pydantic import ValidationError

        from visionforge.utils.config import BatchPredictionConfig

        with pytest.raises(ValidationError, match="checkpoint_path"):
            BatchPredictionConfig(
                checkpoint_path=Path("/nonexistent/model.pth"),
                input_dir=image_dir,
                output_csv=Path("/tmp/out.csv"),
            )

    def test_checkpoint_is_dir_raises(self, tmp_path: Path, image_dir: Path) -> None:
        """checkpoint_path must be a file, not a directory."""
        from pydantic import ValidationError

        from visionforge.utils.config import BatchPredictionConfig

        with pytest.raises(ValidationError, match="checkpoint_path"):
            BatchPredictionConfig(
                checkpoint_path=tmp_path,
                input_dir=image_dir,
                output_csv=Path("/tmp/out.csv"),
            )

    def test_input_dir_missing_raises(self, checkpoint_file: Path) -> None:
        """input_dir must point to an existing directory."""
        from pydantic import ValidationError

        from visionforge.utils.config import BatchPredictionConfig

        with pytest.raises(ValidationError, match="input_dir"):
            BatchPredictionConfig(
                checkpoint_path=checkpoint_file,
                input_dir=Path("/nonexistent/dir"),
                output_csv=Path("/tmp/out.csv"),
            )

    def test_input_dir_is_file_raises(
        self, checkpoint_file: Path, tmp_path: Path
    ) -> None:
        """input_dir must be a directory, not a file."""
        from pydantic import ValidationError

        from visionforge.utils.config import BatchPredictionConfig

        with pytest.raises(ValidationError, match="input_dir"):
            BatchPredictionConfig(
                checkpoint_path=checkpoint_file,
                input_dir=checkpoint_file,
                output_csv=Path("/tmp/out.csv"),
            )

    def test_output_csv_no_existence_check(
        self, checkpoint_file: Path, image_dir: Path
    ) -> None:
        """output_csv may not exist yet — no validator required."""
        from visionforge.utils.config import BatchPredictionConfig

        cfg = BatchPredictionConfig(
            checkpoint_path=checkpoint_file,
            input_dir=image_dir,
            output_csv=Path("/tmp/brand_new_file.csv"),
        )
        assert cfg.output_csv == Path("/tmp/brand_new_file.csv")

    def test_recursive_default_true(
        self, checkpoint_file: Path, image_dir: Path
    ) -> None:
        from visionforge.utils.config import BatchPredictionConfig

        cfg = BatchPredictionConfig(
            checkpoint_path=checkpoint_file,
            input_dir=image_dir,
            output_csv=Path("/tmp/out.csv"),
        )
        assert cfg.recursive is True

    def test_default_image_extensions(
        self, checkpoint_file: Path, image_dir: Path
    ) -> None:
        from visionforge.utils.config import BatchPredictionConfig

        cfg = BatchPredictionConfig(
            checkpoint_path=checkpoint_file,
            input_dir=image_dir,
            output_csv=Path("/tmp/out.csv"),
        )
        assert ".png" in cfg.image_extensions
        assert ".jpg" in cfg.image_extensions

    def test_class_names_defaults_none(
        self, checkpoint_file: Path, image_dir: Path
    ) -> None:
        from visionforge.utils.config import BatchPredictionConfig

        cfg = BatchPredictionConfig(
            checkpoint_path=checkpoint_file,
            input_dir=image_dir,
            output_csv=Path("/tmp/out.csv"),
        )
        assert cfg.class_names is None


class TestExperimentConfigBatchPrediction:
    def test_block_literal_accepts_batch_prediction(
        self, multiclass_config: ExperimentConfig
    ) -> None:
        assert multiclass_config.block == "batch_prediction"

    def test_batch_prediction_field_present(
        self, multiclass_config: ExperimentConfig
    ) -> None:
        assert multiclass_config.batch_prediction is not None

    def test_batch_prediction_defaults_to_none(self, tmp_path: Path) -> None:
        """batch_prediction field is optional — defaults to None."""
        # We need a valid data dir for DataConfig
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = ExperimentConfig.model_validate(
            {
                "name": "exp",
                "task": "binary",
                "block": "classification",
                "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
                "training": {"learning_rate": 0.01, "epochs": 1, "batch_size": 2},
                "data": {
                    "base_dir": str(data_dir),
                    "num_workers": 0,
                    "pin_memory": False,
                },
            }
        )
        assert config.batch_prediction is None

    def test_config_exported_in_all(self) -> None:
        import visionforge.utils.config as cfg_module

        assert "BatchPredictionConfig" in cfg_module.__all__


# ── setup ──────────────────────────────────────────────────────────────────────


class TestSetup:
    def test_setup_requires_batch_prediction_config(self, tmp_path: Path) -> None:
        """setup() must raise ValueError when batch_prediction is None."""
        from visionforge.blocks.batch_prediction import BatchPredictionBlock

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = ExperimentConfig.model_validate(
            {
                "name": "exp",
                "task": "binary",
                "block": "classification",
                "model": {"name": "resnet18", "num_classes": 1, "pretrained": False},
                "training": {"learning_rate": 0.01, "epochs": 1, "batch_size": 2},
                "data": {
                    "base_dir": str(data_dir),
                    "num_workers": 0,
                    "pin_memory": False,
                },
            }
        )
        block = BatchPredictionBlock()
        with pytest.raises(ValueError, match="batch_prediction"):
            block.setup(config)

    def test_valid_setup_does_not_raise(
        self, multiclass_config: ExperimentConfig
    ) -> None:
        from visionforge.blocks.batch_prediction import BatchPredictionBlock

        block = BatchPredictionBlock()
        with (
            patch(
                "visionforge.blocks.batch_prediction.ModelFactory.create",
                return_value=TinyMulticlassModel(),
            ),
            patch("visionforge.blocks.batch_prediction.torch.load", return_value={}),
            patch("torch.nn.Module.load_state_dict"),
        ):
            block.setup(multiclass_config)

    def test_report_before_run_returns_empty(
        self, multiclass_config: ExperimentConfig
    ) -> None:
        from visionforge.blocks.batch_prediction import BatchPredictionBlock

        block = BatchPredictionBlock()
        with (
            patch(
                "visionforge.blocks.batch_prediction.ModelFactory.create",
                return_value=TinyMulticlassModel(),
            ),
            patch("visionforge.blocks.batch_prediction.torch.load", return_value={}),
            patch("torch.nn.Module.load_state_dict"),
        ):
            block.setup(multiclass_config)
        assert block.report() == {}


# ── happy path ─────────────────────────────────────────────────────────────────


class TestHappyPath:
    def _run_block(
        self,
        config: ExperimentConfig,
        model: nn.Module,
    ) -> None:
        from visionforge.blocks.batch_prediction import BatchPredictionBlock

        block = BatchPredictionBlock()
        with (
            patch(
                "visionforge.blocks.batch_prediction.ModelFactory.create",
                return_value=model,
            ),
            patch("visionforge.blocks.batch_prediction.torch.load", return_value={}),
            patch("torch.nn.Module.load_state_dict"),
        ):
            block.setup(config)
        block.run()

    def test_csv_is_created(self, multiclass_config: ExperimentConfig) -> None:
        """run() must create the output CSV file."""
        self._run_block(multiclass_config, TinyMulticlassModel())
        assert multiclass_config.batch_prediction is not None
        assert multiclass_config.batch_prediction.output_csv.exists()

    def test_csv_has_one_row_per_image(
        self, multiclass_config: ExperimentConfig, image_dir: Path
    ) -> None:
        """CSV must contain one data row per discovered image."""
        self._run_block(multiclass_config, TinyMulticlassModel())
        assert multiclass_config.batch_prediction is not None
        rows = _read_csv_rows(multiclass_config.batch_prediction.output_csv)
        # image_dir fixture has 3 images
        assert len(rows) == 3

    def test_csv_multiclass_header(self, multiclass_config: ExperimentConfig) -> None:
        """Multiclass CSV must have prob_class_0, prob_class_1, prob_class_2 columns."""
        self._run_block(multiclass_config, TinyMulticlassModel())
        assert multiclass_config.batch_prediction is not None
        header = _read_csv_header(multiclass_config.batch_prediction.output_csv)
        assert "filename" in header
        assert "predicted_class" in header
        assert "prob_class_0" in header
        assert "prob_class_1" in header
        assert "prob_class_2" in header
        # No binary-only column
        assert "prob_positive" not in header

    def test_csv_binary_header(self, binary_config: ExperimentConfig) -> None:
        """Binary CSV must have prob_positive instead of prob_class_N columns."""
        self._run_block(binary_config, TinyBinaryModel())
        assert binary_config.batch_prediction is not None
        header = _read_csv_header(binary_config.batch_prediction.output_csv)
        assert "filename" in header
        assert "predicted_class" in header
        assert "prob_positive" in header
        assert "prob_class_0" not in header

    def test_csv_filename_column_matches_files(
        self, multiclass_config: ExperimentConfig, image_dir: Path
    ) -> None:
        """filename column values must be recognisable image file names."""
        self._run_block(multiclass_config, TinyMulticlassModel())
        assert multiclass_config.batch_prediction is not None
        rows = _read_csv_rows(multiclass_config.batch_prediction.output_csv)
        filenames = {r["filename"] for r in rows}
        expected = {str(p) for p in image_dir.iterdir() if p.suffix == ".png"}
        assert filenames == expected


# ── recursive vs flat walk ─────────────────────────────────────────────────────


class TestRecursiveWalk:
    def test_non_recursive_finds_only_root_images(
        self,
        base_config_dict: dict[str, Any],
        nested_image_dir: Path,
        tmp_path: Path,
    ) -> None:
        """recursive=False must skip images in subdirectories."""
        raw = {
            **base_config_dict,
            "batch_prediction": {
                **base_config_dict["batch_prediction"],
                "input_dir": str(nested_image_dir),
                "output_csv": str(tmp_path / "flat.csv"),
                "recursive": False,
            },
        }
        config = ExperimentConfig.model_validate(raw)

        from visionforge.blocks.batch_prediction import BatchPredictionBlock

        block = BatchPredictionBlock()
        with (
            patch(
                "visionforge.blocks.batch_prediction.ModelFactory.create",
                return_value=TinyMulticlassModel(),
            ),
            patch("visionforge.blocks.batch_prediction.torch.load", return_value={}),
            patch("torch.nn.Module.load_state_dict"),
        ):
            block.setup(config)
        block.run()

        assert config.batch_prediction is not None
        rows = _read_csv_rows(config.batch_prediction.output_csv)
        # nested_image_dir has 1 root image; sub/ has 2 more
        assert len(rows) == 1

    def test_recursive_finds_all_images(
        self,
        base_config_dict: dict[str, Any],
        nested_image_dir: Path,
        tmp_path: Path,
    ) -> None:
        """recursive=True must collect images from all subdirectories."""
        raw = {
            **base_config_dict,
            "batch_prediction": {
                **base_config_dict["batch_prediction"],
                "input_dir": str(nested_image_dir),
                "output_csv": str(tmp_path / "recursive.csv"),
                "recursive": True,
            },
        }
        config = ExperimentConfig.model_validate(raw)

        from visionforge.blocks.batch_prediction import BatchPredictionBlock

        block = BatchPredictionBlock()
        with (
            patch(
                "visionforge.blocks.batch_prediction.ModelFactory.create",
                return_value=TinyMulticlassModel(),
            ),
            patch("visionforge.blocks.batch_prediction.torch.load", return_value={}),
            patch("torch.nn.Module.load_state_dict"),
        ):
            block.setup(config)
        block.run()

        assert config.batch_prediction is not None
        rows = _read_csv_rows(config.batch_prediction.output_csv)
        assert len(rows) == 3


# ── image load failure ─────────────────────────────────────────────────────────


class TestImageLoadFailure:
    def test_corrupt_image_skipped_and_recorded(
        self,
        base_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """A corrupt image must be logged, skipped from CSV, and appear in report."""
        from PIL import Image

        bad_dir = tmp_path / "bad_images"
        bad_dir.mkdir()

        good = bad_dir / "good.png"
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(good)

        corrupt = bad_dir / "corrupt.png"
        corrupt.write_bytes(b"this is not a valid PNG")

        raw = {
            **base_config_dict,
            "batch_prediction": {
                **base_config_dict["batch_prediction"],
                "input_dir": str(bad_dir),
                "output_csv": str(tmp_path / "partial.csv"),
                "recursive": False,
            },
        }
        config = ExperimentConfig.model_validate(raw)

        from visionforge.blocks.batch_prediction import BatchPredictionBlock

        block = BatchPredictionBlock()
        with (
            patch(
                "visionforge.blocks.batch_prediction.ModelFactory.create",
                return_value=TinyMulticlassModel(),
            ),
            patch("visionforge.blocks.batch_prediction.torch.load", return_value={}),
            patch("torch.nn.Module.load_state_dict"),
        ):
            block.setup(config)
        block.run()

        # Only the good image appears in the CSV
        assert config.batch_prediction is not None
        rows = _read_csv_rows(config.batch_prediction.output_csv)
        assert len(rows) == 1

        # The corrupt file must appear in the report
        report = block.report()
        assert "failed_files" in report
        assert any("corrupt.png" in str(f) for f in report["failed_files"])

    def test_run_continues_after_failure(
        self,
        base_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """run() must not raise when some images fail to load."""
        bad_dir = tmp_path / "mixed"
        bad_dir.mkdir()

        from PIL import Image

        for i in range(2):
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(
                bad_dir / f"good_{i}.png"
            )
        (bad_dir / "bad.png").write_bytes(b"garbage")

        raw = {
            **base_config_dict,
            "batch_prediction": {
                **base_config_dict["batch_prediction"],
                "input_dir": str(bad_dir),
                "output_csv": str(tmp_path / "mixed.csv"),
                "recursive": False,
            },
        }
        config = ExperimentConfig.model_validate(raw)

        from visionforge.blocks.batch_prediction import BatchPredictionBlock

        block = BatchPredictionBlock()
        with (
            patch(
                "visionforge.blocks.batch_prediction.ModelFactory.create",
                return_value=TinyMulticlassModel(),
            ),
            patch("visionforge.blocks.batch_prediction.torch.load", return_value={}),
            patch("torch.nn.Module.load_state_dict"),
        ):
            block.setup(config)

        # Must not raise
        block.run()

        assert config.batch_prediction is not None
        rows = _read_csv_rows(config.batch_prediction.output_csv)
        assert len(rows) == 2


# ── class_names mapping ────────────────────────────────────────────────────────


class TestClassNames:
    def test_class_names_used_in_predicted_class_column(
        self,
        base_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """predicted_class values must match class_names when provided."""
        raw = {
            **base_config_dict,
            "batch_prediction": {
                **base_config_dict["batch_prediction"],
                "output_csv": str(tmp_path / "named.csv"),
                "recursive": False,
                "class_names": ["cat", "dog", "bird"],
            },
        }
        config = ExperimentConfig.model_validate(raw)

        from visionforge.blocks.batch_prediction import BatchPredictionBlock

        block = BatchPredictionBlock()
        with (
            patch(
                "visionforge.blocks.batch_prediction.ModelFactory.create",
                return_value=TinyMulticlassModel(),
            ),
            patch("visionforge.blocks.batch_prediction.torch.load", return_value={}),
            patch("torch.nn.Module.load_state_dict"),
        ):
            block.setup(config)
        block.run()

        assert config.batch_prediction is not None
        rows = _read_csv_rows(config.batch_prediction.output_csv)
        valid_names = {"cat", "dog", "bird"}
        for row in rows:
            assert row["predicted_class"] in valid_names

    def test_no_class_names_uses_integer_index(
        self, multiclass_config: ExperimentConfig
    ) -> None:
        """Without class_names, predicted_class must be a digit string."""
        from visionforge.blocks.batch_prediction import BatchPredictionBlock

        block = BatchPredictionBlock()
        with (
            patch(
                "visionforge.blocks.batch_prediction.ModelFactory.create",
                return_value=TinyMulticlassModel(),
            ),
            patch("visionforge.blocks.batch_prediction.torch.load", return_value={}),
            patch("torch.nn.Module.load_state_dict"),
        ):
            block.setup(multiclass_config)
        block.run()

        assert multiclass_config.batch_prediction is not None
        rows = _read_csv_rows(multiclass_config.batch_prediction.output_csv)
        for row in rows:
            assert row["predicted_class"].isdigit()


# ── report structure ───────────────────────────────────────────────────────────


class TestReport:
    def test_report_has_total_processed(
        self, multiclass_config: ExperimentConfig
    ) -> None:
        from visionforge.blocks.batch_prediction import BatchPredictionBlock

        block = BatchPredictionBlock()
        with (
            patch(
                "visionforge.blocks.batch_prediction.ModelFactory.create",
                return_value=TinyMulticlassModel(),
            ),
            patch("visionforge.blocks.batch_prediction.torch.load", return_value={}),
            patch("torch.nn.Module.load_state_dict"),
        ):
            block.setup(multiclass_config)
        block.run()

        report = block.report()
        assert "total_processed" in report
        assert report["total_processed"] == 3

    def test_report_has_failed_files(self, multiclass_config: ExperimentConfig) -> None:
        from visionforge.blocks.batch_prediction import BatchPredictionBlock

        block = BatchPredictionBlock()
        with (
            patch(
                "visionforge.blocks.batch_prediction.ModelFactory.create",
                return_value=TinyMulticlassModel(),
            ),
            patch("visionforge.blocks.batch_prediction.torch.load", return_value={}),
            patch("torch.nn.Module.load_state_dict"),
        ):
            block.setup(multiclass_config)
        block.run()

        report = block.report()
        assert "failed_files" in report
        assert report["failed_files"] == []

    def test_report_has_output_csv(self, multiclass_config: ExperimentConfig) -> None:
        from visionforge.blocks.batch_prediction import BatchPredictionBlock

        block = BatchPredictionBlock()
        with (
            patch(
                "visionforge.blocks.batch_prediction.ModelFactory.create",
                return_value=TinyMulticlassModel(),
            ),
            patch("visionforge.blocks.batch_prediction.torch.load", return_value={}),
            patch("torch.nn.Module.load_state_dict"),
        ):
            block.setup(multiclass_config)
        block.run()

        report = block.report()
        assert "output_csv" in report


# ── registry ───────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_registry_discovers_batch_prediction_block(self) -> None:
        """BlockRegistry must include BatchPredictionBlock after import."""
        import visionforge.blocks.batch_prediction  # noqa: F401
        from visionforge.blocks.registry import BlockRegistry

        registry = BlockRegistry.discover()
        assert "BatchPredictionBlock" in registry


# ── helpers ────────────────────────────────────────────────────────────────────


def _read_csv_header(path: Path) -> list[str]:
    """Return the header row of a CSV file."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or [])


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Return all data rows of a CSV file as dicts."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
