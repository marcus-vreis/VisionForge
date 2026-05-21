from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from visionforge.blocks.export_onnx import ExportONNXBlock
from visionforge.utils.config import ExperimentConfig, ExportONNXConfig

# ── tiny model ────────────────────────────────────────────────────────────────


class TinyModel(nn.Module):
    """2-class head on a fixed-size input for fast ONNX export tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Minimal ImageFolder so DataConfig validates."""
    for split in ["train", "val", "test"]:
        for cls in ["cat", "dog"]:
            (tmp_path / split / cls).mkdir(parents=True)
    return tmp_path


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    """Write a real TinyModel state_dict so the block can load it."""
    model = TinyModel()
    ckpt = tmp_path / "best_model.pth"
    torch.save(model.state_dict(), ckpt)
    return ckpt


@pytest.fixture
def base_config(dataset_root: Path, tmp_path: Path) -> dict[str, Any]:
    """Shared config dict used by most tests."""
    return {
        "name": "onnx_test",
        "task": "multiclass",
        "block": "export_onnx",
        "model": {"name": "alexnet", "num_classes": 2, "pretrained": False},
        "training": {
            "learning_rate": 0.001,
            "epochs": 1,
            "batch_size": 2,
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
    }


def _make_config(
    base: dict[str, Any],
    checkpoint: Path,
    output_onnx: Path,
    **overrides: Any,
) -> ExperimentConfig:
    """Build an ExperimentConfig with export_onnx block configured."""
    onnx_cfg: dict[str, Any] = {
        "checkpoint_path": str(checkpoint),
        "output_onnx": str(output_onnx),
        "opset_version": 17,
        "dynamic_axes": True,
        "validate": True,
        "validation_tolerance": 1e-4,
        "benchmark": True,
        "benchmark_runs": 5,
    }
    onnx_cfg.update(overrides)
    return ExperimentConfig.model_validate({**base, "export_onnx": onnx_cfg})


# ── ExportONNXConfig validation ───────────────────────────────────────────────


class TestExportONNXConfig:
    def test_checkpoint_must_exist(self, tmp_path: Path) -> None:
        """field_validator rejects a checkpoint_path that does not exist on disk."""
        with pytest.raises(ValidationError, match="checkpoint_path"):
            ExportONNXConfig(
                checkpoint_path=tmp_path / "missing.pth",
                output_onnx=tmp_path / "out.onnx",
            )

    def test_checkpoint_must_be_file(self, tmp_path: Path) -> None:
        """field_validator rejects a directory as checkpoint_path."""
        d = tmp_path / "adir"
        d.mkdir()
        with pytest.raises(ValidationError, match="checkpoint_path"):
            ExportONNXConfig(
                checkpoint_path=d,
                output_onnx=tmp_path / "out.onnx",
            )

    def test_opset_lower_bound(self, checkpoint: Path, tmp_path: Path) -> None:
        """opset_version < 11 must raise ValidationError."""
        with pytest.raises(ValidationError):
            ExportONNXConfig(
                checkpoint_path=checkpoint,
                output_onnx=tmp_path / "out.onnx",
                opset_version=10,
            )

    def test_opset_upper_bound(self, checkpoint: Path, tmp_path: Path) -> None:
        """opset_version > 20 must raise ValidationError."""
        with pytest.raises(ValidationError):
            ExportONNXConfig(
                checkpoint_path=checkpoint,
                output_onnx=tmp_path / "out.onnx",
                opset_version=21,
            )

    def test_validation_tolerance_must_be_positive(
        self, checkpoint: Path, tmp_path: Path
    ) -> None:
        """validation_tolerance <= 0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            ExportONNXConfig(
                checkpoint_path=checkpoint,
                output_onnx=tmp_path / "out.onnx",
                validation_tolerance=0.0,
            )

    def test_benchmark_runs_minimum(self, checkpoint: Path, tmp_path: Path) -> None:
        """benchmark_runs < 5 must raise ValidationError."""
        with pytest.raises(ValidationError):
            ExportONNXConfig(
                checkpoint_path=checkpoint,
                output_onnx=tmp_path / "out.onnx",
                benchmark_runs=4,
            )

    def test_defaults(self, checkpoint: Path, tmp_path: Path) -> None:
        """ExportONNXConfig defaults are sensible."""
        cfg = ExportONNXConfig(
            checkpoint_path=checkpoint,
            output_onnx=tmp_path / "out.onnx",
        )
        assert cfg.opset_version == 17
        assert cfg.dynamic_axes is True
        assert cfg.run_validate is True
        assert cfg.validation_tolerance == pytest.approx(1e-4)
        assert cfg.benchmark is True
        assert cfg.benchmark_runs == 50


# ── ExperimentConfig literal extension ────────────────────────────────────────


class TestExperimentConfigExtension:
    def test_block_literal_accepts_export_onnx(
        self, base_config: dict[str, Any], checkpoint: Path, tmp_path: Path
    ) -> None:
        """ExperimentConfig must accept block='export_onnx'."""
        cfg = _make_config(base_config, checkpoint, tmp_path / "out.onnx")
        assert cfg.block == "export_onnx"
        assert cfg.export_onnx is not None

    def test_export_onnx_field_defaults_to_none(
        self, base_config: dict[str, Any], dataset_root: Path
    ) -> None:
        """export_onnx field is None when not supplied."""
        cfg = ExperimentConfig.model_validate(base_config)
        assert cfg.export_onnx is None


# ── ExportONNXBlock — setup ───────────────────────────────────────────────────


class TestExportONNXBlockSetup:
    def test_setup_raises_when_config_missing(
        self, base_config: dict[str, Any]
    ) -> None:
        """setup() must raise ValueError when export_onnx config is absent."""
        cfg = ExperimentConfig.model_validate(base_config)
        block = ExportONNXBlock()
        with pytest.raises(ValueError, match="export_onnx"):
            block.setup(cfg)

    def test_report_before_run_returns_empty(
        self, base_config: dict[str, Any], checkpoint: Path, tmp_path: Path
    ) -> None:
        """report() before run() returns an empty dict."""
        cfg = _make_config(base_config, checkpoint, tmp_path / "out.onnx")
        block = ExportONNXBlock()
        with patch(
            "visionforge.blocks.export_onnx.ModelFactory.create",
            return_value=TinyModel(),
        ):
            block.setup(cfg)
        assert block.report() == {}


# ── ExportONNXBlock — happy path ──────────────────────────────────────────────


class TestExportONNXBlockHappyPath:
    @patch(
        "visionforge.blocks.export_onnx.ModelFactory.create",
        return_value=TinyModel(),
    )
    def test_onnx_file_is_written(
        self,
        _mock: Any,
        base_config: dict[str, Any],
        checkpoint: Path,
        tmp_path: Path,
    ) -> None:
        """run() must write the .onnx file to the configured output path."""
        out = tmp_path / "model.onnx"
        cfg = _make_config(base_config, checkpoint, out)
        block = ExportONNXBlock()
        block.setup(cfg)
        block.run()
        assert out.exists()
        assert out.stat().st_size > 0

    @patch(
        "visionforge.blocks.export_onnx.ModelFactory.create",
        return_value=TinyModel(),
    )
    def test_validation_json_pass(
        self,
        _mock: Any,
        base_config: dict[str, Any],
        checkpoint: Path,
        tmp_path: Path,
    ) -> None:
        """onnx_validation.json must exist and show passed=True for a correct export."""
        out = tmp_path / "model.onnx"
        cfg = _make_config(base_config, checkpoint, out)
        block = ExportONNXBlock()
        block.setup(cfg)
        block.run()

        val_json = out.parent / "onnx_validation.json"
        assert val_json.exists()
        data = json.loads(val_json.read_text(encoding="utf-8"))
        assert data["passed"] is True
        assert "max_abs_diff" in data

    @patch(
        "visionforge.blocks.export_onnx.ModelFactory.create",
        return_value=TinyModel(),
    )
    def test_benchmark_json_contains_stats(
        self,
        _mock: Any,
        base_config: dict[str, Any],
        checkpoint: Path,
        tmp_path: Path,
    ) -> None:
        """onnx_benchmark.json must contain mean, p50, p95 as float values."""
        out = tmp_path / "model.onnx"
        cfg = _make_config(base_config, checkpoint, out, benchmark_runs=5)
        block = ExportONNXBlock()
        block.setup(cfg)
        block.run()

        bench_json = out.parent / "onnx_benchmark.json"
        assert bench_json.exists()
        data = json.loads(bench_json.read_text(encoding="utf-8"))
        assert "mean_ms" in data
        assert "p50_ms" in data
        assert "p95_ms" in data
        assert isinstance(data["mean_ms"], float)
        assert isinstance(data["p50_ms"], float)
        assert isinstance(data["p95_ms"], float)

    @patch(
        "visionforge.blocks.export_onnx.ModelFactory.create",
        return_value=TinyModel(),
    )
    def test_report_contains_expected_keys(
        self,
        _mock: Any,
        base_config: dict[str, Any],
        checkpoint: Path,
        tmp_path: Path,
    ) -> None:
        """report() after run() must include file_size_bytes, validation, and benchmark."""
        out = tmp_path / "model.onnx"
        cfg = _make_config(base_config, checkpoint, out, benchmark_runs=5)
        block = ExportONNXBlock()
        block.setup(cfg)
        block.run()
        report = block.report()

        assert "file_size_bytes" in report
        assert report["file_size_bytes"] > 0
        assert "validation" in report
        assert "benchmark" in report

    @patch(
        "visionforge.blocks.export_onnx.ModelFactory.create",
        return_value=TinyModel(),
    )
    def test_opset_version_honoured(
        self,
        _mock: Any,
        base_config: dict[str, Any],
        checkpoint: Path,
        tmp_path: Path,
    ) -> None:
        """The exported ONNX model must carry the requested opset version."""
        import onnx

        out = tmp_path / "model.onnx"
        cfg = _make_config(base_config, checkpoint, out, opset_version=13)
        block = ExportONNXBlock()
        block.setup(cfg)
        block.run()

        model_proto = onnx.load(str(out))
        opsets = {op.domain: op.version for op in model_proto.opset_import}
        assert opsets.get("") == 13

    @patch(
        "visionforge.blocks.export_onnx.ModelFactory.create",
        return_value=TinyModel(),
    )
    def test_static_axes_model(
        self,
        _mock: Any,
        base_config: dict[str, Any],
        checkpoint: Path,
        tmp_path: Path,
    ) -> None:
        """dynamic_axes=False must produce a model with a fixed batch dimension."""
        import onnx

        out = tmp_path / "model_static.onnx"
        cfg = _make_config(base_config, checkpoint, out, dynamic_axes=False)
        block = ExportONNXBlock()
        block.setup(cfg)
        block.run()

        onnx.checker.check_model(str(out))
        model_proto = onnx.load(str(out))
        input_shape = model_proto.graph.input[0].type.tensor_type.shape
        # batch dimension must be a concrete integer (not a symbolic string)
        batch_dim = input_shape.dim[0]
        assert batch_dim.HasField("dim_value")
        assert batch_dim.dim_value == 1


# ── ExportONNXBlock — validation failure ──────────────────────────────────────


class TestExportONNXBlockValidationFailure:
    @patch(
        "visionforge.blocks.export_onnx.ModelFactory.create",
        return_value=TinyModel(),
    )
    def test_validation_failure_when_outputs_differ(
        self,
        _mock: Any,
        base_config: dict[str, Any],
        checkpoint: Path,
        tmp_path: Path,
    ) -> None:
        """onnx_validation.json shows passed=False when onnxruntime returns garbage."""
        import onnxruntime as ort

        out = tmp_path / "model.onnx"
        cfg = _make_config(
            base_config, checkpoint, out, validate=True, validation_tolerance=1e-10
        )

        # Return a deliberately wrong output to force failure.
        wrong_output = [np.ones((1, 2), dtype=np.float32) * 999.0]
        mock_session = MagicMock(spec=ort.InferenceSession)
        mock_session.run.return_value = wrong_output
        mock_session.get_inputs.return_value = [
            MagicMock(name="input", shape=[1, 3, 32, 32])
        ]

        with patch("onnxruntime.InferenceSession", return_value=mock_session):
            block = ExportONNXBlock()
            block.setup(cfg)
            block.run()

        val_json = out.parent / "onnx_validation.json"
        data = json.loads(val_json.read_text(encoding="utf-8"))
        assert data["passed"] is False


# ── ExportONNXBlock — skip flags ──────────────────────────────────────────────


class TestExportONNXBlockSkipFlags:
    @patch(
        "visionforge.blocks.export_onnx.ModelFactory.create",
        return_value=TinyModel(),
    )
    def test_no_validation_json_when_validate_false(
        self,
        _mock: Any,
        base_config: dict[str, Any],
        checkpoint: Path,
        tmp_path: Path,
    ) -> None:
        """onnx_validation.json must not be written when validate=False."""
        out = tmp_path / "model.onnx"
        cfg = _make_config(
            base_config, checkpoint, out, validate=False, benchmark=False
        )
        block = ExportONNXBlock()
        block.setup(cfg)
        block.run()

        assert not (out.parent / "onnx_validation.json").exists()

    @patch(
        "visionforge.blocks.export_onnx.ModelFactory.create",
        return_value=TinyModel(),
    )
    def test_no_benchmark_json_when_benchmark_false(
        self,
        _mock: Any,
        base_config: dict[str, Any],
        checkpoint: Path,
        tmp_path: Path,
    ) -> None:
        """onnx_benchmark.json must not be written when benchmark=False."""
        out = tmp_path / "model.onnx"
        cfg = _make_config(
            base_config, checkpoint, out, validate=False, benchmark=False
        )
        block = ExportONNXBlock()
        block.setup(cfg)
        block.run()

        assert not (out.parent / "onnx_benchmark.json").exists()
