from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from visionforge.utils.regression_config import (
    RegressionConfig,
    RegressionDataConfig,
    RegressionModelConfig,
    RegressionTrainingConfig,
    load_regression_config,
)


def _base_config(tmp_path: Path, overrides: dict | None = None) -> dict:
    raw: dict = {
        "name": "reg_test",
        "model": {"name": "resnet50", "num_targets": 1},
        "data": {"base_dir": str(tmp_path), "target_columns": ["target"]},
        "training": {"epochs": 1, "batch_size": 16, "learning_rate": 0.001},
    }
    if overrides:
        raw.update(overrides)
    return raw


# ── defaults & happy path ─────────────────────────────────────────────────────


class TestDefaults:
    def test_valid_minimal_config(self, tmp_path: Path) -> None:
        cfg = RegressionConfig.model_validate(_base_config(tmp_path))
        assert cfg.task == "regression"
        assert cfg.model.name == "resnet50"
        assert cfg.model.num_targets == 1
        assert cfg.data.target_columns == ["target"]

    def test_model_defaults(self) -> None:
        m = RegressionModelConfig()
        # resnet18 since ADR-100: a first run should be quick.
        assert m.name == "resnet18"
        assert m.num_targets == 1
        assert m.pretrained is True
        assert m.weights_path is None

    def test_training_defaults(self) -> None:
        t = RegressionTrainingConfig()
        assert t.loss == "mse"
        assert t.optimizer == "adam"
        assert t.scheduler.kind == "none"

    def test_data_defaults(self, tmp_path: Path) -> None:
        d = RegressionDataConfig(base_dir=tmp_path)
        assert d.images_dir == "images"
        assert d.train_csv == "train.csv"
        assert d.image_column == "image"
        assert d.target_columns == ["target"]

    def test_output_and_device_defaults(self, tmp_path: Path) -> None:
        cfg = RegressionConfig.model_validate(_base_config(tmp_path))
        assert cfg.output.models_dir == Path("outputs/models")
        assert cfg.device.kind == "cuda"


# ── multi-target coherence ────────────────────────────────────────────────────


class TestTargetCoherence:
    def test_multi_target_matches_columns(self, tmp_path: Path) -> None:
        cfg = RegressionConfig.model_validate(
            _base_config(
                tmp_path,
                {
                    "model": {"name": "resnet50", "num_targets": 3},
                    "data": {
                        "base_dir": str(tmp_path),
                        "target_columns": ["x", "y", "z"],
                    },
                },
            )
        )
        assert cfg.model.num_targets == 3
        assert cfg.data.target_columns == ["x", "y", "z"]

    def test_mismatch_targets_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="num_targets"):
            RegressionConfig.model_validate(
                _base_config(
                    tmp_path,
                    {
                        "model": {"name": "resnet50", "num_targets": 2},
                        "data": {
                            "base_dir": str(tmp_path),
                            "target_columns": ["only_one"],
                        },
                    },
                )
            )

    def test_empty_target_columns_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="at least one column"):
            RegressionDataConfig(base_dir=tmp_path, target_columns=[])

    def test_duplicate_target_columns_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="unique"):
            RegressionDataConfig(base_dir=tmp_path, target_columns=["a", "a"])


# ── field validation ──────────────────────────────────────────────────────────


class TestFieldValidation:
    def test_missing_base_dir_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="base_dir does not exist"):
            RegressionDataConfig(base_dir=tmp_path / "nope", target_columns=["t"])

    def test_base_dir_must_be_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "regular.txt"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(ValidationError, match="must be a directory"):
            RegressionDataConfig(base_dir=f, target_columns=["t"])

    def test_invalid_backbone_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            RegressionConfig.model_validate(
                _base_config(tmp_path, {"model": {"name": "resnet999"}})
            )

    def test_num_targets_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            RegressionModelConfig(num_targets=0)

    def test_invalid_loss_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RegressionTrainingConfig(loss="rmse")  # type: ignore[arg-type]

    def test_weights_path_must_exist(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="weights_path does not exist"):
            RegressionModelConfig(weights_path=tmp_path / "missing.pth")

    def test_weights_path_must_be_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="must be a file"):
            RegressionModelConfig(weights_path=tmp_path)


# ── load_regression_config ────────────────────────────────────────────────────


class TestLoadRegressionConfig:
    def test_loads_valid_yaml(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "reg.yaml"
        cfg_path.write_text(yaml.safe_dump(_base_config(tmp_path)), encoding="utf-8")
        cfg = load_regression_config(cfg_path)
        assert cfg.name == "reg_test"
        assert cfg.task == "regression"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_regression_config(tmp_path / "nope.yaml")

    def test_directory_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a file"):
            load_regression_config(tmp_path)

    def test_non_mapping_yaml_raises(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain a YAML mapping"):
            load_regression_config(cfg_path)
