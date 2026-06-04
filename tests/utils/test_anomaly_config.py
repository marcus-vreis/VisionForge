from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from visionforge.utils.anomaly_config import (
    AnomalyConfig,
    AnomalyDataConfig,
    AnomalyModelConfig,
    AnomalyTrainingConfig,
    load_anomaly_config,
)


def _base_config(tmp_path: Path, overrides: dict | None = None) -> dict:
    raw: dict = {
        "name": "anom_test",
        "model": {"name": "autoencoder"},
        "data": {"base_dir": str(tmp_path)},
        "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.001},
    }
    if overrides:
        raw.update(overrides)
    return raw


# ── defaults & happy path ─────────────────────────────────────────────────────


class TestDefaults:
    def test_valid_minimal_config(self, tmp_path: Path) -> None:
        cfg = AnomalyConfig.model_validate(_base_config(tmp_path))
        assert cfg.task == "anomaly"
        assert cfg.model.name == "autoencoder"

    def test_model_defaults(self) -> None:
        m = AnomalyModelConfig()
        assert m.name == "autoencoder"
        assert m.backbone == "resnet18"
        assert m.latent_dim == 512
        assert 0.0 < m.coreset_ratio <= 1.0
        assert m.pretrained is True

    def test_training_defaults(self) -> None:
        t = AnomalyTrainingConfig()
        assert t.optimizer == "adam"
        assert t.threshold_percentile == 95.0
        assert t.scheduler.kind == "none"

    def test_data_defaults(self, tmp_path: Path) -> None:
        d = AnomalyDataConfig(base_dir=tmp_path)
        assert d.train_dir == "train"
        assert d.test_dir == "test"
        assert d.normal_dir == "good"
        assert d.image_size == 256

    def test_output_and_device_defaults(self, tmp_path: Path) -> None:
        cfg = AnomalyConfig.model_validate(_base_config(tmp_path))
        assert cfg.output.models_dir == Path("outputs/models")
        assert cfg.device.kind == "cuda"


# ── model name + backbone validation ──────────────────────────────────────────


class TestModelName:
    @pytest.mark.parametrize("name", ["autoencoder", "patchcore"])
    def test_supported_models_accepted(self, tmp_path: Path, name: str) -> None:
        cfg = AnomalyConfig.model_validate(
            _base_config(tmp_path, {"model": {"name": name}})
        )
        assert cfg.model.name == name

    def test_invalid_model_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            AnomalyConfig.model_validate(
                _base_config(tmp_path, {"model": {"name": "ganomaly999"}})
            )

    @pytest.mark.parametrize(
        "backbone", ["resnet18", "resnet34", "resnet50", "wide_resnet50_2"]
    )
    def test_supported_backbones_accepted(self, backbone: str) -> None:
        m = AnomalyModelConfig(name="patchcore", backbone=backbone)  # type: ignore[arg-type]
        assert m.backbone == backbone

    def test_invalid_backbone_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AnomalyModelConfig(backbone="vgg16")  # type: ignore[arg-type]


# ── field validation ──────────────────────────────────────────────────────────


class TestFieldValidation:
    def test_missing_base_dir_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="base_dir does not exist"):
            AnomalyDataConfig(base_dir=tmp_path / "nope")

    def test_base_dir_must_be_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "regular.txt"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(ValidationError, match="must be a directory"):
            AnomalyDataConfig(base_dir=f)

    def test_latent_dim_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            AnomalyModelConfig(latent_dim=0)

    def test_coreset_ratio_bounds(self) -> None:
        with pytest.raises(ValidationError):
            AnomalyModelConfig(coreset_ratio=0.0)
        with pytest.raises(ValidationError):
            AnomalyModelConfig(coreset_ratio=1.5)

    def test_threshold_percentile_bounds(self) -> None:
        with pytest.raises(ValidationError):
            AnomalyTrainingConfig(threshold_percentile=-1.0)
        with pytest.raises(ValidationError):
            AnomalyTrainingConfig(threshold_percentile=101.0)

    def test_batch_size_allows_non_power_of_two(self) -> None:
        t = AnomalyTrainingConfig(batch_size=6)
        assert t.batch_size == 6

    def test_image_size_minimum_enforced(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            AnomalyDataConfig(base_dir=tmp_path, image_size=16)


# ── load_anomaly_config ───────────────────────────────────────────────────────


class TestLoadAnomalyConfig:
    def test_loads_valid_yaml(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "anom.yaml"
        cfg_path.write_text(yaml.safe_dump(_base_config(tmp_path)), encoding="utf-8")
        cfg = load_anomaly_config(cfg_path)
        assert cfg.name == "anom_test"
        assert cfg.task == "anomaly"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_anomaly_config(tmp_path / "nope.yaml")

    def test_directory_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a file"):
            load_anomaly_config(tmp_path)

    def test_non_mapping_yaml_raises(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain a YAML mapping"):
            load_anomaly_config(cfg_path)
