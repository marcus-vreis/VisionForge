from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from visionforge.utils.segmentation_config import (
    SegmentationConfig,
    SegmentationDataConfig,
    SegmentationModelConfig,
    SegmentationTrainingConfig,
    load_segmentation_config,
)


def _base_config(tmp_path: Path, overrides: dict | None = None) -> dict:
    raw: dict = {
        "name": "seg_test",
        "model": {"name": "deeplabv3_resnet50", "num_classes": 3},
        "data": {"base_dir": str(tmp_path)},
        "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.001},
    }
    if overrides:
        raw.update(overrides)
    return raw


# ── defaults & happy path ─────────────────────────────────────────────────────


class TestDefaults:
    def test_valid_minimal_config(self, tmp_path: Path) -> None:
        cfg = SegmentationConfig.model_validate(_base_config(tmp_path))
        assert cfg.task == "segmentation"
        assert cfg.model.name == "deeplabv3_resnet50"
        assert cfg.model.num_classes == 3

    def test_model_defaults(self) -> None:
        m = SegmentationModelConfig()
        # unet since ADR-100: deeplabv3_resnet50 at 512px exhausts modest GPUs.
        assert m.name == "unet"
        assert m.num_classes == 2
        assert m.pretrained is True
        assert m.weights_path is None

    def test_training_defaults(self) -> None:
        t = SegmentationTrainingConfig()
        assert t.loss == "cross_entropy"
        assert t.optimizer == "adam"
        assert t.scheduler.kind == "none"

    def test_data_defaults(self, tmp_path: Path) -> None:
        d = SegmentationDataConfig(base_dir=tmp_path)
        assert d.images_subdir == "images"
        assert d.masks_subdir == "masks"
        assert d.train_dir == "train"
        assert d.val_dir == "val"
        assert d.test_dir == "test"
        assert d.ignore_index == 255
        assert d.image_size == 512

    def test_output_and_device_defaults(self, tmp_path: Path) -> None:
        cfg = SegmentationConfig.model_validate(_base_config(tmp_path))
        assert cfg.output.models_dir == Path("outputs/models")
        assert cfg.device.kind == "cuda"


# ── model name validation ─────────────────────────────────────────────────────


class TestModelName:
    @pytest.mark.parametrize(
        "name",
        [
            "unet",
            "deeplabv3_resnet50",
            "deeplabv3_resnet101",
            "deeplabv3_mobilenet_v3_large",
            "fcn_resnet50",
            "fcn_resnet101",
            "lraspp_mobilenet_v3_large",
        ],
    )
    def test_all_supported_backbones_accepted(self, tmp_path: Path, name: str) -> None:
        cfg = SegmentationConfig.model_validate(
            _base_config(tmp_path, {"model": {"name": name, "num_classes": 2}})
        )
        assert cfg.model.name == name

    def test_invalid_backbone_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            SegmentationConfig.model_validate(
                _base_config(tmp_path, {"model": {"name": "segnet999"}})
            )


# ── ignore_index coherence ────────────────────────────────────────────────────


class TestIgnoreIndexCoherence:
    def test_ignore_index_above_class_range_accepted(self, tmp_path: Path) -> None:
        cfg = SegmentationConfig.model_validate(
            _base_config(
                tmp_path,
                {
                    "model": {"name": "fcn_resnet50", "num_classes": 4},
                    "data": {"base_dir": str(tmp_path), "ignore_index": 255},
                },
            )
        )
        assert cfg.data.ignore_index == 255

    def test_ignore_index_colliding_with_class_rejected(self, tmp_path: Path) -> None:
        # ignore_index=2 collides with a real class id when num_classes=4 (ids 0..3)
        with pytest.raises(ValidationError, match="ignore_index"):
            SegmentationConfig.model_validate(
                _base_config(
                    tmp_path,
                    {
                        "model": {"name": "fcn_resnet50", "num_classes": 4},
                        "data": {"base_dir": str(tmp_path), "ignore_index": 2},
                    },
                )
            )

    def test_negative_ignore_index_allowed(self, tmp_path: Path) -> None:
        # -1 is a common "no ignore" sentinel and never collides with a class id.
        cfg = SegmentationConfig.model_validate(
            _base_config(
                tmp_path,
                {"data": {"base_dir": str(tmp_path), "ignore_index": -1}},
            )
        )
        assert cfg.data.ignore_index == -1


# ── field validation ──────────────────────────────────────────────────────────


class TestFieldValidation:
    def test_missing_base_dir_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="base_dir does not exist"):
            SegmentationDataConfig(base_dir=tmp_path / "nope")

    def test_base_dir_must_be_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "regular.txt"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(ValidationError, match="must be a directory"):
            SegmentationDataConfig(base_dir=f)

    def test_num_classes_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            SegmentationModelConfig(num_classes=0)

    def test_invalid_loss_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SegmentationTrainingConfig(loss="focal")  # type: ignore[arg-type]

    @pytest.mark.parametrize("loss", ["cross_entropy", "dice", "combined"])
    def test_supported_losses_accepted(self, loss: str) -> None:
        t = SegmentationTrainingConfig(loss=loss)  # type: ignore[arg-type]
        assert t.loss == loss

    def test_batch_size_allows_non_power_of_two(self) -> None:
        t = SegmentationTrainingConfig(batch_size=6)
        assert t.batch_size == 6

    def test_image_size_minimum_enforced(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            SegmentationDataConfig(base_dir=tmp_path, image_size=16)

    def test_weights_path_must_exist(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="weights_path does not exist"):
            SegmentationModelConfig(weights_path=tmp_path / "missing.pth")

    def test_weights_path_must_be_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="must be a file"):
            SegmentationModelConfig(weights_path=tmp_path)


# ── load_segmentation_config ──────────────────────────────────────────────────


class TestLoadSegmentationConfig:
    def test_loads_valid_yaml(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "seg.yaml"
        cfg_path.write_text(yaml.safe_dump(_base_config(tmp_path)), encoding="utf-8")
        cfg = load_segmentation_config(cfg_path)
        assert cfg.name == "seg_test"
        assert cfg.task == "segmentation"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_segmentation_config(tmp_path / "nope.yaml")

    def test_directory_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a file"):
            load_segmentation_config(tmp_path)

    def test_non_mapping_yaml_raises(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain a YAML mapping"):
            load_segmentation_config(cfg_path)
