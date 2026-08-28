from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from visionforge.utils.detection_config import (
    _ULTRALYTICS_MODELS,
    DetectionConfig,
    DetectionDataConfig,
    DetectionModelConfig,
    DetectionTrainingConfig,
    load_detection_config,
)


def _data_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "data.yaml"
    p.write_text("path: .\ntrain: images\nval: images\nnc: 1\n", encoding="utf-8")
    return p


def _base_config(tmp_path: Path, overrides: dict | None = None) -> dict:
    raw: dict = {
        "name": "det_test",
        "model": {"backend": "ultralytics", "name": "yolo11n", "num_classes": 1},
        "data": {"data_yaml": str(_data_yaml(tmp_path))},
        "training": {"epochs": 1, "batch_size": 16, "learning_rate": 0.01},
    }
    if overrides:
        raw.update(overrides)
    return raw


# ── defaults & happy path ─────────────────────────────────────────────────────


class TestDefaults:
    def test_valid_minimal_config(self, tmp_path: Path) -> None:
        cfg = DetectionConfig.model_validate(_base_config(tmp_path))
        assert cfg.task == "detection"
        assert cfg.model.backend == "ultralytics"
        assert cfg.model.name == "yolo11n"

    def test_model_defaults(self) -> None:
        m = DetectionModelConfig()
        assert m.backend == "ultralytics"
        assert m.name == "yolo11n"
        assert m.pretrained is True

    def test_output_and_device_defaults(self, tmp_path: Path) -> None:
        cfg = DetectionConfig.model_validate(_base_config(tmp_path))
        assert cfg.output.models_dir == Path("outputs/models")
        assert cfg.device.kind == "cuda"


# ── backend ↔ model coherence ─────────────────────────────────────────────────


class TestBackendModelCoherence:
    def test_ultralytics_rejects_torchvision_model(self, tmp_path: Path) -> None:
        raw = _base_config(
            tmp_path,
            {"model": {"backend": "ultralytics", "name": "fasterrcnn_resnet50_fpn"}},
        )
        with pytest.raises(ValidationError, match="ultralytics"):
            DetectionConfig.model_validate(raw)

    def test_torchvision_accepts_faster_rcnn(self, tmp_path: Path) -> None:
        raw = _base_config(
            tmp_path,
            {"model": {"backend": "torchvision", "name": "fasterrcnn_resnet50_fpn"}},
        )
        cfg = DetectionConfig.model_validate(raw)
        assert cfg.model.backend == "torchvision"
        assert cfg.model.name == "fasterrcnn_resnet50_fpn"

    def test_torchvision_rejects_yolo_model(self, tmp_path: Path) -> None:
        raw = _base_config(
            tmp_path, {"model": {"backend": "torchvision", "name": "yolo11n"}}
        )
        with pytest.raises(ValidationError, match="torchvision"):
            DetectionConfig.model_validate(raw)

    def test_num_classes_zero_raises(self, tmp_path: Path) -> None:
        raw = _base_config(tmp_path, {"model": {"name": "yolo11n", "num_classes": 0}})
        with pytest.raises(ValidationError):
            DetectionConfig.model_validate(raw)


# ── training rules (detection differs from classification) ─────────────────────


class TestTrainingRules:
    def test_batch_size_need_not_be_power_of_two(self, tmp_path: Path) -> None:
        """Unlike classification, Ultralytics allows any positive batch size."""
        raw = _base_config(
            tmp_path,
            {"training": {"epochs": 1, "batch_size": 12, "learning_rate": 0.01}},
        )
        cfg = DetectionConfig.model_validate(raw)
        assert cfg.training.batch_size == 12

    def test_non_positive_learning_rate_raises(self, tmp_path: Path) -> None:
        raw = _base_config(
            tmp_path,
            {"training": {"epochs": 1, "batch_size": 16, "learning_rate": 0.0}},
        )
        with pytest.raises(ValidationError):
            DetectionConfig.model_validate(raw)

    def test_zero_epochs_raises(self, tmp_path: Path) -> None:
        raw = _base_config(
            tmp_path,
            {"training": {"epochs": 0, "batch_size": 16, "learning_rate": 0.01}},
        )
        with pytest.raises(ValidationError):
            DetectionConfig.model_validate(raw)


# ── model families (YOLOv8/9/10/11/12/26 + RT-DETR) ───────────────────────────


class TestModelFamilies:
    @pytest.mark.parametrize(
        "name",
        [
            "yolov8n",
            "yolov9t",
            "yolov9c",
            "yolov9e",
            "yolov10n",
            "yolov10b",
            "yolov10x",
            "yolo11n",
            "yolo12m",
            "yolo26n",
            "yolo26x",
            "rtdetr-l",
        ],
    )
    def test_known_ultralytics_models_accepted(self, tmp_path: Path, name: str) -> None:
        raw = _base_config(tmp_path, {"model": {"name": name}})
        cfg = DetectionConfig.model_validate(raw)
        assert cfg.model.name == name

    def test_yolo26_present_in_model_set(self) -> None:
        assert "yolo26n" in _ULTRALYTICS_MODELS
        assert "yolo26x" in _ULTRALYTICS_MODELS

    def test_unknown_yolo_version_rejected(self, tmp_path: Path) -> None:
        raw = _base_config(tmp_path, {"model": {"name": "yolo99n"}})
        with pytest.raises(ValidationError, match="ultralytics"):
            DetectionConfig.model_validate(raw)


# ── extended training hyperparameters ─────────────────────────────────────────


class TestTrainingHyperparameters:
    def test_defaults_match_ultralytics(self) -> None:
        t = DetectionTrainingConfig()
        assert t.optimizer == "auto"
        assert t.momentum == pytest.approx(0.937)
        assert t.weight_decay == pytest.approx(0.0005)
        assert t.lrf == pytest.approx(0.01)
        assert t.cos_lr is False
        assert t.box == pytest.approx(7.5)
        assert t.cls == pytest.approx(0.5)
        assert t.dfl == pytest.approx(1.5)
        assert t.close_mosaic == 10
        assert t.amp is True
        assert t.freeze is None

    def test_augmentation_defaults(self) -> None:
        a = DetectionTrainingConfig().augmentation
        assert a.fliplr == pytest.approx(0.5)
        assert a.mosaic == pytest.approx(1.0)
        assert a.flipud == pytest.approx(0.0)
        assert a.auto_augment == "randaugment"
        assert a.erasing == pytest.approx(0.4)

    def test_workers_defaults_to_automatic(self) -> None:
        """-1, not Ultralytics' 8 and not the old fixed 2 (ADR-103).

        The number that works depends on the free memory of the machine, not on
        the platform: on Windows every worker is a spawned process reloading
        torch's CUDA DLLs, and too many exhaust the page file (WinError 1455).
        `-1` asks the trainer to measure that at load time.
        """
        assert DetectionTrainingConfig().workers == -1

    def test_an_explicit_worker_count_is_still_honoured(self) -> None:
        assert DetectionTrainingConfig(workers=4).workers == 4

    def test_zero_workers_keeps_its_own_meaning(self) -> None:
        """Load in the main process — a real choice, not "automatic"."""
        assert DetectionTrainingConfig(workers=0).workers == 0

    def test_below_automatic_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            DetectionTrainingConfig(workers=-2)

    def test_accepts_overrides(self, tmp_path: Path) -> None:
        raw = _base_config(
            tmp_path,
            {
                "training": {
                    "epochs": 1,
                    "batch_size": 16,
                    "learning_rate": 0.01,
                    "optimizer": "AdamW",
                    "cos_lr": True,
                    "box": 5.0,
                    "augmentation": {"mosaic": 0.0, "mixup": 0.2, "fliplr": 0.0},
                }
            },
        )
        cfg = DetectionConfig.model_validate(raw)
        assert cfg.training.optimizer == "AdamW"
        assert cfg.training.cos_lr is True
        assert cfg.training.box == pytest.approx(5.0)
        assert cfg.training.augmentation.mosaic == pytest.approx(0.0)
        assert cfg.training.augmentation.mixup == pytest.approx(0.2)

    def test_invalid_optimizer_rejected(self, tmp_path: Path) -> None:
        raw = _base_config(
            tmp_path,
            {
                "training": {
                    "epochs": 1,
                    "batch_size": 16,
                    "learning_rate": 0.01,
                    "optimizer": "rmsprop_lowercase",
                }
            },
        )
        with pytest.raises(ValidationError):
            DetectionConfig.model_validate(raw)

    def test_probability_bounds_enforced(self, tmp_path: Path) -> None:
        raw = _base_config(
            tmp_path,
            {
                "training": {
                    "epochs": 1,
                    "batch_size": 16,
                    "learning_rate": 0.01,
                    "augmentation": {"fliplr": 1.5},
                }
            },
        )
        with pytest.raises(ValidationError):
            DetectionConfig.model_validate(raw)


# ── dataset source ────────────────────────────────────────────────────────────


class TestDataSource:
    def test_requires_a_source(self) -> None:
        with pytest.raises(ValidationError, match="data_yaml|base_dir"):
            DetectionDataConfig()

    def test_base_dir_layout_is_accepted(self, tmp_path: Path) -> None:
        d = DetectionDataConfig(base_dir=tmp_path)
        assert d.base_dir == tmp_path

    def test_missing_data_yaml_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="data_yaml"):
            DetectionDataConfig(data_yaml=tmp_path / "nope.yaml")

    def test_base_dir_must_be_directory(self, tmp_path: Path) -> None:
        not_a_dir = tmp_path / "plain.txt"
        not_a_dir.write_text("x", encoding="utf-8")
        with pytest.raises(ValidationError, match="directory"):
            DetectionDataConfig(base_dir=not_a_dir)

    def test_image_size_floor(self) -> None:
        with pytest.raises(ValidationError):
            DetectionDataConfig(base_dir=None, data_yaml=None, image_size=16)


# ── weights_path ──────────────────────────────────────────────────────────────


class TestWeightsPath:
    def test_missing_weights_path_raises(self, tmp_path: Path) -> None:
        raw = _base_config(
            tmp_path,
            {
                "model": {
                    "name": "yolo11n",
                    "weights_path": str(tmp_path / "missing.pt"),
                }
            },
        )
        with pytest.raises(ValidationError, match="weights_path"):
            DetectionConfig.model_validate(raw)


# ── load_detection_config ─────────────────────────────────────────────────────


class TestLoadDetectionConfig:
    def test_loads_valid_yaml(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "exp.yaml"
        cfg_path.write_text(yaml.safe_dump(_base_config(tmp_path)), encoding="utf-8")
        cfg = load_detection_config(cfg_path)
        assert cfg.name == "det_test"

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_detection_config(tmp_path / "nope.yaml")

    def test_non_mapping_raises(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_detection_config(cfg_path)
