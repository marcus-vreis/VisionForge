import pytest
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from visionforge.utils.config import ExperimentConfig, load_config


def make_raw_config(tmp_path: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a valid raw config dict, with optional field overrides."""
    raw: dict[str, Any] = {
        "name": "test_experiment",
        "task": "binary",
        "model": {
            "name": "resnet50",
            "num_classes": 1,
            "pretrained": False,
        },
        "training": {
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 16,
            "early_stopping_patience": 5,
            "optimizer": "adam",
            "weight_decay": 0.0,
        },
        "data": {
            "base_dir": str(tmp_path),
            "train_dir": "train",
            "val_dir": "val",
            "test_dir": "test",
            "image_size": 224,
            "num_workers": 0,
            "pin_memory": False,
        },
    }
    if overrides:
        for key, value in overrides.items():
            keys = key.split(".")
            d: dict[str, Any] = raw
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value
    return raw


def write_yaml(tmp_path: Path, data: dict[str, Any], filename: str = "config.yaml") -> Path:
    """Write a dict as a YAML file and return its path."""
    path = tmp_path / filename
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)
    return path


class TestLoadConfig:

    def test_valid_config_loads_successfully(self, tmp_path: Path) -> None:
        """load_config() should return a valid ExperimentConfig from a YAML file."""
        path = write_yaml(tmp_path, make_raw_config(tmp_path))

        config = load_config(path)

        assert isinstance(config, ExperimentConfig)
        assert config.name == "test_experiment"
        assert config.model.name == "resnet50"
        assert config.training.learning_rate == 0.001

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """load_config() should raise FileNotFoundError for non-existent files."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "does_not_exist.yaml")

    def test_output_defaults_applied_when_missing(self, tmp_path: Path) -> None:
        """OutputConfig defaults should be applied if output section is omitted."""
        path = write_yaml(tmp_path, make_raw_config(tmp_path))

        config = load_config(path)

        assert config.output.models_dir == Path("outputs/models")
        assert config.output.logs_dir == Path("outputs/logs")


class TestModelConfig:

    def test_invalid_model_name_raises(self, tmp_path: Path) -> None:
        """Unknown model names should raise ValidationError."""
        path = write_yaml(tmp_path, make_raw_config(tmp_path, {"model.name": "resnet999"}))

        with pytest.raises(ValidationError):
            load_config(path)

    def test_num_classes_zero_raises(self, tmp_path: Path) -> None:
        """num_classes must be >= 1."""
        path = write_yaml(tmp_path, make_raw_config(tmp_path, {"model.num_classes": 0}))

        with pytest.raises(ValidationError):
            load_config(path)


class TestTrainingConfig:

    def test_negative_learning_rate_raises(self, tmp_path: Path) -> None:
        """learning_rate must be > 0."""
        path = write_yaml(tmp_path, make_raw_config(tmp_path, {"training.learning_rate": -0.001}))

        with pytest.raises(ValidationError):
            load_config(path)

    def test_zero_epochs_raises(self, tmp_path: Path) -> None:
        """epochs must be >= 1."""
        path = write_yaml(tmp_path, make_raw_config(tmp_path, {"training.epochs": 0}))

        with pytest.raises(ValidationError):
            load_config(path)

    def test_batch_size_not_power_of_two_raises(self, tmp_path: Path) -> None:
        """batch_size must be a power of 2."""
        path = write_yaml(tmp_path, make_raw_config(tmp_path, {"training.batch_size": 12}))

        with pytest.raises(ValidationError, match="power of 2"):
            load_config(path)

    def test_valid_batch_sizes(self, tmp_path: Path) -> None:
        """Common power-of-2 batch sizes should all be accepted."""
        for size in [1, 2, 4, 8, 16, 32, 64, 128]:
            path = write_yaml(
                tmp_path,
                make_raw_config(tmp_path, {"training.batch_size": size}),
                filename=f"config_{size}.yaml",
            )
            config = load_config(path)
            assert config.training.batch_size == size

    def test_invalid_optimizer_raises(self, tmp_path: Path) -> None:
        """Optimizer must be one of adam, sgd, adamw."""
        path = write_yaml(tmp_path, make_raw_config(tmp_path, {"training.optimizer": "rmsprop"}))

        with pytest.raises(ValidationError):
            load_config(path)


class TestDataConfig:

    def test_nonexistent_base_dir_raises(self, tmp_path: Path) -> None:
        """base_dir must exist on disk."""
        path = write_yaml(
            tmp_path, make_raw_config(tmp_path, {"data.base_dir": "/nonexistent/path"})
        )

        with pytest.raises(ValidationError, match="does not exist"):
            load_config(path)


class TestCrossValidation:

    def test_binary_task_with_num_classes_two_raises(self, tmp_path: Path) -> None:
        """Binary task must have num_classes=1."""
        path = write_yaml(tmp_path, make_raw_config(tmp_path, {"model.num_classes": 2}))

        with pytest.raises(ValidationError, match="Binary task"):
            load_config(path)

    def test_multiclass_task_with_num_classes_one_raises(self, tmp_path: Path) -> None:
        """Multiclass task must have num_classes >= 2."""
        path = write_yaml(
            tmp_path, make_raw_config(tmp_path, {"task": "multiclass", "model.num_classes": 1})
        )

        with pytest.raises(ValidationError, match="Multiclass task"):
            load_config(path)

    def test_multiclass_task_with_valid_num_classes(self, tmp_path: Path) -> None:
        """Multiclass task with num_classes >= 2 should be valid."""
        path = write_yaml(
            tmp_path, make_raw_config(tmp_path, {"task": "multiclass", "model.num_classes": 5})
        )

        config = load_config(path)
        assert config.task == "multiclass"
        assert config.model.num_classes == 5
