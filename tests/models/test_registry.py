from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from visionforge.models.factory import ModelFactory
from visionforge.models.registry import (
    build_custom_model,
    clear_registry,
    load_user_models,
    register_model,
    registered_models,
)
from visionforge.models.regression_factory import RegressionModelFactory
from visionforge.models.segmentation_factory import SegmentationModelFactory
from visionforge.utils.config import ModelConfig
from visionforge.utils.regression_config import RegressionModelConfig
from visionforge.utils.segmentation_config import SegmentationModelConfig

# Repo-root user_models/ shipped with the project (holds example_custom_model.py).
_REPO_USER_MODELS = Path(__file__).resolve().parents[2] / "user_models"


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """Keep the module-global registry isolated between tests."""
    clear_registry()
    yield
    clear_registry()


def _write_model_file(directory: Path, stem: str, name: str, in_features: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.py").write_text(
        "import torch.nn as nn\n"
        "from visionforge.models.registry import register_model\n"
        f'@register_model("{name}")\n'
        f"def build(num_classes):\n"
        f"    return nn.Linear({in_features}, num_classes)\n",
        encoding="utf-8",
    )


class TestRegisterAndBuild:
    def test_register_then_build_returns_module(self) -> None:
        @register_model("unit_net")
        def build(num_classes: int) -> nn.Module:
            return nn.Linear(8, num_classes)

        assert "unit_net" in registered_models()
        model = build_custom_model("unit_net", num_outputs=5)
        assert isinstance(model, nn.Linear)
        assert model.out_features == 5

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            register_model("  ")

    def test_unknown_model_raises_with_available_names(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError, match="not registered"):
            build_custom_model("missing", num_outputs=2, user_models_dir=tmp_path)


class TestLoadUserModels:
    def test_loads_py_files_and_registers(self, tmp_path: Path) -> None:
        _write_model_file(tmp_path, "my_model", "from_file", in_features=4)
        names = load_user_models(tmp_path)
        assert "from_file" in names
        model = build_custom_model("from_file", num_outputs=3, user_models_dir=tmp_path)
        assert isinstance(model, nn.Linear)
        assert model.in_features == 4 and model.out_features == 3

    def test_missing_directory_is_noop(self, tmp_path: Path) -> None:
        assert load_user_models(tmp_path / "does_not_exist") == []

    def test_underscore_files_are_skipped(self, tmp_path: Path) -> None:
        _write_model_file(tmp_path, "_helper", "private", in_features=4)
        assert "private" not in load_user_models(tmp_path)

    def test_bad_file_does_not_abort_discovery(self, tmp_path: Path) -> None:
        (tmp_path / "broken.py").write_text("this is not valid python !!!", "utf-8")
        _write_model_file(tmp_path, "good", "good_net", in_features=4)
        names = load_user_models(tmp_path)
        assert "good_net" in names  # the broken file is logged, not fatal


class TestShippedExample:
    def test_example_tiny_cnn_loads_and_runs(self) -> None:
        load_user_models(_REPO_USER_MODELS)
        assert "example_tiny_cnn" in registered_models()
        model = build_custom_model(
            "example_tiny_cnn", num_outputs=4, user_models_dir=_REPO_USER_MODELS
        ).eval()
        with torch.no_grad():
            out = model(torch.zeros(2, 3, 16, 16))
        assert out.shape == (2, 4)


class TestModelFactoryCustomPath:
    def test_factory_builds_registered_custom_model(self) -> None:
        @register_model("factory_net")
        def build(num_classes: int) -> nn.Module:
            return nn.Linear(6, num_classes)

        config = ModelConfig(custom_model="factory_net", num_classes=7)
        model = ModelFactory.create(config)
        assert isinstance(model, nn.Linear)
        assert model.out_features == 7

    def test_builtin_path_unaffected_when_custom_model_absent(self) -> None:
        # custom_model defaults to None -> the torchvision path still runs.
        config = ModelConfig(name="resnet18", num_classes=3, pretrained=False)
        model = ModelFactory.create(config)
        assert isinstance(model, nn.Module)

    def test_blank_custom_model_coerced_to_none(self) -> None:
        # An untouched GUI text field ("") must not trigger the custom path.
        assert ModelConfig(custom_model="   ").custom_model is None
        assert ModelConfig(custom_model="").custom_model is None


class TestRegressionFactoryCustomPath:
    def test_factory_builds_registered_custom_model_with_num_targets(self) -> None:
        @register_model("reg_net")
        def build(num_outputs: int) -> nn.Module:
            return nn.Linear(6, num_outputs)

        config = RegressionModelConfig(custom_model="reg_net", num_targets=3)
        model = RegressionModelFactory.create(config)
        assert isinstance(model, nn.Linear)
        assert model.out_features == 3  # the head width == num_targets

    def test_blank_custom_model_coerced_to_none(self) -> None:
        assert RegressionModelConfig(custom_model="  ").custom_model is None


class TestSegmentationFactoryCustomPath:
    def test_factory_builds_registered_custom_model_with_num_classes(self) -> None:
        @register_model("seg_net")
        def build(num_outputs: int) -> nn.Module:
            return nn.Conv2d(3, num_outputs, kernel_size=1)

        config = SegmentationModelConfig(custom_model="seg_net", num_classes=5)
        model = SegmentationModelFactory.create(config)
        assert isinstance(model, nn.Conv2d)
        assert model.out_channels == 5  # per-pixel logits == num_classes

    def test_blank_custom_model_coerced_to_none(self) -> None:
        assert SegmentationModelConfig(custom_model="").custom_model is None
