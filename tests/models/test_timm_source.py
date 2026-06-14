from __future__ import annotations

import sys
import types
from typing import Any

import pytest
import torch.nn as nn
from pydantic import ValidationError

from visionforge.models.factory import ModelFactory
from visionforge.models.regression_factory import RegressionModelFactory
from visionforge.models.timm_source import build_timm_model
from visionforge.utils.config import ModelConfig
from visionforge.utils.regression_config import RegressionModelConfig


class TestBuildTimmModel:
    def test_calls_create_model_with_output_dim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def create_model(name: str, pretrained: bool, num_classes: int) -> nn.Module:
            captured.update(name=name, pretrained=pretrained, num_classes=num_classes)
            return nn.Linear(4, num_classes)

        fake_timm = types.ModuleType("timm")
        fake_timm.create_model = create_model  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "timm", fake_timm)

        model = build_timm_model("convnext_tiny", num_outputs=7, pretrained=False)
        assert isinstance(model, nn.Linear)
        assert model.out_features == 7
        assert captured == {
            "name": "convnext_tiny",
            "pretrained": False,
            "num_classes": 7,
        }


class TestFactoryTimmRouting:
    def test_classification_factory_routes_to_timm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: dict[str, Any] = {}

        def fake(name: str, *, num_outputs: int, pretrained: bool) -> nn.Module:
            called.update(name=name, num_outputs=num_outputs, pretrained=pretrained)
            return nn.Linear(4, num_outputs)

        monkeypatch.setattr("visionforge.models.factory.build_timm_model", fake)
        model = ModelFactory.create(
            ModelConfig(timm_model="convnext_tiny", num_classes=5, pretrained=False)
        )
        assert isinstance(model, nn.Linear)
        assert model.out_features == 5
        assert called == {
            "name": "convnext_tiny",
            "num_outputs": 5,
            "pretrained": False,
        }

    def test_regression_factory_routes_to_timm_with_num_targets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake(name: str, *, num_outputs: int, pretrained: bool) -> nn.Module:
            return nn.Linear(4, num_outputs)

        monkeypatch.setattr(
            "visionforge.models.regression_factory.build_timm_model", fake
        )
        model = RegressionModelFactory.create(
            RegressionModelConfig(
                timm_model="convnext_tiny", num_targets=3, pretrained=False
            )
        )
        assert isinstance(model, nn.Linear)
        assert model.out_features == 3  # head width == num_targets


class TestModelSourceValidation:
    def test_custom_and_timm_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValidationError, match="only one"):
            ModelConfig(custom_model="a", timm_model="b")
        with pytest.raises(ValidationError, match="only one"):
            RegressionModelConfig(custom_model="a", timm_model="b")

    def test_blank_timm_model_coerced_to_none(self) -> None:
        assert ModelConfig(timm_model="  ").timm_model is None
        assert RegressionModelConfig(timm_model="").timm_model is None
        # blank custom + real timm is fine (blank coerces away before the check)
        cfg = ModelConfig(custom_model="", timm_model="convnext_tiny")
        assert cfg.custom_model is None
        assert cfg.timm_model == "convnext_tiny"
