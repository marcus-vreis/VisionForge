from __future__ import annotations

import pytest
import torch

from visionforge.models.anomaly_factory import (
    AnomalyModelFactory,
    ConvAutoencoder,
    PatchCore,
)
from visionforge.utils.anomaly_config import AnomalyModelConfig


class TestConvAutoencoder:
    def test_reconstruction_matches_input_shape(self) -> None:
        ae = ConvAutoencoder(latent_dim=32).eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = ae(x)
        assert out.shape == x.shape

    def test_handles_non_power_of_two(self) -> None:
        ae = ConvAutoencoder(latent_dim=16).eval()
        x = torch.randn(1, 3, 70, 50)
        with torch.no_grad():
            out = ae(x)
        assert out.shape == x.shape


class TestPatchCore:
    def test_extract_returns_patch_features(self) -> None:
        pc = PatchCore(backbone="resnet18", pretrained=False, coreset_ratio=0.5).eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            feats = pc.extract(x)
        assert feats.ndim == 3
        assert feats.shape[0] == 2  # batch preserved

    def test_fit_builds_memory_bank(self) -> None:
        pc = PatchCore(backbone="resnet18", pretrained=False, coreset_ratio=0.5)
        feats = torch.randn(40, 8)
        pc.fit(feats)
        assert pc.is_fitted
        # coreset keeps ~ratio of the patches, at least 1
        assert 1 <= pc.memory_size <= 40

    def test_score_returns_one_value_per_image(self) -> None:
        pc = PatchCore(backbone="resnet18", pretrained=False, coreset_ratio=0.5).eval()
        x = torch.randn(3, 3, 64, 64)
        with torch.no_grad():
            pc.fit(pc.extract(x).reshape(-1, pc.feature_dim))
            scores = pc.score(x)
        assert scores.shape == (3,)
        assert torch.isfinite(scores).all()

    def test_score_before_fit_raises(self) -> None:
        pc = PatchCore(backbone="resnet18", pretrained=False, coreset_ratio=0.5).eval()
        x = torch.randn(1, 3, 64, 64)
        with pytest.raises(RuntimeError, match="fit"):
            pc.score(x)


class TestAnomalyModelFactory:
    def test_creates_autoencoder(self) -> None:
        model = AnomalyModelFactory.create(
            AnomalyModelConfig(name="autoencoder", latent_dim=16)
        )
        assert isinstance(model, ConvAutoencoder)

    def test_creates_patchcore(self) -> None:
        model = AnomalyModelFactory.create(
            AnomalyModelConfig(name="patchcore", backbone="resnet18", pretrained=False)
        )
        assert isinstance(model, PatchCore)
