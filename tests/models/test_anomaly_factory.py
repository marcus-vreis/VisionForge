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

    def test_fit_reports_progress_as_the_bank_fills(self) -> None:
        """Building the bank takes minutes to hours; silence reads as hung."""
        pc = PatchCore(backbone="resnet18", pretrained=False, coreset_ratio=0.5)
        seen: list[tuple[int, int]] = []

        pc.fit(
            torch.randn(40, 8), progress=lambda done, total: seen.append((done, total))
        )

        assert seen, "fit reported nothing"
        assert seen[-1][0] == seen[-1][1], "the last report must be complete"
        # Monotonic and bounded — a bar that goes backwards is worse than none.
        assert all(a[0] <= b[0] for a, b in zip(seen, seen[1:], strict=False))
        assert all(0 <= done <= total for done, total in seen)

    def test_progress_is_throttled_for_a_large_bank(self) -> None:
        """~100 reports whatever k is: one per selection would be a flood."""
        pc = PatchCore(backbone="resnet18", pretrained=False, coreset_ratio=1.0)
        calls = 0

        def count(done: int, total: int) -> None:
            nonlocal calls
            calls += 1

        # 2000 patches, ratio 1.0 clamps to "keep everything" — use a real
        # subsample instead so the greedy loop actually runs.
        pc._coreset_ratio = 0.5
        pc.fit(torch.randn(2000, 8), progress=count)

        assert calls <= 110, f"{calls} progress calls is a flood"
        assert calls >= 50, f"{calls} progress calls is too coarse to watch"

    def test_fit_without_a_callback_still_works(self) -> None:
        pc = PatchCore(backbone="resnet18", pretrained=False, coreset_ratio=0.5)

        pc.fit(torch.randn(40, 8))

        assert pc.is_fitted

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

    def test_fitted_state_reloads_into_fresh_instance(self) -> None:
        # A fresh instance must load a fitted checkpoint despite the memory bank
        # growing from [0, C] to [K, C] (batch-predict reload path, ADR-041 s3).
        trained = PatchCore(
            backbone="resnet18", pretrained=False, coreset_ratio=0.5
        ).eval()
        x = torch.randn(3, 3, 64, 64)
        with torch.no_grad():
            trained.fit(trained.extract(x).reshape(-1, trained.feature_dim))
        assert trained.memory_size > 0

        fresh = PatchCore(backbone="resnet18", pretrained=False, coreset_ratio=0.5)
        fresh.load_state_dict(trained.state_dict())  # must not raise on _memory
        fresh.eval()  # eval mode so BatchNorm uses the loaded running stats

        assert fresh.memory_size == trained.memory_size
        with torch.no_grad():
            assert torch.allclose(fresh.score(x), trained.score(x))


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
