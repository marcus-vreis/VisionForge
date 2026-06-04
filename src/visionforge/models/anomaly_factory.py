"""Anomaly-detection models (Phase 9).

Two model families behind one factory:

- **ConvAutoencoder** — a hand-rolled convolutional encoder/decoder trained to
  reconstruct normal images; the per-image anomaly score is the reconstruction
  error. Dependency-light, fully trainable, no pretrained weights.
- **PatchCore** — a frozen ImageNet backbone extracts mid-level patch features
  over the normal train set; a greedy-coreset subsample is stored as a memory
  bank. The per-image anomaly score is the max over patches of the distance to
  the nearest memory-bank feature. No gradient training; "fit" is one pass +
  subsample. Uses torch ``cdist`` (no faiss). See PHASE9_ANOMALY_PLAN.md.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from visionforge.utils.anomaly_config import AnomalyModelConfig


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder; reconstructs the (normalized) input image.

    The decoder realigns to the input resolution with bilinear interpolation so
    arbitrary (even non-power-of-two) input sizes round-trip to the same shape.
    """

    def __init__(self, latent_dim: int = 512, in_channels: int = 3) -> None:
        super().__init__()

        def down(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(i, o, 4, stride=2, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
            )

        def up(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose2d(i, o, 4, stride=2, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
            )

        self.encoder = nn.Sequential(
            down(in_channels, 32),
            down(32, 64),
            down(64, 128),
            nn.Conv2d(128, latent_dim, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 128, 1),
            nn.ReLU(inplace=True),
            up(128, 64),
            up(64, 32),
            up(32, 32),
            nn.Conv2d(32, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon: torch.Tensor = self.decoder(z)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(
                recon, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return recon


class PatchCore(nn.Module):
    """Memory-bank anomaly model over a frozen backbone's patch features."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        coreset_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self._coreset_ratio = coreset_ratio
        net = cast(tv_models.ResNet, self._build_backbone(backbone, pretrained))
        # Stem through layer2 + layer3 are the standard PatchCore taps.
        self._stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool, net.layer1
        )
        self._layer2 = net.layer2
        self._layer3 = net.layer3
        for p in self.parameters():
            p.requires_grad_(False)
        self.feature_dim = self._infer_feature_dim()
        self.register_buffer("_memory", torch.empty(0, self.feature_dim))

    @staticmethod
    def _build_backbone(name: str, pretrained: bool) -> nn.Module:
        builders = {
            "resnet18": tv_models.resnet18,
            "resnet34": tv_models.resnet34,
            "resnet50": tv_models.resnet50,
            "wide_resnet50_2": tv_models.wide_resnet50_2,
        }
        weights = "DEFAULT" if pretrained else None
        return cast(nn.Module, builders[name](weights=weights))

    def _infer_feature_dim(self) -> int:
        with torch.no_grad():
            feats = self._forward_features(torch.zeros(1, 3, 64, 64))
        return int(feats.shape[-1])

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``[B, P, C]`` patch features (layer2 ⊕ layer3, aligned)."""
        h = self._stem(x)
        f2 = self._layer2(h)
        f3 = self._layer3(f2)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        cat = torch.cat([f2, f3], dim=1)  # [B, C, H, W]
        b, c, hh, ww = cat.shape
        return cat.permute(0, 2, 3, 1).reshape(b, hh * ww, c)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``[B, P, C]`` patch features for a batch."""
        return self._forward_features(x)

    @property
    def is_fitted(self) -> bool:
        return bool(self._memory.numel() > 0)

    @property
    def memory_size(self) -> int:
        return int(self._memory.shape[0])

    def fit(self, features: torch.Tensor) -> None:
        """Build the memory bank from ``[M, C]`` normal patch features.

        Greedy k-center coreset keeps ``coreset_ratio`` of the patches (≥ 1).
        """
        m = features.shape[0]
        k = max(1, int(round(self._coreset_ratio * m)))
        memory = self._greedy_coreset(features, k)
        self._memory = memory

    @staticmethod
    def _greedy_coreset(features: torch.Tensor, k: int) -> torch.Tensor:
        m = features.shape[0]
        if k >= m:
            return features.clone()
        selected = [0]
        min_dist = torch.cdist(features, features[0:1]).squeeze(1)
        for _ in range(1, k):
            idx = int(torch.argmax(min_dist).item())
            selected.append(idx)
            d = torch.cdist(features, features[idx : idx + 1]).squeeze(1)
            min_dist = torch.minimum(min_dist, d)
        return features[selected].clone()

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``[B]`` image-level anomaly scores (max patch distance)."""
        if not self.is_fitted:
            raise RuntimeError("PatchCore.score called before fit().")
        feats = self._forward_features(x)  # [B, P, C]
        b, p, _ = feats.shape
        flat = feats.reshape(b * p, -1)
        dists = torch.cdist(flat, self._memory)  # [B*P, K]
        nn_dist = dists.min(dim=1).values.reshape(b, p)
        return nn_dist.max(dim=1).values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.score(x)


class AnomalyModelFactory:
    """Instantiates an anomaly model from an AnomalyModelConfig."""

    @staticmethod
    def create(config: AnomalyModelConfig) -> nn.Module:
        """Build the autoencoder or PatchCore model."""
        if config.name == "autoencoder":
            return ConvAutoencoder(latent_dim=config.latent_dim)
        if config.name == "patchcore":
            return PatchCore(
                backbone=config.backbone,
                pretrained=config.pretrained,
                coreset_ratio=config.coreset_ratio,
            )
        raise ValueError(f"Unknown anomaly model: {config.name}")


__all__ = ["AnomalyModelFactory", "ConvAutoencoder", "PatchCore"]
