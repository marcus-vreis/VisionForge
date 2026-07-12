"""Example custom task — object counting. Copy as a starting point, or delete.

Registers ``example_counting``: a tiny CNN learns to count bright dots in
synthetic 32×32 images. Everything is generated on the fly (no dataset on
disk), so the task trains out of the box — open the GUI, pick the
"Contagem (exemplo)" tab, press Treinar. It exists to show the full ADR-058
surface working end to end:

    GET  /api/custom/example_counting/schema
    POST /api/custom/example_counting/run
    POST /api/custom/example_counting/sweep         e.g. {"max_count": [3, 6]}
    POST /api/custom/example_counting/replicates

Scaffold your own task with ``visionforge new-task <key>`` and see
user_tasks/README.md for the walkthrough.
"""

from __future__ import annotations

import torch
from pydantic import Field
from torch import nn
from torch.utils.data import DataLoader, Dataset

from visionforge.tasks import BaseTaskConfig, TaskSpec, register_task


class CountingConfig(BaseTaskConfig):
    """Task-specific knobs on top of BaseTaskConfig's shared blocks."""

    n_samples: int = Field(default=192, ge=16, description="Synthetic images per split")
    max_count: int = Field(
        default=5, ge=1, le=20, description="Maximum number of dots per image"
    )


# ADR-030 note: this class lives in a path-loaded task file, so DataLoader
# workers (spawn) cannot re-import it — the demo loaders below therefore use
# num_workers=0. Datasets needing workers must live in an importable package.
class DotsDataset(Dataset):
    """32×32 grayscale images, each with N bright 2×2 dots; target = N."""

    def __init__(self, n: int, max_count: int, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.images = torch.zeros(n, 1, 32, 32)
        self.counts = torch.randint(0, max_count + 1, (n, 1), generator=gen).float()
        for i in range(n):
            for _ in range(int(self.counts[i].item())):
                x = int(torch.randint(0, 30, (1,), generator=gen).item())
                y = int(torch.randint(0, 30, (1,), generator=gen).item())
                self.images[i, 0, y : y + 2, x : x + 2] = 1.0
        self.images += 0.05 * torch.randn(self.images.shape, generator=gen)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.counts[idx]


@register_task(
    key="example_counting",
    label="Contagem (exemplo)",
    accent="#2dd4bf",
    description="Conte objetos em imagens sintéticas — exemplo do SDK",
    metrics={"mae": "lower", "rmse": "lower"},
    primary_metric="mae",
)
class CountingTask(TaskSpec[CountingConfig]):
    """Level 1 example: four hooks, the generic engine drives the loop."""

    Config = CountingConfig

    def build_model(self, cfg: CountingConfig) -> nn.Module:
        """A small conv stack ending in a global pool and one linear output."""
        return nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
        )

    def build_loaders(self, cfg: CountingConfig):
        """Three synthetic splits, seeded from the run's training seed."""
        seed = cfg.training.seed
        n_eval = max(cfg.n_samples // 4, 16)

        def _loader(ds: DotsDataset, shuffle: bool) -> DataLoader:
            # num_workers=0: see the DotsDataset note above (ADR-030).
            return DataLoader(
                ds, batch_size=cfg.training.batch_size, shuffle=shuffle, num_workers=0
            )

        return (
            _loader(DotsDataset(cfg.n_samples, cfg.max_count, seed), shuffle=True),
            _loader(DotsDataset(n_eval, cfg.max_count, seed + 1), shuffle=False),
            _loader(DotsDataset(n_eval, cfg.max_count, seed + 2), shuffle=False),
        )

    def compute_loss(
        self, model: nn.Module, batch: tuple, cfg: CountingConfig
    ) -> torch.Tensor:
        device = next(model.parameters()).device
        images, counts = (t.to(device) for t in batch)
        return nn.functional.mse_loss(model(images), counts)

    def compute_metrics(
        self, model: nn.Module, loader: DataLoader, cfg: CountingConfig
    ) -> dict[str, float]:
        device = next(model.parameters()).device
        abs_err: list[torch.Tensor] = []
        with torch.no_grad():
            for images, counts in loader:
                pred = model(images.to(device))
                abs_err.append((pred - counts.to(device)).abs().flatten())
        errors = torch.cat(abs_err)
        return {
            "mae": errors.mean().item(),
            "rmse": errors.pow(2).mean().sqrt().item(),
        }
