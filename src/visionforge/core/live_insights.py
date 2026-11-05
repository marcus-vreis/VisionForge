"""Grad-CAM of a few fixed validation images while a classifier trains (ADR-162).

The training must not notice it. A run with these insights and the same run
without them have the same history, number for number, which is what lets a run
that watched itself still be the run a paper reports. Four rules keep it so:

1. no random number generator moves: the samples are chosen once with a
   generator of their own, and every pass runs inside a fork of the global ones;
2. no ``.grad`` of a parameter is written: the gradient is taken with
   ``torch.autograd.grad`` with respect to the activation only;
3. the model goes back to the mode it was in, and runs in ``eval`` meanwhile, so
   BatchNorm statistics and dropout are untouched;
4. a failure is a warning, never the end of the run.
"""

from __future__ import annotations

import random
import shutil
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from visionforge.core.gradcam import overlay_cam, resolve_target_layer

# How many validation images are explained. Enough to see whether the model looks
# at the object, few enough to cost a handful of forward passes per epoch.
LIVE_GRADCAM_SAMPLES = 6


@contextmanager
def preserved_rng(device: torch.device) -> Iterator[None]:
    """Leave Python, NumPy and torch (CPU and the run's GPU) generators as found."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


def choose_samples(labels: Sequence[int], k: int, seed: int) -> list[int]:
    """``k`` dataset indices, the classes taken in turn so each one shows.

    Drawn once with a generator of its own, so the choice is the same on every
    run of the same seed and costs the training's generators nothing.
    """
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(labels), generator=gen).tolist()
    by_class: dict[int, list[int]] = {}
    for index in order:
        by_class.setdefault(int(labels[index]), []).append(index)
    queues = [by_class[c] for c in sorted(by_class)]
    chosen: list[int] = []
    while len(chosen) < k and any(queues):
        for queue in queues:
            if queue and len(chosen) < k:
                chosen.append(queue.pop(0))
    return chosen


def gradcam_batch(
    model: nn.Module, target_layer: nn.Module, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grad-CAM maps ``[B, H, W]`` in [0, 1] and the predicted classes ``[B]``.

    ``images`` is ``[B, 3, H, W]``, normalized as the model was trained. The score
    of each image is the logit of its predicted class; a single-logit (binary)
    model predicts class 1 when the logit is positive, and its score for class 0
    is the negated logit. The gradient is taken with respect to the activation
    only, so no parameter's ``.grad`` changes (rule 2 above).
    """
    captured: list[torch.Tensor] = []
    handle = target_layer.register_forward_hook(
        lambda _m, _i, output: captured.append(output)
    )
    try:
        # The input requires grad so a frozen backbone (feature extraction)
        # still puts the activation in the graph.
        inputs = images.clone().requires_grad_(True)
        output = model(inputs)
    finally:
        handle.remove()
    activation = captured[-1]
    if output.shape[1] == 1:
        logit = output[:, 0]
        predicted = (logit > 0).long()
        score = torch.where(predicted == 1, logit, -logit).sum()
    else:
        predicted = output.argmax(dim=1)
        score = output.gather(1, predicted[:, None]).sum()
    (grad,) = torch.autograd.grad(score, activation)
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1))
    cam = cam - cam.amin(dim=(1, 2), keepdim=True)
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    cam = F.interpolate(
        cam.unsqueeze(1), size=images.shape[-2:], mode="bilinear", align_corners=False
    ).squeeze(1)
    return cam.detach(), predicted.detach()


def _denormalized(
    image: torch.Tensor, mean: Sequence[float], std: Sequence[float]
) -> torch.Tensor:
    """``[3, H, W]`` normalized → ``[3, H, W]`` in [0, 1], with the run's own stats."""
    m = torch.tensor(mean, dtype=image.dtype).view(3, 1, 1)
    s = torch.tensor(std, dtype=image.dtype).view(3, 1, 1)
    return (image.detach().cpu() * s + m).clamp(0.0, 1.0)


class LiveGradCam:
    """Writes the Grad-CAM of fixed validation samples and reports it.

    Built with :meth:`create`, which returns ``None`` when the option is off or
    the validation set cannot be read sample by sample.
    """

    def __init__(
        self,
        every: int,
        total_epochs: int,
        run_dir: Path,
        samples: list[tuple[torch.Tensor, int]],
        class_names: Sequence[str],
        normalization: tuple[Sequence[float], Sequence[float]],
        device: torch.device,
    ) -> None:
        self._every = every
        self._total_epochs = total_epochs
        self._dir = run_dir / "insights"
        self._samples = samples
        self._class_names = list(class_names)
        self._mean, self._std = normalization
        self._device = device
        self._best = float("inf")
        self._failed = False

    @classmethod
    def create(
        cls,
        config: Any,
        val_loader: Any,
        class_names: Sequence[str],
        run_dir: Path,
        device: torch.device,
        best_so_far: float = float("inf"),
    ) -> LiveGradCam | None:
        every = int(config.training.live_gradcam_every)
        if every <= 0:
            return None
        # Read by index, never by iterating the loader: a persistent-worker
        # loader would advance its own generator.
        dataset = getattr(val_loader, "dataset", None)
        labels = getattr(dataset, "targets", None)
        if dataset is None or labels is None or len(labels) == 0:
            logger.warning(
                "Live Grad-CAM is off: the validation set has no labels to choose from."
            )
            return None
        with preserved_rng(device):
            indices = choose_samples(labels, LIVE_GRADCAM_SAMPLES, config.training.seed)
            samples = [(dataset[i][0], int(dataset[i][1])) for i in indices]
        transforms = config.data.transforms
        insights = cls(
            every,
            config.training.epochs,
            run_dir,
            samples,
            class_names,
            (transforms.normalize_mean, transforms.normalize_std),
            device,
        )
        # A resumed run carries its best so far: the next epoch is not a new
        # best just because the counter started over.
        insights._best = best_so_far
        return insights

    def due(self, epoch: int, is_best: bool) -> bool:
        """Every ``every`` epochs, the last one, and each new best."""
        return epoch % self._every == 0 or epoch == self._total_epochs or is_best

    def after_epoch(
        self,
        model: nn.Module,
        epoch: int,
        val_loss: float,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        """Explain the samples when the epoch is due, and report the images.

        ``is_best`` follows the classifier's checkpoint rule: the lowest
        validation loss, replaced only on a strict gain.
        """
        is_best = val_loss < self._best
        if is_best:
            self._best = val_loss
        if self._failed or not self.due(epoch, is_best):
            return
        try:
            images = self._write(model, epoch, is_best)
        except Exception as exc:  # rule 4: the run goes on
            self._failed = True
            logger.warning(
                "Live Grad-CAM failed at epoch {} and is off for this run: {}",
                epoch,
                exc,
            )
            return
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "insight",
                    "kind": "gradcam",
                    "epoch": epoch,
                    "best": is_best,
                    "images": images,
                }
            )

    def _write(
        self, model: nn.Module, epoch: int, is_best: bool
    ) -> list[dict[str, Any]]:
        net = model.module if isinstance(model, nn.DataParallel) else model
        was_training = net.training
        batch = torch.stack([image for image, _ in self._samples]).to(self._device)
        with preserved_rng(self._device):
            net.eval()
            try:
                with torch.enable_grad():
                    cams, predicted = gradcam_batch(
                        net, resolve_target_layer(net), batch
                    )
            finally:
                net.train(was_training)
        latest = self._dir / "latest"
        latest.mkdir(parents=True, exist_ok=True)
        images: list[dict[str, Any]] = []
        for i, ((image, label), cam, pred) in enumerate(
            zip(self._samples, cams, predicted.tolist(), strict=True)
        ):
            path = latest / f"{i}.png"
            base = _denormalized(image, self._mean, self._std)
            overlay_cam(base, cam, normalized=False).save(path)
            images.append(
                {
                    "path": path.as_posix(),
                    "true_class": self._name(label),
                    "predicted_class": self._name(int(pred)),
                }
            )
        if is_best:
            shutil.copytree(latest, self._dir / "best", dirs_exist_ok=True)
        logger.info("Live Grad-CAM written for epoch {}.", epoch)
        return images

    def _name(self, index: int) -> str:
        return (
            self._class_names[index]
            if 0 <= index < len(self._class_names)
            else str(index)
        )


__all__ = [
    "LIVE_GRADCAM_SAMPLES",
    "LiveGradCam",
    "choose_samples",
    "gradcam_batch",
    "preserved_rng",
]
