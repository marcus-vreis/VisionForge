"""A training resolution that suits the images actually in the dataset.

The default of 224 comes from ImageNet, and it is the right guess when nothing
is known about the data. Once the data is on disk, something *is* known, and the
guess can be wrong in a way that costs real time: CIFAR-10 ships 32x32 images,
so training at 224 upscales every one of them by seven, spending roughly fifty
times the computation to look at pixels that were invented by the resize
(ADR-100).

This suggests, and never imposes. Two reasons it must not decide by itself:

- **Pretrained weights carry an expected scale.** ImageNet features were learned
  at ~224, and feeding 32 to a pretrained ResNet works but discards much of what
  those weights know. Smaller is cheaper, not automatically better.
- **Attention models require the size they were trained at.** `vit_b_16` and
  `swin_t` build fixed position embeddings; a different input is not slower, it
  is an error. For those the answer is always 224, whatever the dataset holds.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PIL import Image

# Architectures whose input size is part of the checkpoint, not a preference.
_FIXED_INPUT = ("vit", "swin", "maxvit")
_FIXED_SIZE = 224

# What ImageNet-pretrained convolutional weights expect; also the ceiling for a
# suggestion, since going above it costs computation without adding detail the
# weights can use.
_PRETRAINED_SCALE = 224

# Below this, a resize destroys more than it saves.
_FLOOR = 64

# Sizes are rounded to this, because most architectures downsample by 32 and a
# ragged input silently pads.
_STRIDE = 32

_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def median_image_side(root: Path, *, sample: int = 200) -> int | None:
    """The median of the shorter side across a sample of the dataset's images.

    The shorter side is what a square resize is bound by, so it is the number
    that decides how much real detail survives. Returns None when nothing
    readable is found — a caller with no measurement should keep its default
    rather than invent one.
    """
    sides: list[int] = []
    for path in _iter_images(root, sample):
        try:
            with Image.open(path) as img:
                sides.append(min(img.width, img.height))
        except Exception:  # noqa: BLE001 - an unreadable file is not fatal here
            continue
    if not sides:
        return None
    sides.sort()
    return sides[len(sides) // 2]


def _iter_images(root: Path, limit: int) -> Iterable[Path]:
    seen = 0
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in _EXTENSIONS:
            continue
        yield path
        seen += 1
        if seen >= limit:
            return


def suggested_image_size(
    root: Path, architecture: str = "", *, pretrained: bool = True
) -> int | None:
    """A training size for this dataset, or None when the default should stand.

    Args:
        root: dataset directory to sample.
        architecture: model name; attention families pin the answer to 224.
        pretrained: whether ImageNet weights are being used, which sets the
            scale those features expect.
    """
    arch = (architecture or "").lower()
    if any(arch.startswith(prefix) for prefix in _FIXED_INPUT):
        return _FIXED_SIZE

    median = median_image_side(root)
    if median is None:
        return None

    ceiling = _PRETRAINED_SCALE if pretrained else max(median, _FLOOR)
    target = min(median, ceiling)
    rounded = max(_FLOOR, int(round(target / _STRIDE)) * _STRIDE)
    return rounded


__all__ = ["median_image_side", "suggested_image_size"]
