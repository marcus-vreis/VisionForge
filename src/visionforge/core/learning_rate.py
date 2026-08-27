"""A starting learning rate that suits the architecture and the optimizer.

One default cannot serve both halves of the model list, and the grid that
produced this says so plainly (ADR-099, 4 classes, 3 epochs, same data and seed
throughout):

| model           | Adam 1e-3        | SGD 1e-3         |
|-----------------|------------------|------------------|
| resnet50        | 0.72             | 0.53 (undertrained) |
| vgg16           | **0.25 collapse**| 0.80             |
| alexnet         | **0.25 collapse**| 0.81             |
| efficientnet_b1 | 0.86             | 0.34 (undertrained) |

The split follows batch normalization. VGG and AlexNet predate it and carry
huge fully-connected heads: an Adam step of 1e-3 saturates them in the first
iterations and they never recover, predicting one class for everything. The
normalized architectures tolerate that step and instead suffer under SGD, where
1e-3 without momentum barely moves them.

So the suggestion is a function of both. It is a *starting point* offered in the
interface, never a value forced onto a config the researcher wrote — the whole
failure mode this addresses is a number appearing without the user's knowledge.
"""

from __future__ import annotations

# Architectures without batch normalization, which need a gentler Adam step.
_UNNORMALIZED = ("vgg", "alexnet")

# Measured starting points. SGD here is plain SGD (no momentum), which is what
# the classification trainer builds.
_ADAM_DEFAULT = 1e-3
_ADAM_UNNORMALIZED = 1e-4
_SGD_DEFAULT = 1e-2


def suggested_learning_rate(architecture: str, optimizer: str) -> float:
    """A learning rate that trains this pair, instead of collapsing it.

    Args:
        architecture: model name, e.g. ``resnet50`` or ``vgg16``.
        optimizer: ``adam``, ``adamw`` or ``sgd``.

    Returns:
        The suggested starting learning rate.
    """
    arch = (architecture or "").lower()
    opt = (optimizer or "").lower()
    if opt == "sgd":
        return _SGD_DEFAULT
    if any(arch.startswith(prefix) for prefix in _UNNORMALIZED):
        return _ADAM_UNNORMALIZED
    return _ADAM_DEFAULT


def is_collapse_prone(architecture: str, optimizer: str, learning_rate: float) -> bool:
    """Whether this exact trio is the one measured to collapse.

    Narrow on purpose: it answers "have we watched this fail?", not "might this
    be suboptimal". A warning that fires on merely unusual settings is a warning
    people learn to dismiss.
    """
    arch = (architecture or "").lower()
    opt = (optimizer or "").lower()
    if opt not in ("adam", "adamw"):
        return False
    if not any(arch.startswith(prefix) for prefix in _UNNORMALIZED):
        return False
    return learning_rate > _ADAM_UNNORMALIZED


__all__ = ["is_collapse_prone", "suggested_learning_rate"]
