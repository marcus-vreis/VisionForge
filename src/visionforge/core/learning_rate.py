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

The attention-based families were measured the same way when they were added
(ADR-100), and they behave like the first group:

| model         | Adam 1e-3         | AdamW 1e-4 |
|---------------|-------------------|------------|
| vit_b_16      | 0.41              | 0.85       |
| swin_t        | **0.25 collapse** | 0.88       |

Swin collapses outright at 1e-3 and ViT merely fails to learn, so both are
fine-tuned at 1e-4 — the rate their papers use, now confirmed here rather than
taken on faith.

So the suggestion is a function of both. It is a *starting point* offered in the
interface, never a value forced onto a config the researcher wrote — the whole
failure mode this addresses is a number appearing without the user's knowledge.
"""

from __future__ import annotations

# Architectures that need a gentler adaptive step: the pre-BatchNorm CNNs and
# the attention families, for different reasons but with the same remedy.
_UNNORMALIZED = ("vgg", "alexnet")
_ATTENTION = ("vit", "swin", "convnext", "maxvit")

# Measured starting points. SGD here is plain SGD (no momentum), which is what
# the classification trainer builds.
_ADAM_DEFAULT = 1e-3
_ADAM_UNNORMALIZED = 1e-4
_ADAM_ATTENTION = 1e-4
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
    if any(arch.startswith(prefix) for prefix in _ATTENTION):
        return _ADAM_ATTENTION
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
    fragile = _UNNORMALIZED + _ATTENTION
    if not any(arch.startswith(prefix) for prefix in fragile):
        return False
    return learning_rate > _ADAM_UNNORMALIZED


def suggested_optimizer(architecture: str) -> str:
    """The optimizer these weights are normally fine-tuned with.

    Attention models are trained with decoupled weight decay in every paper
    that introduced them, and measured better here too (0.85 and 0.88 with
    AdamW against 0.41 and 0.25 with Adam).
    """
    arch = (architecture or "").lower()
    if any(arch.startswith(prefix) for prefix in _ATTENTION):
        return "adamw"
    return "adam"


__all__ = ["is_collapse_prone", "suggested_learning_rate", "suggested_optimizer"]
