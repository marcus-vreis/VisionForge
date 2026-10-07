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
| convnext_tiny | **0.25 collapse** | **0.91**   |

Swin and ConvNeXt collapse outright at 1e-3 and ViT merely fails to learn, so
all are fine-tuned at 1e-4 — the rate their papers use, now confirmed here
rather than taken on faith.

Everything above was measured on *one model per family* (vgg16, not vgg11 or
vgg19; vit_b_16, not vit_l_16). ``maxvit`` shares the remedy because it is an
attention family, but it was never run, so it carries no evidence below. What
the interface may say about a model is exactly what ``collapse_evidence``
returns for it — the numbers in this table, keyed by family, and nothing more.

The *recovery* (the AdamW 1e-4 column) is evidence only where it was run on this
same four-class setup: vit_b_16, swin_t and convnext_tiny. VGG16 at 1e-4 was run
once, on a two-class problem (ADR-099: 0.50 collapsed, 0.88 at 1e-4), which is a
different measurement and is not quoted next to a four-class one; AlexNet at
1e-4 was never run. For those, and for every sibling, the rate is a suggestion
and nothing is claimed about how it trains.

So the suggestion is a function of both. It is a *starting point* offered in the
interface, never a value forced onto a config the researcher wrote — the whole
failure mode this addresses is a number appearing without the user's knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Architectures that need a gentler adaptive step: the pre-BatchNorm CNNs and
# the attention families, for different reasons but with the same remedy.
_UNNORMALIZED = ("vgg", "alexnet")
_ATTENTION = ("vit", "swin", "convnext", "maxvit")

# Suggested starting points. The docstring tables say which of them were
# measured; the rest follow the family's remedy. SGD here is plain SGD (no
# momentum), which is what the classification trainer builds.
_ADAM_DEFAULT = 1e-3
_ADAM_UNNORMALIZED = 1e-4
_ADAM_ATTENTION = 1e-4
_SGD_DEFAULT = 1e-2


@dataclass(frozen=True)
class CollapseEvidence:
    """What was actually observed for one family at Adam 1e-3 (ADR-099/100).

    Attributes:
        measured_on: the single model that was run. Its siblings (vgg11,
            vit_l_16, ...) share the family but were not measured.
        accuracy: the accuracy that run reached, on 4 classes after 3 epochs.
        outcome: ``collapse`` when it predicted one class for every image,
            ``fails_to_learn`` when it merely stayed near chance without
            collapsing to a single class.
        recovered_accuracy: the accuracy the same model reached on the same
            data at the suggested setting, or ``None`` when that run was never
            made on this setup. ``None`` is not "bad": it means no claim.
    """

    measured_on: str
    accuracy: float
    outcome: Literal["collapse", "fails_to_learn"]
    recovered_accuracy: float | None = None


# One entry per family prefix, mirroring the tables in the module docstring.
# ViT is the odd one out: 0.41 on four classes is above the 0.25 of a one-class
# prediction, so it failed to learn rather than collapsed. ``maxvit`` has no
# entry on purpose: it was never measured. ``recovered_accuracy`` is the AdamW
# 1e-4 column, left out for vgg16 (only run on two classes) and alexnet (never).
_EVIDENCE: dict[str, CollapseEvidence] = {
    "vgg": CollapseEvidence("vgg16", 0.25, "collapse"),
    "alexnet": CollapseEvidence("alexnet", 0.25, "collapse"),
    "swin": CollapseEvidence("swin_t", 0.25, "collapse", recovered_accuracy=0.88),
    "convnext": CollapseEvidence(
        "convnext_tiny", 0.25, "collapse", recovered_accuracy=0.91
    ),
    "vit": CollapseEvidence(
        "vit_b_16", 0.41, "fails_to_learn", recovered_accuracy=0.85
    ),
}


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


def collapse_evidence(architecture: str) -> CollapseEvidence | None:
    """What was measured for this architecture's family at Adam 1e-3, if anything.

    ``None`` means the family is either not collapse-prone or was never run, and
    in both cases a caller has no number to quote.
    """
    arch = (architecture or "").lower()
    for prefix, evidence in _EVIDENCE.items():
        if arch.startswith(prefix):
            return evidence
    return None


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


__all__ = [
    "CollapseEvidence",
    "collapse_evidence",
    "is_collapse_prone",
    "suggested_learning_rate",
    "suggested_optimizer",
]
