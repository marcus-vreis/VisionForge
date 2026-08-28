"""Tell the researcher when a run produced a number instead of a result.

A model that predicts one class for every image still reports an accuracy, and
on a balanced two-class problem that accuracy is 0.50 — a value indistinguishable
at a glance from "trained a bit, did not learn much". It is not the same thing:
nothing was learned, and the number is the class prior.

This was found by training the real grid rather than reading the code (ADR-099):
VGG16 and AlexNet with Adam at the default learning rate of 1e-3 collapse to a
single class, while the same architectures with SGD reach 0.80. Neither the log
nor the report said anything was wrong. The defaults are now nudged per
architecture, but a default can always be overridden into the same hole, so the
detection is the part that has to exist.

Every task family can collapse in its own way, and they share one shape:
**the model stopped distinguishing its inputs**. Classification predicts one
class, regression predicts one value, segmentation paints every pixel the same,
detection finds nothing at all. Each check below is that same question asked in
the vocabulary of one task.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

# A loss that falls by less than this fraction over the whole run is, for
# reporting purposes, a loss that did not move. Calibrated on the measured grid
# (ADR-099) rather than picked: the runs that learned nothing fell 3.7%
# (VGG16+Adam, 1.446 -> 1.393) and 4.4% (resnet50+SGD, 1.372 -> 1.312), while
# the ones that learned fell 49% (VGG16+SGD) and 62% (resnet50+Adam). Ten
# percent sits in the empty middle of that gap.
#
# A run resumed from an already-converged checkpoint can legitimately fall less
# than this, which is why the result is a warning to read the metrics with,
# never a failure.
_STAGNANT_LOSS_RATIO = 0.10


@dataclass(frozen=True)
class HealthWarning:
    """One thing about a finished run the researcher needs to be told."""

    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """JSON-ready form for run.json."""
        return asdict(self)


def collapsed_predictions(
    predictions: Sequence[int], n_classes: int
) -> HealthWarning | None:
    """Warn when every prediction is the same class.

    The accuracy of such a model is the frequency of whichever class it picked,
    which is why a balanced binary problem reports exactly 0.50.
    """
    if len(predictions) == 0 or n_classes < 2:
        return None
    distinct = {int(p) for p in predictions}
    if len(distinct) > 1:
        return None
    only = distinct.pop()
    return HealthWarning(
        code="collapsed_predictions",
        message=(
            f"O modelo previu a mesma classe ({only}) para todas as "
            f"{len(predictions)} imagens de validação — ele não aprendeu a "
            f"distinguir as classes. A acurácia mostrada é apenas a proporção "
            f"dessa classe. Causa mais comum: learning rate alto demais para "
            f"esta arquitetura (VGG e AlexNet com Adam costumam precisar de "
            f"1e-4, não 1e-3). Tente reduzir o learning rate ou trocar o "
            f"otimizador para SGD."
        ),
    )


def stagnant_loss(losses: Sequence[float]) -> HealthWarning | None:
    """Warn when the training loss never meaningfully moved."""
    values = [float(v) for v in losses if v is not None]
    if len(values) < 2:
        return None
    first, last = values[0], min(values)
    if first <= 0:
        return None
    if (first - last) / abs(first) > _STAGNANT_LOSS_RATIO:
        return None
    return HealthWarning(
        code="stagnant_loss",
        message=(
            f"A loss de treino praticamente não caiu ({first:.4f} → {last:.4f} "
            f"em {len(values)} épocas). O modelo não está aprendendo: revise o "
            f"learning rate (alto demais diverge, baixo demais não sai do "
            f"lugar) e confira se os rótulos do dataset estão corretos."
        ),
    )


def constant_predictions(
    values: Sequence[float], *, label: str = "valor"
) -> HealthWarning | None:
    """Warn when a regressor outputs essentially one number for every input."""
    array = np.asarray([float(v) for v in values], dtype=float).ravel()
    if array.size < 2:
        return None
    spread = float(array.std())
    scale = max(abs(float(array.mean())), 1e-9)
    if spread / scale > 1e-3:
        return None
    return HealthWarning(
        code="constant_predictions",
        message=(
            f"O modelo previu praticamente o mesmo {label} "
            f"({float(array.mean()):.4f}) para todas as entradas — ele está "
            f"chutando a média em vez de usar a imagem. Revise o learning rate "
            f"e a normalização dos alvos."
        ),
    )


def frozen_random_backbone(
    *, pretrained: bool, mode: str | None, frozen_params: int
) -> HealthWarning | None:
    """Warn when feature extraction is freezing weights that were never trained.

    Feature extraction is worth doing because the frozen weights already know
    something. Applied to a randomly initialised network it freezes noise and
    trains only the head against it — a unet without pretrained weights leaves
    31.4M random parameters fixed and 195 trainable (ADR-101).
    """
    if mode != "feature_extraction" or pretrained:
        return None
    return HealthWarning(
        code="frozen_random_backbone",
        message=(
            f"Feature extraction congelou {frozen_params:,} pesos que nunca "
            f"foram treinados (o modelo está sem pesos pré-treinados). Congelar "
            f"faz sentido quando os pesos já aprenderam algo; aqui eles são "
            f"aleatórios. Ative os pesos pré-treinados, ou use fine-tuning para "
            f"treinar a rede inteira."
        ).replace(",", "."),
    )


def collapsed_segmentation(
    per_class_iou: Sequence[float], *, present_classes: int
) -> HealthWarning | None:
    """Warn when the segmenter painted every pixel the same class.

    The giveaway is one class with a real IoU and the rest at zero: the mask is
    uniform, and the mIoU that gets reported is that one class divided by the
    number of classes.
    """
    values = [float(v) for v in per_class_iou]
    if len(values) < 2 or present_classes < 2:
        return None
    scored = [v for v in values if v > 0.01]
    if len(scored) > 1:
        return None
    return HealthWarning(
        code="collapsed_segmentation",
        message=(
            "A segmentação previu uma única classe em todos os pixels — as "
            "outras ficaram com IoU zero. O mIoU mostrado é o dessa classe "
            "dividido pelo número de classes, não uma medida de qualidade. "
            "Revise o learning rate e verifique se as máscaras têm os índices "
            "de classe esperados."
        ),
    )


def no_detections(map50: float | None, epochs_done: int) -> HealthWarning | None:
    """Warn when a detector finished without finding anything.

    mAP is zero both when a model predicts nothing and when everything it
    predicts is wrong; either way there is no result to read.
    """
    if epochs_done < 1 or map50 is None:
        return None
    if map50 > 0.001:
        return None
    return HealthWarning(
        code="no_detections",
        message=(
            f"O modelo terminou {epochs_done} época(s) com mAP@50 igual a zero: "
            f"ele não acertou nenhuma caixa. Confira se os rótulos estão no "
            f"formato YOLO esperado e se o número de classes bate com o "
            f"data.yaml; depois disso, o learning rate."
        ),
    )


def summarize(warnings: Sequence[HealthWarning | None]) -> list[dict[str, str]]:
    """Drop the Nones and hand back what run.json should carry."""
    return [w.to_dict() for w in warnings if w is not None]


__all__ = [
    "HealthWarning",
    "collapsed_predictions",
    "collapsed_segmentation",
    "frozen_random_backbone",
    "no_detections",
    "constant_predictions",
    "stagnant_loss",
    "summarize",
]
