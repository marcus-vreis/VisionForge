"""The sentence that explains why a model's Adam 1e-3 default is flagged.

It states what was measured and nothing past it (ADR-099, ADR-100). The
measurement is one model per family at Adam 1e-3, on 4 classes, so the note names
that model, says how it failed, and says plainly when the model being asked
about is not the one that was run. A family that was never run gets no number
at all, only the suggestion.

Every number here comes from classification runs. The route does not know the
task, so this note (which the interface does not show; it words its own from the
response's numbers) is the classification one.

The suggested setting is worded the same way. "It trains normally" is a claim,
so it is made only as a number, only for the model whose recovery was run on the
same setup (``recovered_accuracy``); everywhere else the setting is just
suggested.
"""

from __future__ import annotations

from visionforge.core.learning_rate import CollapseEvidence

# The grid behind every number in learning_rate.py used four classes.
_MEASURED_CLASSES = 4


def collapse_note(
    architecture: str,
    optimizer: str,
    learning_rate: float,
    evidence: CollapseEvidence | None,
) -> str:
    """Portuguese note for a collapse-prone architecture.

    Args:
        architecture: the model the researcher chose.
        optimizer: the suggested optimizer.
        learning_rate: the suggested learning rate.
        evidence: what was measured for this family, or ``None`` if nothing was.

    Returns:
        A sentence that cites only the measurement ``evidence`` carries.
    """
    if evidence is None:
        return (
            f"{architecture}: Adam a 1e-3 não foi medido para esta família; "
            f"sugerimos {optimizer} a {learning_rate:g}, o mesmo das famílias "
            f"de atenção medidas."
        )

    accuracy = f"acurácia {evidence.accuracy:.2f} em {_MEASURED_CLASSES} classes"
    failure = (
        "previu uma classe só" if evidence.outcome == "collapse" else "aprendeu pouco"
    )
    measured_itself = architecture.lower() == evidence.measured_on
    if measured_itself:
        measured = f"{architecture} {failure} com Adam a 1e-3 ({accuracy})."
    else:
        measured = (
            f"{architecture}: o {evidence.measured_on}, da mesma família, {failure} "
            f"com Adam a 1e-3 ({accuracy}); este modelo não foi medido."
        )
    if measured_itself and evidence.recovered_accuracy is not None:
        remedy = (
            f"Com {optimizer} a {learning_rate:g}, a acurácia foi "
            f"{evidence.recovered_accuracy:.2f} nas mesmas condições."
        )
    else:
        remedy = f"Sugerimos {optimizer} a {learning_rate:g}."
    return f"{measured} {remedy}"
