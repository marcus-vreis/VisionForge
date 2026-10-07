"""The sentence that explains why a model's Adam 1e-3 default is flagged.

It states what was measured and nothing past it (ADR-099, ADR-100). The
measurement is one model per family at Adam 1e-3, on 4 classes, so the note names
that model, says how it failed, and says plainly when the model being asked
about is not the one that was run. A family that was never run gets no number
at all, only the suggestion.
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
        "previu uma classe só" if evidence.outcome == "collapse" else "não aprendeu"
    )
    if architecture.lower() == evidence.measured_on:
        measured = f"{architecture} {failure} com Adam a 1e-3 ({accuracy})."
    else:
        measured = (
            f"{architecture}: o {evidence.measured_on}, da mesma família, {failure} "
            f"com Adam a 1e-3 ({accuracy}); este modelo não foi medido."
        )
    return f"{measured} Com {optimizer} a {learning_rate:g} treina normal."
