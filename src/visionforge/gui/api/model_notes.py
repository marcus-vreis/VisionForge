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
from visionforge.utils.messages import tr

# The grid behind every number in learning_rate.py used four classes.
_MEASURED_CLASSES = 4


def collapse_note(
    architecture: str,
    optimizer: str,
    learning_rate: float,
    evidence: CollapseEvidence | None,
) -> str:
    """Note for a collapse-prone architecture, in the language of the request.

    Args:
        architecture: the model the researcher chose.
        optimizer: the suggested optimizer.
        learning_rate: the suggested learning rate.
        evidence: what was measured for this family, or ``None`` if nothing was.

    Returns:
        A sentence that cites only the measurement ``evidence`` carries.
    """
    if evidence is None:
        return tr(
            "model.collapse_unmeasured",
            architecture=architecture,
            optimizer=optimizer,
            rate=learning_rate,
        )

    accuracy = tr(
        "model.accuracy_on_classes",
        accuracy=evidence.accuracy,
        classes=_MEASURED_CLASSES,
    )
    failure = tr(
        "model.failure_collapse"
        if evidence.outcome == "collapse"
        else "model.failure_learned_little"
    )
    measured_itself = architecture.lower() == evidence.measured_on
    if measured_itself:
        measured = tr(
            "model.measured_itself",
            architecture=architecture,
            failure=failure,
            accuracy=accuracy,
        )
    else:
        measured = tr(
            "model.measured_sibling",
            architecture=architecture,
            measured_on=evidence.measured_on,
            failure=failure,
            accuracy=accuracy,
        )
    if measured_itself and evidence.recovered_accuracy is not None:
        remedy = tr(
            "model.remedy_measured",
            optimizer=optimizer,
            rate=learning_rate,
            recovered=evidence.recovered_accuracy,
        )
    else:
        remedy = tr("model.remedy_suggested", optimizer=optimizer, rate=learning_rate)
    return f"{measured} {remedy}"
