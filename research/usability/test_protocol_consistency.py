"""The protocol, the runbook and the session sheet record the same fields (#229)."""

from __future__ import annotations

import csv
import re
from pathlib import Path

HERE = Path(__file__).parent
OUTCOMES = {"completed", "assisted", "failed", "withdrawn"}


def _text(name: str) -> str:
    return (HERE / name).read_text(encoding="utf-8")


def _outcomes_named(text: str) -> set[str]:
    return set(re.findall(r"`(completed|assisted|failed|withdrawn)`", text))


def test_protocol_and_runbook_define_the_same_outcomes() -> None:
    assert _outcomes_named(_text("PROTOCOL.md")) == OUTCOMES
    assert _outcomes_named(_text("RUNBOOK.md")) == OUTCOMES


def test_the_session_sheet_has_the_fields_the_protocol_relies_on() -> None:
    with (HERE / "session_template.csv").open(encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh))
    assert {"outcome", "elapsed_seconds", "assistance_count"} <= set(header)


def test_help_means_a_counted_intervention_everywhere() -> None:
    # "assisted" is 1 or more interventions in the protocol and the runbook,
    # and the form points its "com ajuda" at the same definition.
    assert "1 or more" in _text("PROTOCOL.md")
    assert "1 ou mais" in _text("RUNBOOK.md")
    assert "Facilitation and timing" in _text("FORM.md")


def test_the_limitations_are_written_down() -> None:
    protocol = _text("PROTOCOL.md")
    for limitation in (
        "Practice effect",
        "Facilitator is the author",
        "No control group",
    ):
        assert limitation in protocol
