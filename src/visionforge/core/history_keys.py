"""Read the per-epoch history of a saved run under today's names (ADR-241).

Anomaly scored every epoch on the test split under the name ``val_auroc``; it
is ``test_auroc`` now. A ``run.json`` (or a resume snapshot) written before the
rename still says ``val_auroc``, so every reader that plots, compares, exports
or resumes a history passes it through here first. This is the one place the
old name is known on the server.

Only anomaly runs are renamed: a researcher's own task (ADR-058) streams
``val_<metric>`` for a metric it measured on a validation split, so a custom
``val_auroc`` means what it says and is left alone.
"""

from __future__ import annotations

from typing import Any

# Old history key -> current key, for anomaly runs.
_ANOMALY_RENAMED = {"val_auroc": "test_auroc"}


def upgrade_anomaly_epoch(record: dict[str, Any]) -> dict[str, Any]:
    """One anomaly epoch record under the current names; the input is untouched.

    A record that already carries the new name keeps its value.
    """
    out = dict(record)
    for old, new in _ANOMALY_RENAMED.items():
        if old in out:
            value = out.pop(old)
            out.setdefault(new, value)
    return out


def _is_anomaly_run(data: dict[str, Any]) -> bool:
    top = data.get("task")
    if isinstance(top, str) and top.startswith("custom:"):
        return False
    config = data.get("config")
    return isinstance(config, dict) and config.get("task") == "anomaly"


def upgrade_run_history(data: dict[str, Any]) -> dict[str, Any]:
    """A ``run.json`` record whose ``history`` uses the current names.

    Returns ``data`` itself when there is nothing to rename (not an anomaly
    run, or no history list), otherwise a shallow copy with a new history.
    Entries that are not dicts are kept as they are.
    """
    history = data.get("history")
    if not isinstance(history, list) or not _is_anomaly_run(data):
        return data
    return {
        **data,
        "history": [
            upgrade_anomaly_epoch(row) if isinstance(row, dict) else row
            for row in history
        ],
    }


__all__ = ["upgrade_anomaly_epoch", "upgrade_run_history"]
