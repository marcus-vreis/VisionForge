"""Third-party run exporters, discovered through the ``visionforge.exporters`` entry-point (ADR-244).

The built-in exporters (MLflow, W&B, Comet, Zenodo, GitHub, DVC, Hugging Face Hub) keep
their own routes. This module is the way for a separate package to add one more
without touching VisionForge: it declares an object that follows :class:`RunExporter`
under the ``visionforge.exporters`` entry-point group, and ``visionforge export`` runs
it. A plugin that cannot be loaded is skipped with a warning, never fatal.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol, runtime_checkable

from loguru import logger

from visionforge.utils.messages import tr

ENTRY_POINT_GROUP = "visionforge.exporters"
RUN_RECORD = "run.json"


@dataclass(frozen=True)
class ExportResult:
    """Where an exporter put the run, and anything the user should know about it."""

    destination: str
    notes: tuple[str, ...] = ()


@runtime_checkable
class RunExporter(Protocol):
    """What an exporter plugin provides: a name, a description and ``export``."""

    name: str
    description: str

    def export(self, run_dir: Path, options: Mapping[str, str]) -> ExportResult:
        """Send the run in ``run_dir`` somewhere and say where."""
        ...


def _follows_protocol(obj: object) -> bool:
    """True when ``obj`` has a non-empty ``name``, a ``description`` and a callable ``export``."""
    name = getattr(obj, "name", None)
    return (
        isinstance(name, str)
        and bool(name)
        and isinstance(getattr(obj, "description", None), str)
        and callable(getattr(obj, "export", None))
    )


def discover_exporters() -> dict[str, RunExporter]:
    """Load every exporter declared under ``visionforge.exporters``, keyed by ``name``.

    An entry-point is a class (instantiated without arguments) or an instance. One that
    fails to load, to instantiate or to follow the protocol is logged and skipped; on a
    duplicate name the first one found stays.
    """
    found: dict[str, RunExporter] = {}
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        try:
            loaded = entry.load()
            candidate = loaded() if isinstance(loaded, type) else loaded
        except Exception as exc:  # a plugin may raise anything
            logger.warning(tr("exporters.load_failed", entry=entry.name, error=exc))
            continue
        if not _follows_protocol(candidate):
            logger.warning(tr("exporters.bad_protocol", entry=entry.name))
            continue
        if candidate.name in found:
            logger.warning(
                tr("exporters.duplicate", name=candidate.name, entry=entry.name)
            )
            continue
        found[candidate.name] = candidate
    return found


def run_exporter(
    exporter: RunExporter, run_dir: Path, options: Mapping[str, str]
) -> ExportResult:
    """Call ``exporter`` on ``run_dir`` after checking that it holds a ``run.json``."""
    if not (run_dir / RUN_RECORD).is_file():
        raise ValueError(tr("exporters.no_run_json", run_dir=run_dir))
    return exporter.export(run_dir, options)
