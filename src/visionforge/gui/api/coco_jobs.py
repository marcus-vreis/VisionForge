"""Background jobs that convert a dataset between YOLO and COCO (ADR-169).

The machinery (registry, progress, cancellation, one job at a time) is shared with
the other derived-dataset jobs in :mod:`visionforge.gui.api.derived_jobs`. Export and
import share one registry, so only one conversion runs at a time (both are
disk-bound). The preview keeps its analysis in memory so that the plan (a split for
a JSON, a destination name) answers in milliseconds and the apply writes exactly
what the user saw.
"""

from __future__ import annotations

from pathlib import Path

from visionforge.core.dataset_coco import (
    ExportAnalysis,
    analyze_coco_export,
    build_coco_export,
)
from visionforge.core.dataset_coco_import import (
    ImportAnalysis,
    ImportPlan,
    analyze_coco_import,
    build_coco_import,
)
from visionforge.gui.api.derived_jobs import DerivedJob, DerivedJobs

_EXPORT_RESULT = ("path", "images", "boxes", "excluded", "files_linked", "files_copied")
_IMPORT_RESULT = ("path", "images", "boxes", "files_linked", "files_copied")


class CocoJobs(DerivedJobs):
    """Registry of the latest COCO conversion preview and apply jobs."""

    def __init__(self) -> None:
        super().__init__("coco")

    def start_export_preview(self, root: Path, suggested_name: str) -> DerivedJob:
        """Launch the read pass of an export.

        Raises:
            JobBusyError: if another job is still running.
        """
        job = self.register("preview")

        def run() -> None:
            analysis = analyze_coco_export(
                root, token=job.token, on_progress=job.progress
            )
            if analysis is None:
                job.state = "cancelled"
                return
            job.analysis = analysis
            job.result = {**analysis.summary(), "suggested_name": suggested_name}
            job.state = "done"

        self.spawn(job, run)
        return job

    def start_export_apply(
        self, analysis: ExportAnalysis, dest_name: str
    ) -> DerivedJob:
        """Launch the write of an export.

        Raises:
            JobBusyError: if another job is still running.
        """
        job = self.register("apply")

        def run() -> None:
            result = build_coco_export(
                analysis, dest_name, token=job.token, on_progress=job.progress
            )
            if result is None:
                job.state = "cancelled"
                return
            job.result = {key: result[key] for key in _EXPORT_RESULT}
            job.state = "done"

        self.spawn(job, run)
        return job

    def start_import_preview(self, root: Path, suggested_name: str) -> DerivedJob:
        """Launch the read-and-convert pass of an import.

        Raises:
            JobBusyError: if another job is still running.
        """
        job = self.register("preview")

        def run() -> None:
            analysis = analyze_coco_import(
                root, token=job.token, on_progress=job.progress
            )
            if analysis is None:
                job.state = "cancelled"
                return
            job.analysis = analysis
            job.result = {**analysis.summary(), "suggested_name": suggested_name}
            job.state = "done"

        self.spawn(job, run)
        return job

    def start_import_apply(
        self, analysis: ImportAnalysis, plan: ImportPlan, dest_name: str
    ) -> DerivedJob:
        """Launch the write of an import.

        Raises:
            JobBusyError: if another job is still running.
        """
        job = self.register("apply")

        def run() -> None:
            result = build_coco_import(
                analysis, plan, dest_name, token=job.token, on_progress=job.progress
            )
            if result is None:
                job.state = "cancelled"
                return
            job.result = {key: result[key] for key in _IMPORT_RESULT}
            job.result["classes"] = len(result["classes"])
            job.state = "done"

        self.spawn(job, run)
        return job


__all__ = ["CocoJobs"]
