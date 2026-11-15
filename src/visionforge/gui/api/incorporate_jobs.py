"""Background jobs that incorporate new images into a dataset (ADR-166).

The machinery (registry, progress, cancellation, one job at a time) is shared with
the merge jobs in :mod:`visionforge.gui.api.derived_jobs`; this module only says
what an incorporate preview and apply run and report.
"""

from __future__ import annotations

from pathlib import Path

from visionforge.core.dataset_explorer import ExplorerTask
from visionforge.core.dataset_incorporate import (
    Analysis,
    Plan,
    analyze_incorporation,
    build_incorporated_dataset,
)
from visionforge.gui.api.derived_jobs import DerivedJob, DerivedJobs, JobBusyError

IncorporateJob = DerivedJob


class IncorporateJobs(DerivedJobs):
    """Registry of the latest incorporate preview and apply jobs."""

    def __init__(self) -> None:
        super().__init__("incorporate")

    def start_preview(
        self,
        base: Path,
        task: ExplorerTask,
        source: Path,
        split_dirs: dict[str, str],
        suggested_name: str,
    ) -> IncorporateJob:
        """Launch the read-and-compare pass.

        Raises:
            JobBusyError: if another job is still running.
        """
        job = self.register("preview")

        def run() -> None:
            analysis = analyze_incorporation(
                base,
                task,
                source,
                split_dirs,
                token=job.token,
                on_progress=job.progress,
            )
            if analysis is None:
                job.state = "cancelled"
                return
            job.analysis = analysis
            job.result = {
                "suggested_name": suggested_name,
                "task": analysis.task,
                "base_dir": str(analysis.base_root),
                "source_dir": str(analysis.source_root),
                "base_classes": analysis.base_classes,
                "base_counts": analysis.base_counts,
                "present": analysis.present,
                "source_images": analysis.source_images,
                "new_classes": [
                    {"name": c.name, "images": c.images, "source_id": c.source_id}
                    for c in analysis.new_classes
                ],
            }
            job.state = "done"

        self.spawn(job, run)
        return job

    def start_apply(
        self, analysis: Analysis, plan: Plan, dest_name: str
    ) -> IncorporateJob:
        """Launch the write.

        Raises:
            JobBusyError: if another job is still running.
        """
        job = self.register("apply")

        def run() -> None:
            result = build_incorporated_dataset(
                analysis, plan, dest_name, token=job.token, on_progress=job.progress
            )
            if result is None:
                job.state = "cancelled"
                return
            registry = result["registry"]
            job.result = {
                "path": result["path"],
                "added": result["added"],
                "files_linked": registry["files_linked"],
                "files_copied": registry["files_copied"],
                "classes": registry["classes"],
            }
            job.state = "done"

        self.spawn(job, run)
        return job


__all__ = ["IncorporateJob", "IncorporateJobs", "JobBusyError"]
