"""Background jobs that merge two datasets into a new one (ADR-167).

Shares its machinery with the incorporate jobs
(:mod:`visionforge.gui.api.derived_jobs`); this module says what a merge preview
and apply run and report. The preview keeps its :class:`MergeAnalysis` so that
changing a row of the class mapping re-plans without hashing both datasets again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from visionforge.core.dataset_explorer import ExplorerTask
from visionforge.core.dataset_merge import (
    SIDES,
    MergeAnalysis,
    MergePlan,
    analyze_merge,
    build_merged_dataset,
)
from visionforge.gui.api.derived_jobs import DerivedJob, DerivedJobs

MergeJob = DerivedJob


def preview_result(analysis: MergeAnalysis, suggested_name: str) -> dict[str, Any]:
    """What the page shows once both datasets were read."""
    return {
        "suggested_name": suggested_name,
        "task": analysis.task,
        "a_dir": str(analysis.roots["a"]),
        "b_dir": str(analysis.roots["b"]),
        "classes": {
            side: [{"name": c.name, "images": c.images} for c in analysis.classes[side]]
            for side in SIDES
        },
        "present": analysis.present,
        "found": analysis.found,
        "repeated_within": analysis.repeated_within,
        "similar_names": analysis.similar_names,
    }


class MergeJobs(DerivedJobs):
    """Registry of the latest merge preview and apply jobs."""

    def __init__(self) -> None:
        super().__init__("merge")

    def start_preview(
        self,
        a_dir: Path,
        task: ExplorerTask,
        b_dir: Path,
        a_split_dirs: dict[str, str],
        b_split_dirs: dict[str, str],
        suggested_name: str,
    ) -> MergeJob:
        """Launch the read-and-hash pass.

        Raises:
            JobBusyError: if another job is still running.
        """
        job = self.register("preview")

        def run() -> None:
            analysis = analyze_merge(
                a_dir,
                task,
                b_dir,
                a_split_dirs,
                b_split_dirs,
                token=job.token,
                on_progress=job.progress,
            )
            if analysis is None:
                job.state = "cancelled"
                return
            job.analysis = analysis
            job.result = preview_result(analysis, suggested_name)
            job.state = "done"

        self.spawn(job, run)
        return job

    def start_apply(
        self, analysis: MergeAnalysis, plan: MergePlan, dest_name: str
    ) -> MergeJob:
        """Launch the write.

        Raises:
            JobBusyError: if another job is still running.
        """
        job = self.register("apply")

        def run() -> None:
            result = build_merged_dataset(
                analysis, plan, dest_name, token=job.token, on_progress=job.progress
            )
            if result is None:
                job.state = "cancelled"
                return
            registry = result["registry"]
            job.result = {
                "path": result["path"],
                "images": result["images"],
                "files_linked": registry["files_linked"],
                "files_copied": registry["files_copied"],
                "classes": registry["classes"],
            }
            job.state = "done"

        self.spawn(job, run)
        return job
