"""Background jobs that re-split a dataset into a new one (ADR-168).

The machinery (registry, progress, cancellation, one job at a time) is shared with
the other derived-dataset jobs in :mod:`visionforge.gui.api.derived_jobs`; this
module says what a re-split preview and apply run and report. The preview keeps its
:class:`Analysis` so that changing a ratio, the seed or the lock re-plans without
hashing the dataset again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from visionforge.core.dataset_explorer import SPLIT_NAMES, ExplorerTask
from visionforge.core.dataset_resplit import (
    Analysis,
    Plan,
    analyze_resplit,
    build_resplit_dataset,
)
from visionforge.gui.api.derived_jobs import DerivedJob, DerivedJobs

ResplitJob = DerivedJob


def preview_result(analysis: Analysis, suggested_name: str) -> dict[str, Any]:
    """What the page shows once the dataset was read."""
    return {
        "suggested_name": suggested_name,
        "task": analysis.task,
        "base_dir": str(analysis.root),
        "classes": analysis.classes,
        "present": analysis.present,
        "images": len(analysis.items),
        "per_split": {
            s: sum(1 for i in analysis.items if i.split == s) for s in SPLIT_NAMES
        },
        "group_csv": analysis.group_csv,
        "group_count": analysis.group_count,
        "ungrouped": analysis.ungrouped,
        "unknown_entries": analysis.unknown_entries,
    }


class ResplitJobs(DerivedJobs):
    """Registry of the latest re-split preview and apply jobs."""

    def __init__(self) -> None:
        super().__init__("resplit")

    def start_preview(
        self,
        root: Path,
        task: ExplorerTask,
        split_dirs: dict[str, str],
        group_csv: str | None,
        suggested_name: str,
    ) -> ResplitJob:
        """Launch the read-and-hash pass.

        Raises:
            JobBusyError: if another job is still running.
        """
        job = self.register("preview")

        def run() -> None:
            analysis = analyze_resplit(
                root,
                task,
                split_dirs,
                group_csv,
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

    def start_apply(self, analysis: Analysis, plan: Plan, dest_name: str) -> ResplitJob:
        """Launch the write.

        Raises:
            JobBusyError: if another job is still running.
        """
        job = self.register("apply")

        def run() -> None:
            result = build_resplit_dataset(
                analysis, plan, dest_name, token=job.token, on_progress=job.progress
            )
            if result is None:
                job.state = "cancelled"
                return
            registry = result["registry"]
            job.result = {
                "path": result["path"],
                "images": result["images"],
                "moved": result["moved"],
                "files_linked": registry["files_linked"],
                "files_copied": registry["files_copied"],
            }
            job.state = "done"

        self.spawn(job, run)
        return job
