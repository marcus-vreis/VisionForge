"""Check the recorded dataset before continuing an existing experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from visionforge.core.dataset_fingerprint import fingerprint_from_config, same_dataset


def validate_resume_dataset(run_dir: Path, config: Any) -> None:
    """Refuse a confirmed dataset change and warn when identity is unavailable."""
    report_path = run_dir / "run.json"
    if not report_path.is_file():
        logger.warning("Resume cannot verify dataset identity: run.json is absent.")
        return
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            "Cannot validate resume dataset: run.json is unreadable."
        ) from exc
    if not isinstance(report, dict):
        raise ValueError("Cannot validate resume dataset: run.json must be an object.")
    recorded = report.get("dataset_fingerprint")
    if not isinstance(recorded, dict) or recorded.get("method") not in (
        "manifest",
        "content",
    ):
        logger.warning(
            "Resume cannot verify dataset identity: fingerprint is unavailable."
        )
        return
    root = Path(config.data.base_dir).resolve()
    output = getattr(config, "output", None)
    for name in ("models_dir", "graphics_dir", "logs_dir", "reports_dir"):
        directory = getattr(output, name, None)
        if directory is not None and Path(directory).resolve().is_relative_to(root):
            # The recorded root manifest includes the run's own changing files.
            logger.warning(
                "Resume cannot verify dataset identity: output directories are "
                "inside the dataset root. Keep datasets and outputs separate."
            )
            return
    method: Literal["manifest", "content"] = (
        "content" if recorded["method"] == "content" else "manifest"
    )
    current = fingerprint_from_config(config, method=method)
    matches = same_dataset(recorded, current)
    if matches is False:
        raise ValueError(
            "Cannot resume: dataset has changed since the recorded run. "
            "Start a new experiment to preserve the original results."
        )
    if matches is None:
        logger.warning(
            "Resume cannot verify dataset identity: fingerprints are not comparable."
        )
