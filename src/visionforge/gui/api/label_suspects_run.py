"""Suspicious labels of a classification run's dataset (ADR-237).

The run's checkpoint predicts every image of each split through the batch
prediction block (the same inference as ``/runs/{id}/batch_predict``); the
per-split CSVs land in ``<run_dir>/label_suspects/`` and the report next to
them, so the screen can show it again without predicting twice.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from visionforge.core.label_suspects import find_suspects, read_folder_predictions
from visionforge.gui.api.config_scope import confine_config
from visionforge.gui.api.dataset_scope import current_scope
from visionforge.gui.api.datasets_routes import (
    SCAN_BASE_MISSING_KEY,
    _remember_dataset_root,
)
from visionforge.gui.api.profiles import Profile
from visionforge.gui.api.run_history import RUN_FILENAME
from visionforge.gui.api.run_history import run_task as _run_task
from visionforge.utils.config import ExperimentConfig
from visionforge.utils.messages import tr

OUTPUT_DIR = "label_suspects"
REPORT_FILE = "report.json"
SPLITS = ("train", "val", "test")


class LabelSuspectsRequest(BaseModel):
    """Which splits to predict; training's is in-sample, so it is optional."""

    splits: list[str] = Field(default_factory=lambda: ["val", "test"], min_length=1)


def saved_report(run_dir: Path) -> dict[str, Any] | None:
    """The last report of this run, or None if it was never computed.

    Raises:
        ValueError: the report is there but cannot be read (cut short while
            it was written, or edited by hand).
    """
    path = run_dir / OUTPUT_DIR / REPORT_FILE
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(tr("suspects.report_unreadable", run_id=run_dir.name)) from exc
    if not isinstance(report, dict):
        raise ValueError(tr("suspects.report_unreadable", run_id=run_dir.name))
    return report


def confine_stored_config(run_dir: Path, profile: Profile) -> None:
    """On a server, read the run's stored config again like a new one.

    The folders a server shares may have changed since the run trained, as
    ``/resume`` assumes (ADR-218): the dataset is predicted only while it is
    still in the person's scope. On a local install this does nothing.

    Raises:
        HTTPException: 400 when the stored config no longer validates; 422
            when it names something outside the scope.
    """
    if current_scope() is None:
        return
    try:
        data = json.loads((run_dir / RUN_FILENAME).read_text(encoding="utf-8"))
        config = ExperimentConfig.model_validate(data["config"])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            400, tr("run.stored_config_invalid", run_id=run_dir.name, error=exc)
        ) from exc
    confine_config(profile.scope_model(config), profile)


def execute_label_suspects(run_dir: Path, req: LabelSuspectsRequest) -> dict[str, Any]:
    """Predict each requested split with the run's model and find the suspects.

    Raises:
        ValueError: the run is not a classification run, has no class names or
            no checkpoint, or names a split that is not train, val or test.
        FileNotFoundError: the dataset folder is gone.
    """
    data: dict[str, Any] = json.loads(
        (run_dir / RUN_FILENAME).read_text(encoding="utf-8")
    )
    if _run_task(data) != "classification":
        raise ValueError(tr("suspects.classification_only"))
    unknown = [s for s in req.splits if s not in SPLITS]
    if unknown:
        raise ValueError(tr("suspects.unknown_split", splits=", ".join(unknown)))
    class_names = [str(n) for n in data.get("class_names") or []]
    checkpoint = data.get("artifacts", {}).get("model")
    if len(class_names) < 2 or not checkpoint:
        raise ValueError(tr("suspects.no_classes"))
    config: dict[str, Any] = data.get("config") or {}
    base_dir = (config.get("data") or {}).get("base_dir")
    if not base_dir:
        raise ValueError(tr("suspects.no_base_dir"))
    base = Path(base_dir)
    if not base.is_dir():
        raise FileNotFoundError(tr(SCAN_BASE_MISSING_KEY, path=base))
    _remember_dataset_root(base)
    out = run_dir / OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    splits: dict[str, Any] = {}
    for split in req.splits:
        folder = base / str(config["data"].get(f"{split}_dir", split))
        if folder.is_dir():
            splits[split] = _split_report(
                folder, split, out, config, checkpoint, class_names
            )
    report = {
        "run_dir": str(run_dir),
        "computed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "class_names": class_names,
        "splits": splits,
        "missing_splits": [s for s in req.splits if s not in splits],
    }
    (out / REPORT_FILE).write_text(json.dumps(report), encoding="utf-8")
    return report


def _split_report(
    folder: Path,
    split: str,
    out: Path,
    config: dict[str, Any],
    checkpoint: str,
    class_names: list[str],
) -> dict[str, Any]:
    from visionforge.blocks.batch_prediction import BatchPredictionBlock

    csv_path = out / f"{split}.csv"
    block = BatchPredictionBlock()
    block.setup(
        ExperimentConfig.model_validate(
            {
                **config,
                "block": "batch_prediction",
                "batch_prediction": {
                    "checkpoint_path": checkpoint,
                    "input_dir": str(folder.resolve()),
                    "output_csv": str(csv_path.resolve()),
                    "recursive": True,
                    "class_names": class_names,
                },
            }
        )
    )
    block.run()
    failed = list(block.report().get("failed_files", []))
    paths, probs, labels = read_folder_predictions(csv_path, folder, class_names)
    found = find_suspects(probs, labels, class_names) if len(paths) else None
    if found is not None:
        for item in found["suspects"]:
            item["path"] = paths[item.pop("index")]
    return {
        "folder": str(folder),
        "images": len(paths),
        "failed": len(failed),
        "in_sample": split == "train",
        **(
            found
            or {
                "suspect_count": 0,
                "suspects": [],
                "pairs": [],
                "joint": [],
                "thresholds": [],
            }
        ),
    }
