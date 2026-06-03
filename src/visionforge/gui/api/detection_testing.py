"""Per-model test for detection runs: evaluate a saved checkpoint on a new set.

The detection analogue of ``routes._execute_run_test`` (classification): it loads
the run's best checkpoint, runs it over a new YOLO dataset, computes mAP@0.5, and
appends the result to ``run.json``'s ``tests[]`` (ADR-013 test-history contract).

Backend split mirrors the trainer (ADR-033/034): the torchvision path rebuilds
the detector + DataLoader and reuses ``detection_metrics`` (CPU, dependency-light);
the Ultralytics path drives ``YOLO(...).val`` (patchable module global, like the
trainer) and reads mAP off its results.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from visionforge.gui.api.schemas import RunTestRequest, RunTestResponse
from visionforge.utils.detection_config import DetectionConfig

# Preferred evaluation split order: a held-out split first, train last.
_EVAL_SPLITS = ("val", "test", "train")


def evaluate_detection_run(
    run_dir: Path, req: RunTestRequest, data: dict[str, Any]
) -> RunTestResponse:
    """Evaluate a detection run's checkpoint on ``req.base_dir`` and record mAP.

    Raises:
        FileNotFoundError: if the run has no usable checkpoint on disk.
        ValueError: if the new dataset has no resolvable YOLO split.
    """
    config = _config_for_new_dataset(data["config"], req.base_dir)

    checkpoint = data.get("artifacts", {}).get("model")
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError(
            f"Run '{run_dir.name}' não tem um checkpoint utilizável "
            f"(artifacts.model: {checkpoint!r})."
        )

    if config.model.backend == "torchvision":
        metrics = _evaluate_torchvision(config, Path(checkpoint), req)
    else:
        metrics = _evaluate_ultralytics(config, Path(checkpoint), run_dir)

    timestamp = datetime.now()
    test_id = f"test_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
    label = req.label or Path(req.base_dir).name or "test"
    base_dir_str = str(Path(req.base_dir).resolve())
    record: dict[str, Any] = {
        "test_id": test_id,
        "label": label,
        "base_dir": base_dir_str,
        "timestamp": timestamp.isoformat(),
        "metrics": metrics,
        "artifacts": {},
    }
    data.setdefault("tests", []).append(record)
    (run_dir / "run.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    return RunTestResponse(
        test_id=test_id,
        run_id=run_dir.name,
        label=label,
        base_dir=base_dir_str,
        timestamp=timestamp,
        metrics=metrics,
        artifacts={},
    )


def _config_for_new_dataset(
    config_dict: dict[str, Any], base_dir: str
) -> DetectionConfig:
    """Clone the run's config but pointed at the new dataset root."""
    cloned = dict(config_dict)
    cloned["data"] = {
        **(config_dict.get("data") or {}),
        "base_dir": base_dir,
        "data_yaml": None,  # force base_dir resolution against the new set
    }
    return DetectionConfig.model_validate(cloned)


def _evaluate_torchvision(
    config: DetectionConfig, checkpoint: Path, req: RunTestRequest
) -> dict[str, Any]:
    """Rebuild the detector, load the checkpoint, and compute mAP@0.5 (CPU)."""
    import torch
    from torch.utils.data import DataLoader

    from visionforge.core.detection_data import resolve_yolo_split
    from visionforge.core.detection_dataset import DetectionDataset, detection_collate
    from visionforge.core.detection_metrics import mean_average_precision_50
    from visionforge.models.detection_factory import build_torchvision_detector

    base = Path(req.base_dir)
    resolved = next(
        (r for s in _EVAL_SPLITS if (r := resolve_yolo_split(base, s)) is not None),
        None,
    )
    if resolved is None:
        raise ValueError(
            f"Nenhum split YOLO encontrado em '{base}' "
            f"(esperado 'images/<split>' ou '<split>/images')."
        )
    images_dir, labels_dir = resolved
    loader = DataLoader(
        DetectionDataset(images_dir, labels_dir),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate,
    )

    model = build_torchvision_detector(
        config.model.name, config.model.num_classes, pretrained=False
    )
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    preds: list[dict[str, torch.Tensor]] = []
    gts: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for images, targets in loader:
            for out in model(list(images)):
                preds.append({k: v.detach().cpu() for k, v in out.items()})
            for t in targets:
                gts.append({"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()})

    return {"map50": mean_average_precision_50(preds, gts).map50}


def _evaluate_ultralytics(
    config: DetectionConfig, checkpoint: Path, run_dir: Path
) -> dict[str, Any]:
    """Drive ``YOLO(checkpoint).val`` over a synthesized data.yaml and read mAP."""
    from visionforge.core import detection_trainer
    from visionforge.core.detection_data import DetectionDataModule

    if detection_trainer.YOLO is None:
        raise RuntimeError(
            "ultralytics is not installed. Install the detection extra: "
            "pip install 'visionforge[detection]'."
        )

    data_yaml = DetectionDataModule(config).resolve_data_yaml(out_dir=run_dir / "tests")
    model = detection_trainer.YOLO(str(checkpoint))
    results = model.val(data=str(data_yaml), verbose=False)
    box = results.box
    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }


__all__ = ["evaluate_detection_run"]
