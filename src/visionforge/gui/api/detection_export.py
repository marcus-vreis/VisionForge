"""ONNX export for detection runs (Ultralytics-native).

The detection analogue of ``ExportONNXBlock`` (classification). Ultralytics ships
its own ONNX exporter (``YOLO(...).export(format="onnx")``), so we drive that for
the primary backend. The torchvision detection export path is intentionally not
supported yet (its ONNX tracing needs model-specific handling); it raises a clear
error so the GUI can keep the action hidden for those runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from visionforge.gui.api.schemas import ExportOnnxRequest, ExportOnnxResponse


def export_detection_run(
    run_dir: Path, req: ExportOnnxRequest, data: dict[str, Any]
) -> ExportOnnxResponse:
    """Export a detection run's checkpoint to ONNX.

    Raises:
        FileNotFoundError: if the run has no usable checkpoint.
        ValueError: for the (unsupported) torchvision backend.
        RuntimeError: if the ultralytics extra is not installed.
    """
    config: dict[str, Any] = data.get("config", {})
    backend = (config.get("model") or {}).get("backend", "ultralytics")

    checkpoint = data.get("artifacts", {}).get("model")
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError(
            f"Run '{run_dir.name}' não tem um checkpoint utilizável "
            f"(artifacts.model: {checkpoint!r})."
        )

    if backend != "ultralytics":
        raise ValueError(
            "Export ONNX para detectores torchvision ainda não é suportado; "
            "use o backend Ultralytics."
        )

    from visionforge.core import detection_trainer

    if detection_trainer.YOLO is None:
        raise RuntimeError(
            "ultralytics is not installed. Install the detection extra: "
            "pip install 'visionforge-studio[detection]'."
        )

    model = detection_trainer.YOLO(str(checkpoint))
    exported = Path(
        model.export(format="onnx", opset=req.opset_version, dynamic=req.dynamic_axes)
    )

    # Honour an explicit output path by moving the Ultralytics output there.
    if req.output_onnx:
        target = Path(req.output_onnx)
        target.parent.mkdir(parents=True, exist_ok=True)
        if exported.resolve() != target.resolve():
            exported.replace(target)
        exported = target

    # Record the exported artifact on the run so it shows up next time.
    data.setdefault("artifacts", {})["onnx"] = str(exported.resolve())
    (run_dir / "run.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    size = exported.stat().st_size if exported.is_file() else 0
    return ExportOnnxResponse(
        output_onnx=str(exported.resolve()),
        file_size_bytes=size,
        validation=None,
        benchmark=None,
    )


__all__ = ["export_detection_run"]
