from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from loguru import logger

from visionforge.blocks.base import ExperimentBlock
from visionforge.models.factory import ModelFactory
from visionforge.utils.config import ExperimentConfig


class ExportONNXBlock(ExperimentBlock):
    """Export a trained checkpoint to ONNX, validate outputs, and benchmark latency."""

    def setup(self, config: ExperimentConfig) -> None:
        """Prepare model and capture image size from data config.

        Raises:
            ValueError: if export_onnx config is absent.
        """
        if config.export_onnx is None:
            raise ValueError(
                "ExportONNXBlock requires export_onnx to be set in ExperimentConfig."
            )
        self._config = config
        self._onnx_cfg = config.export_onnx
        self._image_size = config.data.transforms.image_size

        model = ModelFactory.create(config.model)
        state_dict = torch.load(
            str(self._onnx_cfg.checkpoint_path),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)  # type: ignore[arg-type]
        model.eval()
        self._model: nn.Module = model

        self._report_data: dict[str, Any] = {}

    def run(self) -> None:
        """Export to ONNX, optionally validate and benchmark."""
        dummy = torch.zeros(1, 3, self._image_size, self._image_size)

        self._export(dummy)

        validation_result: dict[str, Any] | None = None
        if self._onnx_cfg.run_validate:
            validation_result = self._validate(dummy)

        benchmark_result: dict[str, Any] | None = None
        if self._onnx_cfg.benchmark:
            benchmark_result = self._benchmark()

        file_size = self._onnx_cfg.output_onnx.stat().st_size
        self._report_data = {
            "file_size_bytes": file_size,
            "validation": validation_result,
            "benchmark": benchmark_result,
        }
        logger.info(
            "ONNX export complete: {} ({} bytes)", self._onnx_cfg.output_onnx, file_size
        )

    def report(self) -> dict[str, Any]:
        """Return export summary with file size, validation, and benchmark results."""
        return self._report_data

    # ── private ───────────────────────────────────────────────────────────────

    def _export(self, dummy: torch.Tensor) -> None:
        """Write the ONNX file, with optional dynamic batch axis."""
        cfg = self._onnx_cfg
        cfg.output_onnx.parent.mkdir(parents=True, exist_ok=True)

        dynamic_axes: dict[str, dict[int, str]] | None = None
        if cfg.dynamic_axes:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }

        with torch.no_grad():
            # dynamo=False forces the legacy TorchScript-based exporter, which works
            # without onnxscript and is stable for classification models.
            # args must be a tuple in torch 2.9+ even for single-input models.
            torch.onnx.export(
                self._model,
                (dummy,),
                str(cfg.output_onnx),
                opset_version=cfg.opset_version,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                dynamo=False,
            )

        logger.debug(
            "Exported ONNX to {} (opset {})", cfg.output_onnx, cfg.opset_version
        )

    def _validate(self, dummy: torch.Tensor) -> dict[str, Any]:
        """Compare PyTorch and ONNX Runtime outputs on the dummy input.

        Returns:
            Dict with keys 'passed' (bool) and 'max_abs_diff' (float).
        """
        with torch.no_grad():
            pt_out: np.ndarray = self._model(dummy).numpy()

        session = ort.InferenceSession(
            str(self._onnx_cfg.output_onnx),
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        ort_out: np.ndarray = session.run(None, {input_name: dummy.numpy()})[0]

        max_diff = float(np.max(np.abs(pt_out - ort_out)))
        passed = bool(
            np.allclose(pt_out, ort_out, atol=self._onnx_cfg.validation_tolerance)
        )

        result: dict[str, Any] = {"passed": passed, "max_abs_diff": max_diff}

        val_path = self._onnx_cfg.output_onnx.parent / "onnx_validation.json"
        val_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("ONNX validation: passed={}, max_abs_diff={:.6e}", passed, max_diff)

        return result

    def _benchmark(self) -> dict[str, Any]:
        """Time onnxruntime inference for benchmark_runs passes (first 3 are warmup).

        Returns:
            Dict with 'mean_ms', 'p50_ms', 'p95_ms'.
        """
        cfg = self._onnx_cfg
        dummy_np = np.zeros(
            (1, 3, self._image_size, self._image_size), dtype=np.float32
        )

        session = ort.InferenceSession(
            str(cfg.output_onnx),
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name

        # Warmup passes are excluded so they don't skew mean/percentiles.
        warmup = 3
        latencies_ms: list[float] = []
        for i in range(cfg.benchmark_runs + warmup):
            t0 = time.perf_counter()
            session.run(None, {input_name: dummy_np})
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if i >= warmup:
                latencies_ms.append(elapsed_ms)

        arr = np.array(latencies_ms)
        result: dict[str, Any] = {
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "runs": cfg.benchmark_runs,
        }

        bench_path = cfg.output_onnx.parent / "onnx_benchmark.json"
        bench_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info(
            "ONNX benchmark: mean={:.2f}ms p50={:.2f}ms p95={:.2f}ms",
            result["mean_ms"],
            result["p50_ms"],
            result["p95_ms"],
        )

        return result


__all__ = ["ExportONNXBlock"]
