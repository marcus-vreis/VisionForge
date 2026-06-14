"""Export any image model to ONNX, with optional validation + latency benchmark.

Task-agnostic: operates on any ``nn.Module`` taking a ``(1, 3, H, W)`` float
input. Shared by classification (``ExportONNXBlock``) and the standalone-task
export endpoints — each task's factory builds the model; this module only knows
how to trace, validate and time it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from loguru import logger
from torch import nn


def export_to_onnx(
    model: nn.Module,
    image_size: int,
    output_onnx: Path,
    *,
    opset_version: int = 17,
    dynamic_axes: bool = True,
    validate: bool = True,
    benchmark: bool = True,
    benchmark_runs: int = 50,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Trace ``model`` to ONNX and return file size + optional validation/benchmark."""
    model.eval()
    output_onnx = Path(output_onnx)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, image_size, image_size)

    _export(model, dummy, output_onnx, opset_version, dynamic_axes)

    validation = _validate(model, dummy, output_onnx, tolerance) if validate else None
    bench = (
        _benchmark(model, image_size, output_onnx, benchmark_runs)
        if benchmark
        else None
    )

    file_size = output_onnx.stat().st_size
    logger.info("ONNX export complete: {} ({} bytes)", output_onnx, file_size)
    return {
        "file_size_bytes": file_size,
        "validation": validation,
        "benchmark": bench,
    }


def _export(
    model: nn.Module,
    dummy: torch.Tensor,
    output_onnx: Path,
    opset_version: int,
    dynamic_axes: bool,
) -> None:
    """Write the ONNX file, with an optional dynamic batch axis."""
    axes: dict[str, dict[int, str]] | None = None
    if dynamic_axes:
        axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    with torch.no_grad():
        # dynamo=False forces the legacy TorchScript exporter (stable, no
        # onnxscript dep); args must be a tuple in torch 2.9+.
        torch.onnx.export(
            model,
            (dummy,),
            str(output_onnx),
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=axes,
            dynamo=False,
        )


def _validate(
    model: nn.Module, dummy: torch.Tensor, output_onnx: Path, tolerance: float
) -> dict[str, Any]:
    """Compare PyTorch and ONNX Runtime outputs on the dummy input."""
    with torch.no_grad():
        pt_out: np.ndarray = model(dummy).numpy()

    session = ort.InferenceSession(str(output_onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ort_out: np.ndarray = session.run(None, {input_name: dummy.numpy()})[0]

    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    passed = bool(np.allclose(pt_out, ort_out, atol=tolerance))
    result: dict[str, Any] = {"passed": passed, "max_abs_diff": max_diff}

    (output_onnx.parent / "onnx_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    logger.info("ONNX validation: passed={}, max_abs_diff={:.6e}", passed, max_diff)
    return result


def _benchmark(
    model: nn.Module, image_size: int, output_onnx: Path, benchmark_runs: int
) -> dict[str, Any]:
    """Benchmark onnxruntime vs PyTorch latency on the same dummy input."""
    dummy_np = np.zeros((1, 3, image_size, image_size), dtype=np.float32)
    session = ort.InferenceSession(str(output_onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    warmup = 3
    latencies_ms: list[float] = []
    for i in range(benchmark_runs + warmup):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_np})
        if i >= warmup:
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    torch_mean_ms = _benchmark_torch(model, dummy_np, benchmark_runs, warmup)

    arr = np.array(latencies_ms)
    onnx_mean = float(arr.mean())
    result: dict[str, Any] = {
        "mean_ms": onnx_mean,
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "torch_mean_ms": torch_mean_ms,
        "speedup": (torch_mean_ms / onnx_mean) if onnx_mean > 0 else 0.0,
        "runs": benchmark_runs,
    }
    (output_onnx.parent / "onnx_benchmark.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    logger.info(
        "ONNX benchmark: onnx={:.2f}ms torch={:.2f}ms speedup={:.2f}x",
        result["mean_ms"],
        result["torch_mean_ms"],
        result["speedup"],
    )
    return result


def _benchmark_torch(
    model: nn.Module, dummy_np: np.ndarray, benchmark_runs: int, warmup: int
) -> float:
    """Mean PyTorch inference latency (ms) over ``benchmark_runs`` passes."""
    tensor = torch.from_numpy(dummy_np)
    latencies_ms: list[float] = []
    with torch.no_grad():
        for i in range(benchmark_runs + warmup):
            t0 = time.perf_counter()
            model(tensor)
            if i >= warmup:
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    return float(np.array(latencies_ms).mean())


__all__ = ["export_to_onnx"]
