"""Device controls without accelerator execution or training."""

import importlib
import os
import subprocess
import sys

import pytest
import torch

from research.workflow_benchmark import phase_contract as contract

MODULES = (
    "native_classification",
    "paired_classification",
    "native_segmentation",
    "paired_segmentation",
)


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize(
    "args,message",
    [
        (["--device", "invalid"], "invalid choice"),
        (["--device", "cuda"], "--parity-atol"),
        (["--device", "cuda", "--parity-atol", "0.001"], "CUDA is unavailable"),
    ],
)
def test_cli_rejects_before_output(tmp_path, module, args, message):
    output = tmp_path / "output"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "research.workflow_benchmark." + module,
            "missing-data",
            "missing-weights",
            str(output),
            *args,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
    )
    assert process.returncode != 0
    assert message in process.stderr
    assert not output.exists()


@pytest.mark.parametrize("module", MODULES)
def test_programmatic_cuda_preflight(tmp_path, monkeypatch, module):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runner = importlib.import_module("research.workflow_benchmark." + module)
    function = runner.run_pair if module.startswith("paired") else runner.run_native
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        function(
            tmp_path / "data",
            tmp_path / "weights",
            tmp_path / "output",
            seed=37,
            epochs=1,
            batch_size=2,
            device="cuda",
            parity_atol=0.001,
        )
    assert not (tmp_path / "output").exists()


def test_cuda_prepare_preserves_process(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        contract, "prepare_cpu_process", lambda: pytest.fail("CPU preparation on CUDA")
    )
    contract.prepare_process("cuda")
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_cpu_prepare_delegates(monkeypatch):
    calls = []
    monkeypatch.setattr(contract, "prepare_cpu_process", lambda: calls.append("cpu"))
    contract.prepare_process("cpu")
    assert calls == ["cpu"]


@pytest.mark.parametrize("failure", [False, True])
def test_cuda_phase_sync_peak(tmp_path, monkeypatch, failure):
    events = []
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: (
            events.append("sync") if device == 0 else pytest.fail("wrong GPU")
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: (
            events.append("reset") if device == 0 else pytest.fail("wrong GPU")
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda device: 1234 if device == 0 else pytest.fail("wrong GPU"),
    )
    monkeypatch.setattr(
        contract.time,
        "perf_counter",
        lambda: events.append("clock") or float(len(events)),
    )
    recorder = contract.PhaseRecorder(
        "test",
        tmp_path / "report.json",
        {p: [p] for p in contract.PHASES},
        {"device": "cuda"},
    )

    def run():
        with recorder.phase("setup"):
            events.append("work")
            if failure:
                raise ValueError("fault")

    if failure:
        with pytest.raises(contract.PhaseFailure):
            run()
    else:
        run()
    assert events == ["reset", "sync", "clock", "work", "sync", "clock"]
    assert recorder.report["phases"]["setup"]["peak_vram_bytes"] == 1234
    assert recorder.report["conditions"]["device"] == "cuda"


def test_weight_tolerance():
    a = {"w": torch.tensor([1.0]), "counter": torch.tensor(2)}
    b = {"w": torch.tensor([1.001]), "counter": torch.tensor(2)}
    equal, diff = contract.compare_weights(a, b, 0.002)
    assert equal
    assert diff == pytest.approx(0.001, abs=1e-7)
    assert contract.compare_weights(a, b, 0.0)[0] is False
    assert (
        contract.compare_weights(a, {**b, "counter": torch.tensor(3)}, 2.0)[0] is False
    )


@pytest.mark.parametrize("device,atol", [("cpu", None), ("cpu", 0.25)])
def test_cpu_tolerance_default(device, atol):
    assert contract.validate_device(device, atol) == (0.0 if atol is None else atol)


@pytest.mark.parametrize("atol", [-1.0, float("nan"), float("inf")])
def test_invalid_tolerance(atol):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        contract.validate_device("cpu", atol)


@pytest.mark.parametrize("module", MODULES)
def test_programmatic_requires_cuda_tolerance(tmp_path, module):
    runner = importlib.import_module("research.workflow_benchmark." + module)
    function = runner.run_pair if module.startswith("paired") else runner.run_native
    with pytest.raises(ValueError, match="--parity-atol"):
        function(
            tmp_path / "data",
            tmp_path / "weights",
            tmp_path / "output",
            seed=37,
            epochs=1,
            batch_size=2,
            device="cuda",
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("module", ["paired_classification", "paired_segmentation"])
def test_config_uses_device(tmp_path, module):
    runner = importlib.import_module("research.workflow_benchmark." + module)
    kwargs = {
        "seed": 37,
        "epochs": 1,
        "batch_size": 2,
        "image_size": 32,
        "learning_rate": 0.01,
    }
    if module == "paired_classification":
        kwargs["num_classes"] = 2
    config = runner._config(tmp_path, tmp_path / "output", device="cuda", **kwargs)
    assert config["device"] == {"kind": "cuda"}
    assert config["training"]["deterministic"] is True


def test_cuda_conditions(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda device: "mock GPU" if device == 0 else pytest.fail("wrong GPU"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(total_memory=8192),
    )
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 999)
    contract.validate_device("cuda", 0.001)
    conditions = contract.capture_conditions("test load", "cuda")
    assert conditions["device"] == "cuda"
    assert conditions["gpu_name"] == "mock GPU"
    assert conditions["gpu_total_memory_bytes"] == 8192
    assert conditions["cuda_version"] == torch.version.cuda
    assert conditions["cudnn_version"] == 999
    assert conditions["cudnn_benchmark"] is False
    assert conditions["cudnn_deterministic"] is True
    assert conditions["concurrent_workload"] == "test load"


def test_cpu_phase_never_touches_cuda(tmp_path, monkeypatch):
    def forbidden():
        pytest.fail("CUDA operation in CPU phase")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", forbidden)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", forbidden)
    recorder = contract.PhaseRecorder(
        "test",
        tmp_path / "report.json",
        {p: [p] for p in contract.PHASES},
        contract.capture_conditions("none"),
    )
    for phase in contract.PHASES:
        with recorder.phase(phase):
            pass
    report = recorder.complete({"device": "cpu"})
    assert report["device"] == report["conditions"]["device"] == "cpu"
    assert all("peak_vram_bytes" not in entry for entry in report["phases"].values())


def test_cuda_workspace_precedes_torch_import():
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
import os
from research.workflow_benchmark import phase_contract
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name == 'torch':
        assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
        raise RuntimeError('checked before torch')
    return original(name, *args, **kwargs)
builtins.__import__ = checked
try:
    phase_contract.prepare_process('cuda')
except RuntimeError as exc:
    assert str(exc) == 'checked before torch'
else:
    raise AssertionError('torch was not imported')
""",
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="real CUDA smoke requires a visible GPU"
)
def test_real_cuda_phase_smoke(tmp_path):
    contract.prepare_process("cuda")
    recorder = contract.PhaseRecorder(
        "smoke",
        tmp_path / "report.json",
        {p: [p] for p in contract.PHASES},
        contract.capture_conditions("none", "cuda"),
    )
    with recorder.phase("setup"):
        tensor = torch.ones(1024, device="cuda")
        assert tensor.device.type == "cuda"
    assert (
        recorder.report["phases"]["setup"]["peak_vram_bytes"]
        >= tensor.numel() * tensor.element_size()
    )


@pytest.mark.parametrize("module", ["paired_classification", "paired_segmentation"])
@pytest.mark.parametrize("atol,passes", [(0.002, True), (0.0001, False)])
def test_cuda_pair_reports_weight_tolerance(
    tmp_path, monkeypatch, module, atol, passes
):
    """Exercise pair comparison with real CPU state files and no training calls."""
    import json
    from types import SimpleNamespace

    from visionforge.core import dataset_audit, dataset_fingerprint
    from visionforge.utils import environment

    paired = importlib.import_module("research.workflow_benchmark." + module)
    native = importlib.import_module(
        "research.workflow_benchmark." + module.replace("paired_", "native_")
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(native, "preflight", lambda *args, **kwargs: {})
    monkeypatch.setattr(dataset_audit, "audit_dataset", lambda *args: {"ok": True})
    monkeypatch.setattr(
        dataset_fingerprint,
        "fingerprint_dataset",
        lambda *args, **kwargs: SimpleNamespace(to_dict=lambda: {"content": "same"}),
    )
    monkeypatch.setattr(environment, "capture_environment", lambda: {})
    left, right = tmp_path / "left.pth", tmp_path / "right.pth"
    torch.save({"w": torch.tensor([1.0])}, left)
    torch.save({"w": torch.tensor([1.001])}, right)
    baseline = {
        "initial_weights_sha256": "same",
        "class_to_idx": {"a": 0},
        "history": [],
        "best_epoch": 1,
        "best_val_loss": 1.0,
        "test_accuracy": 1.0,
        "test_targets": [0],
        "test_predictions": [0],
        "best_val_miou": 1.0,
        "test_metrics": {"miou": 1.0},
        "test_confusion_per_image": [[[1, 0], [0, 1]]],
        "test_predictions_sha256": "same",
    }

    def runner(path):
        def run(*args, **kwargs):
            assert kwargs["device"] == "cuda"
            assert kwargs["parity_atol"] == atol
            return {**baseline, "best_model": str(path), "device": "cuda"}

        return run

    monkeypatch.setattr(native, "run_native", runner(left))
    monkeypatch.setattr(paired, "run_visionforge", runner(right))
    output = tmp_path / "output"
    if passes:
        report = paired.run_pair(
            tmp_path / "data", left, output, device="cuda", parity_atol=atol
        )
        assert report["status"] == "passed"
    else:
        with pytest.raises(ValueError, match="best_weights"):
            paired.run_pair(
                tmp_path / "data", left, output, device="cuda", parity_atol=atol
            )
    report = json.loads((output / "paired_result.json").read_text())
    assert report["device"] == "cuda"
    assert report["parity_atol"] == atol
    assert report["weights_max_abs_diff"] == pytest.approx(0.001, abs=1e-7)
    assert report["mismatches"] == ([] if passes else ["best_weights"])
