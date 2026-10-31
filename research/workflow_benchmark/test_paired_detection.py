"""CPU detection parity and failure evidence with real SSDLite training."""

import copy
import json
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import pytest

from . import native_detection, phase_contract
from . import paired_detection as paired
from .detection_fixture import make_detection_fixture


@pytest.fixture(scope="module")
def parity(tmp_path_factory):
    root = tmp_path_factory.mktemp("paired_detection")
    dataset, initial = make_detection_fixture(root)
    report = paired.run_pair(
        dataset, initial, root / "pair", seed=37, epochs=2, batch_size=2
    )
    return root, dataset, initial, report


def test_exact_cpu_parity(parity):
    root, dataset, initial, report = parity
    assert report["status"] == "passed"
    assert report["mismatches"] == []
    assert report["weights_max_abs_diff"] == 0.0
    assert report["native"]["history"] == report["visionforge"]["history"]
    assert len(report["native"]["history"]) == 2
    assert (
        report["native"]["test_predictions_sha256"]
        == report["visionforge"]["test_predictions_sha256"]
    )
    assert (
        report["visionforge"]["initial_weights_injection"]
        == "monkeypatch of detection_trainer.build_torchvision_detector"
    )
    assert report["purpose"] == "engineering_parity_only"
    assert report["execution_order"] == paired.EXECUTION_ORDER
    assert report["environment"]["git_revision"]
    assert report["dataset_fingerprint"] == paired.fingerprint_dataset(dataset)
    assert json.loads((root / "pair/paired_result.json").read_text()) == report
    for side in ("native", "visionforge"):
        assert list(report[side]["phases"]) == list(phase_contract.PHASES)
        assert report[side]["best_epoch"] in (1, 2)
    assert report["native"]["best_epoch"] == report["visionforge"]["best_epoch"]
    with pytest.raises(FileExistsError):
        paired.run_pair(dataset, initial, root / "pair")


@pytest.mark.parametrize(
    "field,value",
    [
        ("best_epoch", 99),
        ("num_classes", 99),
        ("initial_weights_sha256", "changed"),
        ("class_mapping", {}),
    ],
)
def test_comparison_detects_field_drift(parity, field, value):
    a = parity[3]["native"]
    b = copy.deepcopy(a)
    b[field] = value
    assert paired.compare_results(a, b) == [field]


def test_comparison_loss_and_metric_tolerances(parity):
    a = parity[3]["native"]
    b = copy.deepcopy(a)
    b["history"][0]["train_loss"] += 1e-5
    assert paired.compare_results(a, b) == ["history"]
    assert paired.compare_results(a, b, loss_atol=1e-4) == []
    b["history"][0]["val_map50"] += 1e-9
    assert paired.compare_results(a, b, loss_atol=1e-4) == ["history"]


def test_prediction_values_are_exact_on_cpu():
    a = [{"boxes": [[1.0, 2, 3, 4]], "scores": [0.5], "labels": [1]}]
    b = copy.deepcopy(a)
    b[0]["boxes"][0][0] += 1e-8
    assert not paired._predictions_match(a, b, 0)
    assert paired._predictions_match(a, b, 1e-6)
    b[0]["labels"][0] = 2
    assert not paired._predictions_match(a, b, 1)


def test_injection_restored_after_exception():
    import visionforge.core.detection_trainer as trainer

    original = trainer.build_torchvision_detector
    model = object()
    with pytest.raises(RuntimeError, match="injected"):
        with paired.inject_initial_model(model):
            assert trainer.build_torchvision_detector("unused", 3, False) is model
            raise RuntimeError("injected")
    assert trainer.build_torchvision_detector is original


RUNNERS: list[Callable[..., Any]] = [
    native_detection.run_native,
    paired.run_visionforge,
    paired.run_pair,
]


@pytest.mark.parametrize("runner", RUNNERS)
def test_cuda_requires_explicit_tolerance(tmp_path, runner):
    with pytest.raises(ValueError, match="^--parity-atol is required for cuda$"):
        runner(
            tmp_path / "data",
            tmp_path / "weights",
            tmp_path / "out",
            seed=37,
            epochs=1,
            batch_size=2,
            device="cuda",
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("side", ["native", "visionforge"])
@pytest.mark.parametrize("phase", phase_contract.PHASES)
def test_phase_failure_keeps_partial_report(
    tmp_path_factory, monkeypatch, parity, side, phase
):
    _, dataset, initial, report = parity
    original = phase_contract.PhaseRecorder.phase
    from contextlib import contextmanager

    @contextmanager
    def failing(self, name):
        with original(self, name):
            if name == phase:
                raise RuntimeError("injected phase failure")
            yield

    monkeypatch.setattr(phase_contract.PhaseRecorder, "phase", failing)
    if side == "visionforge":
        monkeypatch.setattr(
            native_detection, "run_native", lambda *a, **kw: report["native"]
        )
    output = tmp_path_factory.mktemp("failure") / "pair"
    with pytest.raises(phase_contract.PhaseFailure):
        paired.run_pair(dataset, initial, output, epochs=1)
    saved = json.loads((output / "paired_result.json").read_text())
    assert saved["status"] == "failed"
    assert saved["failed_phase"] == f"{side}.{phase}"
    assert saved[side]["phases"][phase]["status"] == "failed"
    partial = output / side / f"{side}_result.json"
    assert json.loads(partial.read_text()) == saved[side]


def test_pair_import_defers_torch():
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import research.workflow_benchmark.paired_detection; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr


def test_initial_weights_cli_and_no_overwrite(tmp_path):
    path = tmp_path / "initial.pth"
    command = [
        sys.executable,
        "-m",
        "research.workflow_benchmark.native_detection",
        "--make-initial",
        str(path),
        "--seed",
        "37",
    ]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    before = phase_contract.sha256_file(path)
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0
    assert "FileExistsError" in second.stderr
    assert phase_contract.sha256_file(path) == before


def test_fallback_uses_first_split_without_test(tmp_path_factory):
    root = tmp_path_factory.mktemp("fallback")
    dataset, initial = make_detection_fixture(root, split_sizes={"train": 2, "val": 2})
    report = paired.run_pair(dataset, initial, root / "pair", epochs=1)
    assert report["status"] == "passed"
    assert report["mismatches"] == []
    assert report["weights_max_abs_diff"] == 0
    assert (
        report["native"]["test_split"] == report["visionforge"]["test_split"] == "train"
    )
    assert report["native"]["test_stems"] == ["000", "001"]
