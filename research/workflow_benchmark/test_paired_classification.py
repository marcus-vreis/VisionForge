"""Synthetic pairs check implementation parity, not scientific performance."""

import hashlib
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision import models

from research.workflow_benchmark import phase_contract
from research.workflow_benchmark.native_classification import run_native
from research.workflow_benchmark.paired_classification import run_pair, run_visionforge
from research.workflow_benchmark.phase_contract import PhaseFailure


def test_pair_module_defers_torch_import_until_cpu_cli_guard() -> None:
    """Keep accelerator libraries unloaded until the CLI can hide CUDA."""
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import research.workflow_benchmark.paired_classification; "
            "assert 'torch' not in sys.modules, 'torch imported before the CPU guard'",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr


@pytest.mark.parametrize("class_count", [2, 3])
def test_native_and_platform_match_with_shared_weights(
    tmp_path: Path, class_count: int
) -> None:
    dataset = tmp_path / "dataset"
    for role, split in enumerate(("train", "val", "test")):
        for label in range(class_count):
            folder = dataset / split / f"class{label}"
            folder.mkdir(parents=True)
            for index in range(2 if class_count == 3 else 3):
                pixels = np.random.default_rng(
                    role * 100 + label * 10 + index
                ).integers(0, 256, (36, 40, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{index}.png")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, class_count)
    initial = tmp_path / "initial.pth"
    torch.save(model.state_dict(), initial)
    output = tmp_path / "pair"

    report = run_pair(dataset, initial, output, seed=37, epochs=2, batch_size=4)

    assert report["device"] == "cpu"
    assert report["parity_atol"] == 0.0
    assert report["weights_max_abs_diff"] == 0.0
    for side in ("native", "visionforge"):
        assert report[side]["device"] == "cpu"
        assert report[side]["conditions"]["device"] == "cpu"
    assert report["status"] == "passed"
    assert report["purpose"] == "engineering_parity_only"
    assert report["mismatches"] == []
    assert report["native"]["history"] == report["visionforge"]["history"]
    assert json.loads((output / "paired_result.json").read_text()) == report
    assert report["failed_phase"] is None
    assert report["phase_contract"]["version"] == phase_contract.CONTRACT_VERSION
    assert report["execution_order"] == ["native_pytorch", "visionforge_trainer"]
    for path in ("native", "visionforge"):
        assert report[path]["status"] == "completed"
        assert list(report[path]["phases"]) == list(phase_contract.PHASES)
    with pytest.raises(FileExistsError):
        run_pair(dataset, initial, output)


def _fixture(tmp_path: Path, class_count: int = 2) -> tuple[Path, Path]:
    """Create three small ImageFolder splits and a matching initial state."""
    dataset = tmp_path / "dataset"
    for role, split in enumerate(("train", "val", "test")):
        for label in range(2):
            folder = dataset / split / f"class{label}"
            folder.mkdir(parents=True)
            for index in range(3):
                pixels = np.random.default_rng(
                    role * 100 + label * 10 + index
                ).integers(0, 256, (36, 40, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{index}.png")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, class_count)
    initial = tmp_path / "initial.pth"
    torch.save(model.state_dict(), initial)
    return dataset, initial


RUNNERS = {
    "native_pytorch": ("native_result.json", run_native),
    "visionforge_trainer": ("visionforge_result.json", run_visionforge),
}


@pytest.mark.parametrize("workflow", list(RUNNERS))
def test_both_paths_report_the_same_phase_contract(
    tmp_path: Path, workflow: str
) -> None:
    """Each path records the shared boundaries, its own work list and conditions."""
    dataset, initial = _fixture(tmp_path)
    file_name, runner = RUNNERS[workflow]
    output = tmp_path / "run"
    result = runner(
        dataset,
        initial,
        output,
        seed=37,
        epochs=1,
        batch_size=4,
        concurrent_workload="none (test)",
    )
    assert json.loads((output / file_name).read_text(encoding="utf-8")) == result
    assert result["workflow"] == workflow
    assert result["status"] == "completed"
    assert result["failed_phase"] is None
    assert result["phase_contract"]["version"] == phase_contract.CONTRACT_VERSION
    assert result["phase_contract"]["boundaries"] == phase_contract.BOUNDARIES
    assert list(result["phases"]) == list(phase_contract.PHASES)
    for phase in result["phases"].values():
        assert phase["status"] == "completed"
        assert phase["seconds"] >= 0
        assert phase["includes"]
    conditions = result["conditions"]
    assert conditions["concurrent_workload"] == "none (test)"
    assert conditions["cache"]["downloads"] == "none"
    assert conditions["versions"]["torch"] == metadata.version("torch")
    assert (
        result["best_model_sha256"]
        == hashlib.sha256(Path(result["best_model"]).read_bytes()).hexdigest()
    )
    assert not any(key.endswith("_seconds") for key in result)


def _break_image(path: Path) -> None:
    """Replace an image with bytes PIL cannot decode."""
    path.write_bytes(b"not an image")


@pytest.mark.parametrize("workflow", list(RUNNERS))
@pytest.mark.parametrize("failing", ["setup", "training", "evaluation", "evidence"])
def test_forced_failure_in_each_phase_keeps_partial_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow: str, failing: str
) -> None:
    """A real or injected fault leaves status, phase and completed artifacts."""
    dataset, initial = _fixture(tmp_path, class_count=3 if failing == "setup" else 2)
    if failing == "training":
        _break_image(dataset / "train" / "class0" / "0.png")
    elif failing == "evaluation":
        _break_image(dataset / "test" / "class1" / "2.png")
    elif failing == "evidence":
        real_hash = phase_contract.sha256_file

        def failing_hash(path: Path) -> str:
            if Path(path) != initial.resolve():
                raise OSError("injected evidence fault")
            return real_hash(path)

        monkeypatch.setattr(phase_contract, "sha256_file", failing_hash)
    file_name, runner = RUNNERS[workflow]
    output = tmp_path / "run"

    with pytest.raises(PhaseFailure) as caught:
        runner(dataset, initial, output, seed=37, epochs=1, batch_size=4)

    assert caught.value.phase == failing
    report_path = output / file_name
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == caught.value.report
    assert report["workflow"] == workflow
    assert report["status"] == "failed"
    assert report["failed_phase"] == failing
    assert report["error"]
    stop = phase_contract.PHASES.index(failing)
    assert list(report["phases"]) == list(phase_contract.PHASES[: stop + 1])
    if failing in ("evaluation", "evidence"):
        checkpoints = list(output.rglob("best_model.pth"))
        assert checkpoints and all(path.stat().st_size > 0 for path in checkpoints)
    before = report_path.read_bytes()
    with pytest.raises((FileExistsError, ValueError)):
        runner(dataset, initial, output, seed=37, epochs=1, batch_size=4)
    assert report_path.read_bytes() == before


def test_pair_failure_names_path_and_phase_and_embeds_partial_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed native evidence phase stops the pair before the platform path."""
    dataset, initial = _fixture(tmp_path)

    real_hash = phase_contract.sha256_file

    def failing_hash(path: Path) -> str:
        if Path(path) != initial.resolve():
            raise OSError("injected evidence fault")
        return real_hash(path)

    monkeypatch.setattr(phase_contract, "sha256_file", failing_hash)
    output = tmp_path / "pair"
    with pytest.raises(PhaseFailure):
        run_pair(dataset, initial, output, seed=37, epochs=1, batch_size=4)
    report = json.loads((output / "paired_result.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["failed_phase"] == "native.evidence"
    assert report["native"]["failed_phase"] == "evidence"
    assert "visionforge" not in report
    assert report["phase_contract"]["version"] == phase_contract.CONTRACT_VERSION
    assert (output / "native" / "native_result.json").is_file()
