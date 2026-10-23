"""Phase-contract recording, failure reports, legacy scope and the CPU CLI guard."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from research.workflow_benchmark import phase_contract
from research.workflow_benchmark.phase_contract import (
    PHASES,
    PhaseFailure,
    PhaseRecorder,
    report_scope,
)

INCLUDES = {name: [f"{name} work"] for name in PHASES}
CONDITIONS = {"concurrent_workload": "none declared by test"}


def test_completed_report_records_every_phase_in_order(tmp_path: Path) -> None:
    """A complete run records each phase with boundaries, includes and timing kind."""
    report_path = tmp_path / "result.json"
    recorder = PhaseRecorder("demo", report_path, INCLUDES, CONDITIONS)
    for name in PHASES:
        with recorder.phase(name):
            pass

    report = recorder.complete({"answer": 42})

    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert report["status"] == "completed"
    assert report["failed_phase"] is None
    assert report["answer"] == 42
    assert report["conditions"] == CONDITIONS
    contract = report["phase_contract"]
    assert contract["version"] == phase_contract.CONTRACT_VERSION
    assert contract["timing_kind"] == "machine_wall_clock"
    assert contract["human_effort"] == "not_measured"
    assert list(contract["boundaries"]) == list(PHASES)
    assert list(report["phases"]) == list(PHASES)
    for name, phase in report["phases"].items():
        assert phase["status"] == "completed"
        assert phase["seconds"] >= 0
        assert phase["includes"] == INCLUDES[name]
    assert report_scope(report) == phase_contract.CONTRACT_SCOPE


@pytest.mark.parametrize("failing", PHASES)
def test_failure_in_each_phase_keeps_partial_report(
    tmp_path: Path, failing: str
) -> None:
    """The failed phase is named, earlier phases kept and later phases absent."""
    report_path = tmp_path / "result.json"
    recorder = PhaseRecorder("demo", report_path, INCLUDES, CONDITIONS)
    stop = PHASES.index(failing)
    with pytest.raises(PhaseFailure) as caught:
        for name in PHASES[: stop + 1]:
            with recorder.phase(name):
                if name == failing:
                    raise OSError("injected fault")

    assert caught.value.phase == failing
    assert caught.value.workflow == "demo"
    assert isinstance(caught.value.__cause__, OSError)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == caught.value.report
    assert report["status"] == "failed"
    assert report["failed_phase"] == failing
    assert report["error"] == "OSError: injected fault"
    assert list(report["phases"]) == list(PHASES[: stop + 1])
    assert report["phases"][failing]["status"] == "failed"
    assert report["phases"][failing]["seconds"] >= 0
    for name in PHASES[:stop]:
        assert report["phases"][name]["status"] == "completed"


def test_phases_must_follow_contract_order(tmp_path: Path) -> None:
    """Skipping or repeating a phase is a harness bug, not a measurement."""
    recorder = PhaseRecorder("demo", tmp_path / "result.json", INCLUDES, CONDITIONS)
    with pytest.raises(ValueError, match="expected phase 'setup'"):
        with recorder.phase("training"):
            pass
    with recorder.phase("setup"):
        pass
    with pytest.raises(ValueError, match="expected phase 'training'"):
        with recorder.phase("setup"):
            pass
    with pytest.raises(ValueError, match="incomplete"):
        recorder.complete({})
    assert not (tmp_path / "result.json").exists()


def test_includes_must_cover_every_phase(tmp_path: Path) -> None:
    """A report cannot omit what work one of its phases contains."""
    with pytest.raises(ValueError, match="evidence"):
        PhaseRecorder(
            "demo",
            tmp_path / "result.json",
            {name: INCLUDES[name] for name in PHASES[:-1]},
            CONDITIONS,
        )


def test_existing_report_is_never_overwritten(tmp_path: Path) -> None:
    """Both the completed and the failure write refuse an existing file."""
    report_path = tmp_path / "result.json"
    report_path.write_text("earlier", encoding="utf-8")
    recorder = PhaseRecorder("demo", report_path, INCLUDES, CONDITIONS)
    for name in PHASES:
        with recorder.phase(name):
            pass
    with pytest.raises(FileExistsError):
        recorder.complete({})
    failing = PhaseRecorder("demo", report_path, INCLUDES, CONDITIONS)
    with pytest.raises(FileExistsError):
        with failing.phase("setup"):
            raise RuntimeError("boom")
    assert report_path.read_text(encoding="utf-8") == "earlier"


def test_legacy_reports_keep_their_original_scope(tmp_path: Path) -> None:
    """Pre-contract timings are labelled legacy and left byte-for-byte intact."""
    legacy = {
        "workflow": "native_pytorch",
        "setup_seconds": 1.5,
        "train_seconds": 2.0,
        "evaluation_seconds": 0.5,
    }
    path = tmp_path / "native_result.json"
    original = json.dumps(legacy, indent=2) + "\n"
    path.write_text(original, encoding="utf-8")

    assert phase_contract.load_report_scope(path) == phase_contract.LEGACY_SCOPE
    assert path.read_text(encoding="utf-8") == original
    assert report_scope({"phase_contract": {"version": 99}}) == "unknown_contract_99"


def test_capture_conditions_records_cache_versions_and_concurrency() -> None:
    """Conditions name cache state, versions, threads, CUDA mask and priority."""
    conditions = phase_contract.capture_conditions("training job on GPU 0")
    assert conditions["concurrent_workload"] == "training job on GPU 0"
    assert conditions["cache"]["downloads"] == "none"
    assert set(conditions["cache"]) == {
        "dataset",
        "initial_weights",
        "downloads",
        "os_file_cache",
    }
    assert {"python", "torch", "torchvision", "numpy", "pillow"} <= set(
        conditions["versions"]
    )
    assert conditions["cpu_threads"] >= 1
    assert "cuda_visible_devices" in conditions
    assert isinstance(conditions["process_priority"], str)


def test_cli_guard_hides_cuda_limits_threads_and_lowers_priority() -> None:
    """The guard runs before torch loads and leaves one low-priority CPU thread."""
    probe = (
        "import os, sys\n"
        "from research.workflow_benchmark.phase_contract import "
        "prepare_cpu_process, process_priority\n"
        "before = process_priority()\n"
        "prepare_cpu_process()\n"
        "assert 'torch' not in sys.modules, 'guard imported torch'\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'\n"
        "import torch\n"
        "assert not torch.cuda.is_available()\n"
        "assert torch.get_num_threads() == 1, torch.get_num_threads()\n"
        "after = process_priority()\n"
        # A runner that already starts below normal (GitHub's Windows images do)
        # cannot go lower; anywhere else the guard must have changed it.
        "assert after != before or before in ('below_normal', 'idle'), "
        "(before, after)\n"
        "assert after in ('below_normal', 'idle') or after.startswith('nice='), after\n"
    )
    process = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
