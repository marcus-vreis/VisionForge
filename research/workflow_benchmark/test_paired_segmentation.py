"""Paired segmentation harness on synthetic CPU fixtures (ADR-149).

The fixtures check implementation parity, not segmentation quality.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from research.workflow_benchmark import native_segmentation, phase_contract
from research.workflow_benchmark import paired_segmentation as paired
from research.workflow_benchmark.phase_contract import PhaseFailure
from research.workflow_benchmark.segmentation_fixture import make_segmentation_fixture
from research.workflow_benchmark.trimap_labels import TRIMAP_TO_MASK


def test_pair_module_defers_torch_import_until_cpu_cli_guard() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import research.workflow_benchmark.paired_segmentation; "
            "assert 'torch' not in sys.modules, 'torch imported before the CPU guard'",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr


# 0.01 leaves validation mIoU flat (tie keeps epoch 1); 1.0 improves every epoch.
@pytest.mark.parametrize(("learning_rate", "best_epoch"), [(0.01, 1), (1.0, 3)])
def test_native_and_platform_match_on_trimap_derived_masks(
    tmp_path: Path, learning_rate: float, best_epoch: int
) -> None:
    dataset, initial = make_segmentation_fixture(tmp_path)
    output = tmp_path / "pair"

    report = paired.run_pair(
        dataset,
        initial,
        output,
        seed=37,
        epochs=3,
        batch_size=2,
        learning_rate=learning_rate,
    )

    assert report["device"] == "cpu"
    assert report["parity_atol"] == 0.0
    assert report["weights_max_abs_diff"] == 0.0
    for side in ("native", "visionforge"):
        assert report[side]["device"] == "cpu"
        assert report[side]["conditions"]["device"] == "cpu"
    assert report["status"] == "passed", report.get("mismatches")
    assert report["mismatches"] == []
    assert report["failed_phase"] is None
    assert report["purpose"] == "engineering_parity_only"
    assert report["task"] == "segmentation"
    assert report["execution_order"] == ["native_pytorch", "visionforge_trainer"]
    assert report["tolerances"]["metrics_abs"] == 1e-12
    assert report["tolerances"]["losses"] == "exact"
    assert report["label_mapping"]["oxford_trimap"] == {
        str(k): v for k, v in TRIMAP_TO_MASK.items()
    }
    assert report["environment"]["git_revision"]
    assert report["dataset_fingerprint"]["method"] == "content"
    native, platform = report["native"], report["visionforge"]
    assert native["best_epoch"] == platform["best_epoch"] == best_epoch
    assert native["history"] == platform["history"]
    assert native["test_confusion_per_image"] == platform["test_confusion_per_image"]
    assert native["test_predictions_sha256"] == platform["test_predictions_sha256"]
    assert native["best_model_sha256"] and platform["best_model_sha256"]
    for path in (native, platform):
        assert path["status"] == "completed"
        assert list(path["phases"]) == list(phase_contract.PHASES)
    saved = json.loads((output / "paired_result.json").read_text(encoding="utf-8"))
    assert saved == report
    before = (output / "paired_result.json").read_bytes()
    with pytest.raises(FileExistsError):
        paired.run_pair(dataset, initial, output)
    assert (output / "paired_result.json").read_bytes() == before


def test_introduced_divergence_fails_the_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different native normalization must be caught, not tolerated."""
    dataset, initial = make_segmentation_fixture(tmp_path)
    monkeypatch.setattr(native_segmentation, "NORMALIZE_MEAN", (0.4, 0.5, 0.5))
    output = tmp_path / "pair"

    with pytest.raises(ValueError, match="differ"):
        paired.run_pair(dataset, initial, output, seed=37, epochs=1, batch_size=2)

    report = json.loads((output / "paired_result.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["failed_phase"] == "comparison"
    assert "history" in report["mismatches"]
    assert "test_predictions" in report["mismatches"] or (
        "best_weights" in report["mismatches"]
    )


def _result(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "initial_weights_sha256": "a",
        "history": [
            {
                "epoch": 1,
                "train_loss": 0.7,
                "val_loss": 0.6,
                "val_miou": 0.4,
                "val_dice": 0.5,
                "val_pixel_acc": 0.6,
            }
        ],
        "best_epoch": 1,
        "best_val_miou": 0.4,
        "test_metrics": {"miou": 0.3, "dice": 0.4, "pixel_acc": 0.5},
        "test_confusion_per_image": [[[1, 0], [0, 1]]],
        "test_predictions_sha256": "p",
    }
    result = copy.deepcopy(base)
    result.update(changes)
    return result


def test_comparison_tolerates_metric_rounding_only() -> None:
    native = _result()
    rounding = _result()
    rounding["history"][0]["val_miou"] += 1e-13
    rounding["test_metrics"]["dice"] -= 1e-13
    assert paired.compare_results(native, rounding) == []

    drifted = _result()
    drifted["history"][0]["val_miou"] += 1e-9
    assert paired.compare_results(native, drifted) == ["history"]

    loss = _result()
    loss["history"][0]["train_loss"] = 0.7 + 1e-15
    assert paired.compare_results(native, loss) == ["history"]
    assert paired.compare_results(native, loss, loss_atol=1e-12) == []
    cuda_noise = _result()
    cuda_noise["history"][0]["val_loss"] += 1e-9
    assert paired.compare_results(native, cuda_noise) == ["history"]
    assert paired.compare_results(native, cuda_noise, loss_atol=1e-3) == []
    metric_noise = _result()
    metric_noise["history"][0]["val_miou"] += 1e-9
    assert paired.compare_results(native, metric_noise, loss_atol=1e-3) == ["history"]

    test_metric = _result()
    test_metric["test_metrics"]["pixel_acc"] = 0.5 + 1e-9
    assert paired.compare_results(native, test_metric) == ["test_metrics"]

    pixels = _result(test_confusion_per_image=[[[0, 1], [0, 1]]])
    epoch = _result(best_epoch=2)
    assert paired.compare_results(native, pixels) == ["test_confusion_per_image"]
    assert paired.compare_results(native, epoch) == ["best_epoch"]


RUNNERS = {
    "native_pytorch": ("native_result.json", native_segmentation.run_native),
    "visionforge_trainer": ("visionforge_result.json", paired.run_visionforge),
}


@pytest.mark.parametrize("workflow", list(RUNNERS))
@pytest.mark.parametrize("failing", ["setup", "training", "evaluation", "evidence"])
def test_forced_failure_in_each_phase_keeps_partial_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow: str, failing: str
) -> None:
    dataset, initial = make_segmentation_fixture(
        tmp_path, num_classes=3 if failing == "setup" else 2
    )
    if failing == "training":
        (dataset / "train" / "images" / "train_01.png").write_bytes(b"broken")
    elif failing == "evaluation":
        (dataset / "test" / "images" / "test_01.png").write_bytes(b"broken")
    elif failing == "evidence":
        real_hash = phase_contract.sha256_file

        def failing_hash(path: Path) -> str:
            if Path(path).name == "best_model.pth":
                raise OSError("injected evidence fault")
            return real_hash(path)

        monkeypatch.setattr(phase_contract, "sha256_file", failing_hash)
    file_name, runner = RUNNERS[workflow]
    output = tmp_path / "run"

    with pytest.raises(PhaseFailure) as caught:
        runner(dataset, initial, output, seed=37, epochs=1, batch_size=2)

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
    with pytest.raises(FileExistsError):
        runner(dataset, initial, output, seed=37, epochs=1, batch_size=2)
    assert report_path.read_bytes() == before


def test_pair_failure_names_path_and_phase(tmp_path: Path) -> None:
    dataset, initial = make_segmentation_fixture(tmp_path)
    (dataset / "train" / "images" / "train_00.png").write_bytes(b"broken")
    output = tmp_path / "pair"

    with pytest.raises(PhaseFailure):
        paired.run_pair(dataset, initial, output, seed=37, epochs=1, batch_size=2)

    report = json.loads((output / "paired_result.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["failed_phase"] == "native.training"
    assert report["native"]["failed_phase"] == "training"
    assert "visionforge" not in report
