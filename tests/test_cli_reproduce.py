"""``visionforge reproduce``: exit codes, one JSON on stdout, a real CPU training."""

from __future__ import annotations

import contextvars
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

import visionforge.__main__ as main_mod
import visionforge.cli_strategies as cli
from visionforge.core.experiment_bundle import export_experiment_bundle
from visionforge.core.task_runner import RunResult
from visionforge.utils.regression_config import RegressionConfig

R2 = "r2"


class FakeRunner:
    config_type = RegressionConfig

    def __init__(self, value: float, status: str = "success") -> None:
        self.value = value
        self.status = status

    def run(self, cfg: Any) -> RunResult:
        return RunResult(
            metrics={R2: self.value}, status=self.status, error="disk full"
        )

    def metrics(self, result: RunResult) -> dict[str, float]:
        return dict(result.metrics)

    def primary_metric(self) -> str:
        return R2


def _invoke(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["visionforge", *args])
    try:
        # A copy of the context keeps the language the command binds from leaking.
        contextvars.copy_context().run(main_mod.main)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def _config(tmp_path: Path, name: str = "reg") -> dict[str, Any]:
    (tmp_path / "data").mkdir(exist_ok=True)
    return {
        "name": name,
        "task": "regression",
        "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
        "data": {"base_dir": str(tmp_path / "data"), "target_columns": ["target"]},
        "training": {"epochs": 1, "batch_size": 4, "seed": 3},
        "device": {"kind": "cpu"},
    }


def _record_bundle(tmp_path: Path, record: dict[str, Any]) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps({"artifacts": {}, **record}), encoding="utf-8"
    )
    return export_experiment_bundle(run_dir, tmp_path / "bundle.zip")


def _group_record(tmp_path: Path, kind: str = "replicates") -> dict[str, Any]:
    return {
        "config": _config(tmp_path),
        "metrics": {"test_r2": 0.8, "total_epochs": 2},
        "block": kind,
        "group": {
            "kind": kind,
            "seeds": [0, 1],
            "children": [
                {"seed": 0, "status": "success", "metrics": {R2: 0.79}},
                {"seed": 1, "status": "success", "metrics": {R2: 0.81}},
            ],
            # As aggregate_replicates writes it for the two children above;
            # pi95 = 0.8 ± 12.7062 * 0.014142 * sqrt(1.5) = (0.580, 1.0 clipped).
            "aggregates": {
                R2: {
                    "n": 2,
                    "mean": 0.8,
                    "std": 0.0141421356,
                    "std_ddof": 1,
                    "ci95_low": 0.6729,
                    "ci95_high": 0.9271,
                }
            },
        },
    }


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeRunner:
    runner = FakeRunner(0.8)
    monkeypatch.setattr(cli, "runner_for", lambda task: runner)
    return runner


def _out(tmp_path: Path) -> list[str]:
    return ["--output", str(tmp_path / "out")]


def test_a_single_run_prints_one_json_and_exits_0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake: FakeRunner,
) -> None:
    bundle = _record_bundle(
        tmp_path, {"config": _config(tmp_path), "metrics": {"test_r2": 0.75}}
    )
    code = _invoke(monkeypatch, ["reproduce", str(bundle), *_out(tmp_path)])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert set(report) == {
        "bundle",
        "kind",
        "seed",
        "run_dir",
        "environment_diff",
        "metrics",
        "verdict",
    }
    assert (report["kind"], report["seed"], report["verdict"]) == (
        "single",
        3,
        "no_interval",
    )
    assert report["metrics"][0]["delta"] == pytest.approx(0.05)


def test_a_replicate_seed_outside_the_interval_exits_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake: FakeRunner,
) -> None:
    fake.value = 0.2
    bundle = _record_bundle(tmp_path, _group_record(tmp_path))
    args = ["reproduce", str(bundle), "--seed", "1", *_out(tmp_path)]
    code = _invoke(monkeypatch, args)
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["verdict"] == "outside_interval"
    assert report["metrics"][0]["recorded"] == pytest.approx(0.81)


def test_a_replicate_seed_inside_the_interval_exits_0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake: FakeRunner,
) -> None:
    bundle = _record_bundle(tmp_path, _group_record(tmp_path))
    code = _invoke(monkeypatch, ["reproduce", str(bundle), *_out(tmp_path)])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "inside_interval"


def _assert_exit_2(
    capsys: pytest.CaptureFixture[str], code: int, fragment: str
) -> None:
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1
    assert fragment in lines[0]


def test_an_invalid_bundle_exits_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake: FakeRunner,
) -> None:
    path = tmp_path / "x.zip"
    path.write_bytes(b"not a zip")
    code = _invoke(monkeypatch, ["reproduce", str(path)])
    _assert_exit_2(capsys, code, "invalid experiment bundle")


def test_a_missing_bundle_exits_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake: FakeRunner,
) -> None:
    code = _invoke(monkeypatch, ["reproduce", str(tmp_path / "none.zip")])
    _assert_exit_2(capsys, code, "none.zip")


@pytest.mark.parametrize("kind", ["kfold", "sweep", "replicated_comparison"])
def test_an_unsupported_kind_exits_2_naming_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake: FakeRunner,
    kind: str,
) -> None:
    bundle = _record_bundle(tmp_path, _group_record(tmp_path, kind))
    code = _invoke(monkeypatch, ["reproduce", str(bundle)])
    _assert_exit_2(capsys, code, f"'{kind}' bundle is not supported yet")


def test_a_seed_outside_the_group_exits_2_listing_its_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake: FakeRunner,
) -> None:
    bundle = _record_bundle(tmp_path, _group_record(tmp_path))
    code = _invoke(monkeypatch, ["reproduce", str(bundle), "--seed", "9"])
    _assert_exit_2(capsys, code, "seeds: 0, 1")


def test_a_dataset_missing_here_exits_2_asking_for_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake: FakeRunner,
) -> None:
    config = _config(tmp_path)
    config["data"]["base_dir"] = "my_dataset"  # what a public bundle keeps
    bundle = _record_bundle(tmp_path, {"config": config, "metrics": {}})
    monkeypatch.chdir(tmp_path)
    code = _invoke(monkeypatch, ["reproduce", str(bundle)])
    _assert_exit_2(capsys, code, "--data DATASET_DIR")


def test_data_replaces_the_recorded_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake: FakeRunner,
) -> None:
    config = _config(tmp_path)
    config["data"]["base_dir"] = "my_dataset"
    bundle = _record_bundle(tmp_path, {"config": config, "metrics": {}})
    args = ["reproduce", str(bundle), "--data", str(tmp_path / "data")]
    code = _invoke(monkeypatch, [*args, *_out(tmp_path)])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "no_interval"


def test_a_task_without_a_runner_exits_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = {**_config(tmp_path), "task": "keypoints"}
    bundle = _record_bundle(tmp_path, {"config": config, "metrics": {}})
    code = _invoke(monkeypatch, ["reproduce", str(bundle), *_out(tmp_path)])
    _assert_exit_2(capsys, code, "keypoints")


def test_a_failed_training_exits_1_without_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake: FakeRunner,
) -> None:
    fake.status = "failed"
    bundle = _record_bundle(tmp_path, {"config": _config(tmp_path), "metrics": {}})
    code = _invoke(monkeypatch, ["reproduce", str(bundle), *_out(tmp_path)])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "disk full" in captured.err


def test_a_replicate_set_reproduces_on_cpu_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Two seeds of a 1-epoch regression on 32 px images, bundled, one re-trained."""
    from visionforge.utils.selftest_data import build_regression_dataset

    base = build_regression_dataset(tmp_path / "dataset", rows=8)
    config = _config(tmp_path, name="e2e")
    config["data"].update(
        {
            "base_dir": str(base),
            "num_workers": 0,
            "pin_memory": False,
            "transforms": {"image_size": 32},
        }
    )
    config["output"] = {
        key: str(tmp_path / "original" / key.removesuffix("_dir"))
        for key in ("models_dir", "graphics_dir", "logs_dir", "reports_dir")
    }
    envelope = tmp_path / "run.yaml"
    envelope.write_text(
        yaml.safe_dump(
            {"strategy": {"kind": "replicates", "seeds": [0, 1]}, "config": config}
        ),
        encoding="utf-8",
    )
    assert _invoke(monkeypatch, ["run", str(envelope)]) == 0
    (group_dir,) = (tmp_path / "original" / "models" / "e2e_replicates").iterdir()
    group = json.loads((group_dir / "run.json").read_text(encoding="utf-8"))
    originals = {
        Path(c["run_dir"]): (Path(c["run_dir"]) / "run.json").read_bytes()
        for c in group["group"]["children"]
    }
    bundle = export_experiment_bundle(group_dir, tmp_path / "e2e.zip")
    capsys.readouterr()

    out = tmp_path / "repro"
    code = _invoke(monkeypatch, ["reproduce", str(bundle), "--output", str(out)])

    report = json.loads(capsys.readouterr().out)
    assert code == 0, report
    assert report["kind"] == "replicates"
    assert report["seed"] == 0
    assert report["verdict"] == "inside_interval"
    run_dir = Path(report["run_dir"])
    assert run_dir.is_relative_to(out / "models" / "e2e_s0_repro")
    assert (run_dir / "run.json").is_file()
    rows = {m["name"]: m for m in report["metrics"]}
    assert set(rows) >= {"r2", "mae", "rmse"}
    assert all(row["ci95"] is not None for row in rows.values())
    assert all(row["pi95"] is not None for row in rows.values())
    assert report["environment_diff"]["torch"]["same"] is True
    assert report["environment_diff"]["device"]["same"] is True
    for path, raw in originals.items():
        assert (path / "run.json").read_bytes() == raw
