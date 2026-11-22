"""The script that runs on the Colab VM: install, provider fetch, train, pack, Drive.

Nothing here installs a package, downloads a dataset or trains: the callees that
would are replaced where the runner imports them.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import visionforge.core.experiment_bundle as experiment_bundle
import visionforge.gui.api.colab_jobs as cj
import visionforge.gui.api.dataset_download as dataset_download
import visionforge.utils.config as vf_config
from visionforge.gui.api import colab_remote_runner as runner


def _events(out: str) -> list[dict[str, Any]]:
    parsed = [cj.parse_event(line) for line in out.splitlines(keepends=True)]
    return [e for e in parsed if e]


class TestInstall:
    def test_the_matching_version_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs: Any) -> None:
            calls.append(cmd)

        monkeypatch.setattr("importlib.metadata.version", lambda name: "1.2.3")
        monkeypatch.setattr(runner.subprocess, "run", fake_run)

        runner.install("1.2.3")

        assert calls == []

    def test_another_version_is_pip_installed_pinned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[list[str], dict[str, Any]]] = []

        def fake_run(cmd: list[str], **kwargs: Any) -> None:
            calls.append((cmd, kwargs))

        monkeypatch.setattr("importlib.metadata.version", lambda name: "0.0.1")
        monkeypatch.setattr(runner.subprocess, "run", fake_run)

        runner.install("1.2.3")

        assert calls[0][0] == [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "visionforge-studio==1.2.3",
        ]
        assert calls[0][1]["check"] is True

    def test_a_package_not_installed_yet_is_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[list[str]] = []

        def missing(name: str) -> str:
            raise ModuleNotFoundError(name)

        def fake_run(cmd: list[str], **kwargs: Any) -> None:
            calls.append(cmd)

        monkeypatch.setattr("importlib.metadata.version", missing)
        monkeypatch.setattr(runner.subprocess, "run", fake_run)

        runner.install("2.0.0")

        assert calls[0][-1] == "visionforge-studio==2.0.0"

    def test_a_failing_pip_reaches_the_caller(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def missing(name: str) -> str:
            raise ModuleNotFoundError(name)

        def failing_run(cmd: list[str], **kwargs: Any) -> None:
            raise subprocess.CalledProcessError(1, cmd, stderr="no such version")

        monkeypatch.setattr("importlib.metadata.version", missing)
        monkeypatch.setattr(runner.subprocess, "run", failing_run)

        with pytest.raises(subprocess.CalledProcessError):
            runner.install("9.9.9")


class TestFetchDataset:
    def test_a_provider_is_downloaded_into_a_fresh_dataset_folder(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        received: list[tuple[str, dict[str, Any]]] = []

        def fake_download(provider: str, **kwargs: Any) -> None:
            received.append((provider, kwargs))

        monkeypatch.setattr(dataset_download, "download_dataset", fake_download)
        stale = tmp_path / "dataset" / "old.txt"
        stale.parent.mkdir(parents=True)
        stale.write_text("left over", encoding="utf-8")

        base = runner.fetch_dataset(
            {"kind": "provider", "provider": "kaggle", "dataset": "a/b", "limit": 5},
            tmp_path,
        )

        assert base == tmp_path / "dataset"
        assert not stale.exists(), "a previous attempt's files are cleared first"
        assert received == [
            (
                "kaggle",
                {"dataset": "a/b", "out_dir": str(tmp_path / "dataset"), "limit": 5},
            )
        ]

    def test_the_provider_limit_is_optional(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        received: list[dict[str, Any]] = []

        def fake_download(provider: str, **kwargs: Any) -> None:
            received.append(kwargs)

        monkeypatch.setattr(dataset_download, "download_dataset", fake_download)

        runner.fetch_dataset(
            {"kind": "provider", "provider": "roboflow", "dataset": "x"}, tmp_path
        )

        assert received[0]["limit"] is None


class FakeBlock:
    """Records the order the runner drives a block in."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._progress_callback: Any = None

    def setup(self, config: Any) -> None:
        self.calls.append(f"setup:{config}")

    def run(self) -> None:
        self.calls.append("run")

    def report(self) -> dict[str, Any]:
        self.calls.append("report")
        return {"accuracy": 0.9}


class TestBuildBlock:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("cross_validation", "CrossValidationBlock"),
            ("transfer_learning", "TransferLearningBlock"),
            ("model_comparison", "ModelComparisonBlock"),
            ("grid_search", "GridSearchBlock"),
            ("random_search", "RandomSearchBlock"),
            ("classification", "ClassificationBlock"),
            ("anything_else", "ClassificationBlock"),
        ],
    )
    def test_the_block_follows_config_block(self, name: str, expected: str) -> None:
        block = runner.build_block(SimpleNamespace(block=name))
        assert type(block).__name__ == expected


class TestTrain:
    def test_validates_sets_up_streams_events_and_returns_the_report(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        block = FakeBlock()
        validated: list[dict[str, Any]] = []

        def fake_validate(data: dict[str, Any]) -> str:
            validated.append(data)
            return "CONFIG"

        monkeypatch.setattr(vf_config.ExperimentConfig, "model_validate", fake_validate)
        monkeypatch.setattr(runner, "build_block", lambda config: block)

        report = runner.train({"name": "exp"})

        assert validated == [{"name": "exp"}]
        assert block.calls == ["setup:CONFIG", "run", "report"]
        assert block._progress_callback is runner.emit
        assert report == {"accuracy": 0.9}


class TestPackRun:
    @pytest.mark.parametrize(
        "report",
        [None, "text", {}, {"train": None}, {"train": {}}, {"train": {"run_dir": ""}}],
    )
    def test_a_report_without_a_single_run_folder_has_nothing_to_pack(
        self, report: Any
    ) -> None:
        assert runner.pack_run(report) is None

    def test_the_run_folder_is_exported_as_a_bundle(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exported: list[tuple[str, Path]] = []

        def fake_export(run_dir: str, target: Path) -> Path:
            exported.append((run_dir, target))
            target.write_bytes(b"12345")
            return target

        monkeypatch.setattr(runner, "REMOTE_ROOT", tmp_path)
        monkeypatch.setattr(experiment_bundle, "export_experiment_bundle", fake_export)
        stale = tmp_path / runner.BUNDLE_NAME
        stale.write_bytes(b"old bundle, longer than the new one")

        bundle = runner.pack_run({"train": {"run_dir": "/vm/run"}})

        assert exported == [("/vm/run", stale)]
        assert bundle == {"path": str(stale), "size": 5}


class TestEnvironment:
    def test_cpu_only_reports_python_and_torch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        info = runner.environment()

        assert info == {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        }

    def test_a_gpu_adds_its_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Tesla T4")

        assert runner.environment()["gpu"] == "Tesla T4"


PAYLOAD: dict[str, Any] = {
    "version": "1.0.0",
    "source": {"kind": "url", "url": "http://x/y.zip"},
    "config": {"name": "exp", "task": "binary", "data": {}},
    "drive_export": True,
}


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bundle: dict[str, Any] | None,
    export: Any,
) -> list[dict[str, Any]]:
    monkeypatch.setattr(runner, "REMOTE_ROOT", tmp_path)
    monkeypatch.setattr(runner, "install", lambda version: None)
    monkeypatch.setattr(runner, "fetch_dataset", lambda source, root: tmp_path)
    monkeypatch.setattr(runner, "train", lambda cfg: {"accuracy": 1.0})
    monkeypatch.setattr(runner, "pack_run", lambda report: bundle)
    monkeypatch.setattr(runner, "export_to_drive", export)
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    runner.main(PAYLOAD)
    return _events(out.getvalue())


class TestMainDriveExport:
    def test_the_drive_path_is_announced_before_job_end(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        received: list[tuple[Any, str]] = []

        def fake_export(bundle: Any, run_name: str) -> str:
            received.append((bundle, run_name))
            return "/content/drive/MyDrive/VisionForge/runs/exp.zip"

        bundle = {"path": "b.zip", "size": 1}
        events = _run_main(monkeypatch, tmp_path, bundle, fake_export)

        drive = [e for e in events if e.get("stage") == "drive_export"]
        assert drive == [
            {
                "event": "remote",
                "stage": "drive_export",
                "path": "/content/drive/MyDrive/VisionForge/runs/exp.zip",
            }
        ]
        assert received == [(bundle, "exp")]
        assert events[-1]["status"] == "ok"
        assert events[-1]["bundle"] == bundle

    def test_a_full_drive_does_not_cost_the_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def full_drive(bundle: Any, run_name: str) -> str:
            raise OSError("no space left on device")

        events = _run_main(monkeypatch, tmp_path, {"path": "b.zip"}, full_drive)

        drive = [e for e in events if e.get("stage") == "drive_export"]
        assert drive[0]["error"] == "no space left on device"
        assert events[-1]["status"] == "ok"

    def test_the_num_classes_change_is_announced(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def fake_match(cfg: dict[str, Any]) -> dict[str, Any]:
            return {"old": 2, "new": 5}

        monkeypatch.setattr(runner, "match_num_classes", fake_match)

        events = _run_main(monkeypatch, tmp_path, None, lambda bundle, name: None)

        change = [e for e in events if e.get("stage") == "num_classes"]
        assert change == [
            {"event": "remote", "stage": "num_classes", "old": 2, "new": 5}
        ]
