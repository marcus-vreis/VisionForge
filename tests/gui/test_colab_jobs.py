"""Remote training on Colab (ADR-163, phase 2) against a fake CLI: nothing reaches Google."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import subprocess
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from visionforge.gui.api import colab_connection as cc
from visionforge.gui.api import colab_jobs as cj
from visionforge.gui.api import colab_remote_runner as runner
from visionforge.gui.api.run_events import RunEventLog

from .conftest import occupy_queue, release_queue

PAYLOAD = {
    "version": "0.20.2",
    "config": {"name": "exp"},
    "source": {"kind": "url", "url": "https://x/d.zip"},
}


@pytest.fixture(autouse=True)
def vf_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VISIONFORGE_HOME", str(tmp_path))
    return tmp_path


def event_line(event: dict[str, Any]) -> str:
    return f"{runner.EVENT_PREFIX}{json.dumps(event)}\n"


class FakeExec:
    """`colab exec` streaming lines; `kill` ends the stream early, like a killed process."""

    def __init__(self, lines: list[str] | None, delay: float = 0.0) -> None:
        # None: the request was lost, nothing ever comes until the process is killed.
        self._lines = lines
        self._delay = delay
        self.killed = threading.Event()
        self.returncode: int | None = None
        self.stdin = None
        self.pid = 1
        self.stdout = self._stream()

    def _stream(self):  # type: ignore[no-untyped-def]
        if self._lines is None:
            self.killed.wait(10)
            return
        for line in self._lines:
            if self.killed.is_set():
                return
            time.sleep(self._delay)
            yield line

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = -9 if self.killed.is_set() else 0
        return self.returncode

    def kill(self) -> None:
        self.killed.set()


class FakeCli:
    """Records every CLI call; `new` succeeds unless told otherwise."""

    def __init__(
        self,
        exec_lines: list[str],
        new_ok: bool = True,
        delay: float = 0.0,
        warmups: list[str] | None = None,
        runs: list[list[str] | None] | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self.exec_lines = exec_lines
        # What each warm-up exec prints, in order; "" is a lost request (timeout).
        self.warmups = (
            list(warmups) if warmups is not None else [cj.WARMUP_MARKER + "\n"]
        )
        # What each runner exec streams; None is a runner that never starts.
        self.runs = list(runs) if runs is not None else None
        self.execs: list[FakeExec] = []
        self.new_ok = new_ok
        self.delay = delay
        self.exec: FakeExec | None = None
        self.script_seen: str | None = None
        self.sessions_out = "[colab] No active sessions found on server.\n"

    def run(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv[1:])
        command = argv[1]
        if command == "new" and not self.new_ok:
            return subprocess.CompletedProcess(
                argv, 1, "", "400 Bad Request: no quota for T4"
            )
        if command == "sessions":
            return subprocess.CompletedProcess(argv, 0, self.sessions_out, "")
        if command == "exec":
            out = self.warmups.pop(0) if self.warmups else ""
            if not out:
                return subprocess.CompletedProcess(argv, 124, "", "timeout after 60s")
            return subprocess.CompletedProcess(argv, 0, out, "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    def popen(self, argv: list[str], **kwargs: Any) -> FakeExec:
        self.calls.append(argv[1:])
        script = Path(argv[argv.index("-f") + 1])
        self.script_seen = script.read_text(encoding="utf-8")
        lines = self.runs.pop(0) if self.runs is not None else self.exec_lines
        self.exec = FakeExec(lines, self.delay)
        self.execs.append(self.exec)
        return self.exec

    def commands(self) -> list[str]:
        return [c[0] for c in self.calls]


def run(
    cli: FakeCli, cancelled: bool = False
) -> tuple[cj.RemoteResult, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    jobs = cj.ColabJobs(run=cli.run, popen=cli.popen)
    result = jobs.run_job("ana", "T4", PAYLOAD, events.append, lambda: cancelled)
    return result, events


# ── the protocol ──────────────────────────────────────────────────────────────


def test_parse_event_reads_prefixed_lines_and_ignores_the_rest() -> None:
    assert cj.parse_event(event_line({"event": "epoch_end", "epoch": 1})) == {
        "event": "epoch_end",
        "epoch": 1,
    }
    assert cj.parse_event("[colab] Session READY.\n") is None
    assert cj.parse_event(runner.EVENT_PREFIX + "{not json\n") is None
    assert cj.parse_event(runner.EVENT_PREFIX + "[1, 2]\n") is None


def test_build_script_carries_the_payload_and_the_runner() -> None:
    script = cj.build_script(PAYLOAD)
    header, _, body = script.partition("\n")
    blob = (
        header.split('"')[1] if '"' in header else script.splitlines()[1].split('"')[1]
    )
    assert json.loads(base64.b64decode(blob)) == PAYLOAD
    assert "def main(payload" in body
    compile(script, "runner.py", "exec")


def test_parse_sessions_reads_names_and_hardware() -> None:
    text = (
        "[vf-20261105-140000] https://x.prod.colab.dev | Hardware: T4 | Shape: Standard | Variant: GPU\n"
        "[?] https://y.prod.colab.dev | Hardware: CPU | Shape: Standard | Variant: DEFAULT\n"
    )
    sessions = cj.parse_sessions(text)
    assert sessions == [
        cj.RemoteSession("vf-20261105-140000", "T4"),
        cj.RemoteSession("?", "CPU"),
    ]
    assert sessions[0].ours
    assert not sessions[1].ours


def test_session_names_are_safe_for_the_cli() -> None:
    assert cj.is_session_name(cj.session_name())
    assert not cj.is_session_name("../x")
    assert not cj.is_session_name("-s")


# ── one job ───────────────────────────────────────────────────────────────────


def test_a_job_forwards_events_and_ends_with_the_report() -> None:
    cli = FakeCli(
        [
            "[colab] Executing...\n",
            event_line({"event": "start", "total_epochs": 2}),
            event_line({"event": "epoch_end", "epoch": 1, "val_loss": 0.5}),
            event_line(
                {"event": "job_end", "status": "ok", "report": {"accuracy": 0.9}}
            ),
        ]
    )
    result, events = run(cli)
    assert result == cj.RemoteResult("completed", report={"accuracy": 0.9})
    assert [e["event"] for e in events] == ["remote", "remote", "start", "epoch_end"]
    assert [e["stage"] for e in events[:2]] == ["allocated", "kernel_ready"]
    assert cli.commands() == ["new", "exec", "exec", "stop"]


def test_exec_runs_with_a_deadline_longer_than_any_session() -> None:
    cli = FakeCli([event_line({"event": "job_end", "status": "ok"})])
    run(cli)
    exec_call = [c for c in cli.calls if c[0] == "exec"][-1]
    assert float(exec_call[exec_call.index("--timeout") + 1]) >= 24 * 3600


def test_the_script_is_written_for_exec_and_removed_after(vf_home: Path) -> None:
    cli = FakeCli([event_line({"event": "job_end", "status": "ok"})])
    run(cli)
    assert cli.script_seen is not None and "PAYLOAD" in cli.script_seen
    assert list((vf_home / "colab" / "ana" / "jobs").glob("*.py")) == []


def test_a_remote_error_fails_the_job_with_its_message() -> None:
    cli = FakeCli(
        [
            event_line(
                {
                    "event": "job_end",
                    "status": "error",
                    "error": "FileNotFoundError: train",
                    "detail": "",
                }
            ),
        ]
    )
    result, _ = run(cli)
    assert result.outcome == "failed"
    assert result.error == "FileNotFoundError: train"
    assert cli.commands()[-1] == "stop"


def test_no_final_event_is_a_failure_even_if_exec_exits_zero() -> None:
    cli = FakeCli(
        [
            "Traceback (most recent call last):\n",
            "TimeoutError: Timeout waiting for reply\n",
        ]
    )
    result, _ = run(cli)
    assert result.outcome == "failed"
    assert result.error is not None
    assert "TimeoutError" in result.error


def test_a_refused_accelerator_still_stops_the_session() -> None:
    cli = FakeCli([], new_ok=False)
    result, events = run(cli)
    assert result.outcome == "failed"
    assert result.error is not None
    assert "no quota for T4" in result.error
    assert events == []
    assert cli.commands() == ["new", "stop"]


def test_stopping_kills_exec_and_releases_the_vm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cj, "CANCEL_POLL_S", 0.01)
    lines = [event_line({"event": "batch", "batch": i}) for i in range(500)]
    cli = FakeCli(lines, delay=0.005)
    events: list[dict[str, Any]] = []
    jobs = cj.ColabJobs(run=cli.run, popen=cli.popen)
    # The stop comes while the runner streams, after the warm-up answered.
    result = jobs.run_job("ana", "T4", PAYLOAD, events.append, lambda: len(events) > 5)
    assert result.outcome == "stopped"
    assert cli.exec is not None and cli.exec.killed.is_set()
    assert len(events) < 500
    assert cli.commands()[-1] == "stop"


def test_every_call_runs_in_the_profile_cli_home(vf_home: Path) -> None:
    envs: list[str] = []
    cli = FakeCli([event_line({"event": "job_end", "status": "ok"})])
    original = cli.run

    def run_with_env(
        argv: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        envs.append(kwargs["env"]["USERPROFILE"])
        return original(argv, **kwargs)

    jobs = cj.ColabJobs(run=run_with_env, popen=cli.popen)
    jobs.run_job("ana", "T4", PAYLOAD, lambda e: None, lambda: False)
    assert set(envs) == {str(vf_home / "colab" / "ana")}


def test_stop_session_refuses_names_that_are_not_names() -> None:
    cli = FakeCli([])
    jobs = cj.ColabJobs(run=cli.run, popen=cli.popen)
    assert jobs.stop_session("ana", "--all") is False
    assert cli.calls == []
    assert jobs.stop_session("ana", "vf-20261105-140000") is True


# ── the runner on the VM ──────────────────────────────────────────────────────


def test_single_child_steps_into_a_lone_top_folder(tmp_path: Path) -> None:
    (tmp_path / "ds" / "train").mkdir(parents=True)
    (tmp_path / "__MACOSX").mkdir()
    assert runner.single_child(tmp_path) == tmp_path / "ds"
    (tmp_path / "other").mkdir()
    assert runner.single_child(tmp_path) == tmp_path


def test_fetch_dataset_unzips_a_url(tmp_path: Path) -> None:
    archive = tmp_path / "src.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("coffee/train/a/1.png", b"x")
        zf.writestr("coffee/val/a/1.png", b"x")
    base = runner.fetch_dataset(
        {"kind": "url", "url": archive.as_uri()}, tmp_path / "vm"
    )
    assert base.name == "coffee"
    assert (base / "train" / "a" / "1.png").is_file()


def test_prepare_config_points_at_the_vm(tmp_path: Path) -> None:
    local: dict[str, Any] = {
        "name": "exp",
        "data": {
            "base_dir": "C:/datasets/coffee",
            "train_dir": "tr",
            "num_workers": 64,
        },
        "output": {"models_dir": "outputs/profiles/ana/models"},
        "device": {"kind": "multi_cuda", "gpu_ids": [0, 1]},
    }
    cfg = runner.prepare_config(local, tmp_path, "url")
    assert cfg["data"]["base_dir"] == str(tmp_path)
    assert cfg["data"]["train_dir"] == "tr", (
        "a zip keeps the folder names of the config"
    )
    assert cfg["data"]["num_workers"] <= (os.cpu_count() or 1)
    assert cfg["output"]["models_dir"].startswith(str(runner.REMOTE_ROOT))
    assert cfg["device"]["gpu_ids"] is None
    assert local["data"]["base_dir"] == "C:/datasets/coffee", (
        "the local config is not touched"
    )
    provider = runner.prepare_config(local, tmp_path, "provider")
    assert (provider["data"]["train_dir"], provider["data"]["val_dir"]) == (
        "train",
        "val",
    )


def _main_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, train: Any
) -> list[dict[str, Any]]:
    monkeypatch.setattr(runner, "REMOTE_ROOT", tmp_path)
    monkeypatch.setattr(runner, "install", lambda version: None)
    monkeypatch.setattr(runner, "fetch_dataset", lambda source, root: tmp_path)
    monkeypatch.setattr(runner, "train", train)
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    runner.main(PAYLOAD)
    return [
        e for e in map(cj.parse_event, out.getvalue().splitlines(keepends=True)) if e
    ]


def test_runner_ends_with_job_end_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = _main_events(monkeypatch, tmp_path, lambda cfg: {"accuracy": 1.0})
    assert [e.get("stage") for e in events[:3]] == ["install", "dataset", "train"]
    assert events[-1]["event"] == "job_end"
    assert events[-1]["status"] == "ok"
    assert events[-1]["report"] == {"accuracy": 1.0}


def test_runner_turns_any_failure_into_job_end_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(cfg: dict[str, Any]) -> None:
        raise ValueError("no train folder")

    events = _main_events(monkeypatch, tmp_path, boom)
    assert events[-1]["status"] == "error"
    assert events[-1]["error"] == "ValueError: no train folder"
    assert "Traceback" in events[-1]["traceback"]


def test_runner_never_calls_sys_exit() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "sys.exit" not in source.replace("never calls ``sys.exit``", "").replace(
        "the script never calls ``sys.exit``", ""
    )


# ── the route and the queue ───────────────────────────────────────────────────


def _config() -> dict[str, Any]:
    return {"name": "exp", "data": {"base_dir": "C:/does/not/matter"}}


def test_colab_run_needs_a_login() -> None:
    from visionforge.gui.server import app

    client = TestClient(app)
    body = {"config": _config(), "source": {"kind": "url", "url": "https://x/d.zip"}}
    assert client.post("/api/colab/experiment/run", json=body).status_code == 409


def test_colab_run_rejects_a_non_http_url() -> None:
    from visionforge.gui.server import app

    client = TestClient(app)
    body = {"config": _config(), "source": {"kind": "url", "url": "file:///etc/passwd"}}
    assert client.post("/api/colab/experiment/run", json=body).status_code == 422


def test_colab_run_is_queued_with_a_vm_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    monkeypatch.setattr(
        routes_mod._COLAB_CLI, "installed_version", lambda: cc.PINNED_VERSION
    )
    monkeypatch.setattr(routes_mod._COLAB_CLI, "logged_in", lambda slug: True)
    occupy_queue(routes_mod)
    try:
        client = TestClient(app)
        body = {
            "config": _config(),
            "source": {
                "kind": "provider",
                "provider": "torchvision",
                "dataset": "cifar10",
                "limit": 20,
            },
        }
        response = client.post("/api/colab/experiment/run", json=body)
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        pending = client.get("/api/queue").json()["pending"]
        assert pending[0]["strategy"] == "colab"
        assert pending[0]["stop_at"] == "vm"
    finally:
        release_queue(routes_mod)


def test_execute_colab_records_the_report_and_closes_the_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import visionforge.gui.api.routes as routes_mod

    class Jobs:
        def run_job(self, slug, accelerator, payload, on_event, cancelled):  # type: ignore[no-untyped-def]
            on_event({"event": "epoch_end", "epoch": 1})
            return cj.RemoteResult("completed", report={"accuracy": 0.8})

    monkeypatch.setattr(routes_mod, "_COLAB_JOBS", Jobs())

    async def go() -> list[dict[str, Any]]:
        routes_mod._event_queue = RunEventLog("r1")
        await routes_mod._execute_colab("r1", "default", "T4", PAYLOAD)
        await asyncio.sleep(0)
        return [e async for e in routes_mod._event_queue.read()]

    try:
        events = asyncio.run(go())
        assert events == [{"event": "epoch_end", "epoch": 1}]
        assert routes_mod._current_run is not None
        assert routes_mod._current_run["status"] == "completed"
        assert routes_mod._current_run["report"] == {"accuracy": 0.8}
    finally:
        release_queue(routes_mod)
        routes_mod._event_queue = None


def test_colab_run_still_rejects_a_bad_config_before_paying_for_a_vm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    monkeypatch.setattr(
        routes_mod._COLAB_CLI, "installed_version", lambda: cc.PINNED_VERSION
    )
    monkeypatch.setattr(routes_mod._COLAB_CLI, "logged_in", lambda slug: True)
    client = TestClient(app)
    body = {
        "config": {**_config(), "training": {"epochs": 0}},
        "source": {"kind": "url", "url": "https://x/d.zip"},
    }
    response = client.post("/api/colab/experiment/run", json=body)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][:3] == ["body", "config", "training"]


# ── a kernel that is slow to start (second real run, 2026-11-05) ──────────────

OK_END = [event_line({"event": "job_end", "status": "ok", "report": {"accuracy": 0.7}})]


def test_a_slow_kernel_is_warmed_up_before_the_runner() -> None:
    cli = FakeCli([], warmups=["", "", cj.WARMUP_MARKER + "\n"], runs=[OK_END])
    result, events = run(cli)
    assert result.outcome == "completed"
    assert cli.commands() == ["new", "exec", "exec", "exec", "exec", "stop"]
    assert len(cli.execs) == 1, "the runner is sent once the kernel answered"
    assert "kernel_ready" in [e.get("stage") for e in events]


def test_a_kernel_that_never_answers_fails_clearly_and_releases_the_vm() -> None:
    cli = FakeCli([], warmups=["", "", ""], runs=[OK_END])
    result, _ = run(cli)
    assert result.outcome == "failed"
    assert result.error is not None
    assert str(cj.WARMUP_ATTEMPTS) in result.error
    assert cli.execs == [], "no runner is sent to a kernel that never answered"
    assert cli.commands()[-1] == "stop"


def test_a_warm_up_without_the_marker_does_not_count() -> None:
    cli = FakeCli(
        [], warmups=["[colab] Executing...\n", cj.WARMUP_MARKER + "\n"], runs=[OK_END]
    )
    result, _ = run(cli)
    assert result.outcome == "completed"
    assert sum(c[0] == "exec" for c in cli.calls) == 3


def test_a_lost_runner_request_is_sent_again_on_the_same_vm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cj, "STARTUP_TIMEOUT_S", 0.2)
    monkeypatch.setattr(cj, "CANCEL_POLL_S", 0.02)
    cli = FakeCli([], runs=[None, OK_END])
    result, events = run(cli)
    assert result == cj.RemoteResult("completed", report={"accuracy": 0.7})
    assert cli.execs[0].killed.is_set()
    assert len(cli.execs) == 2
    assert "retry" in [e.get("stage") for e in events]
    assert cli.commands().count("new") == 1, "the retry reuses the VM"


def test_a_runner_that_never_starts_fails_after_the_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cj, "STARTUP_TIMEOUT_S", 0.2)
    monkeypatch.setattr(cj, "CANCEL_POLL_S", 0.02)
    cli = FakeCli([], runs=[None, None])
    result, _ = run(cli)
    assert result.outcome == "failed"
    assert result.error is not None
    assert str(cj.RUN_ATTEMPTS) in result.error
    assert all(e.killed.is_set() for e in cli.execs)
    assert cli.commands()[-1] == "stop"


def test_a_slow_runner_that_already_sent_an_event_is_not_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cj, "STARTUP_TIMEOUT_S", 0.2)
    monkeypatch.setattr(cj, "CANCEL_POLL_S", 0.02)
    slow = (
        [event_line({"event": "remote", "stage": "install"})]
        + [event_line({"event": "batch", "batch": i}) for i in range(30)]
        + OK_END
    )
    cli = FakeCli([], runs=[slow], delay=0.02)
    result, _ = run(cli)
    assert result.outcome == "completed"
    assert len(cli.execs) == 1


def test_stopping_during_the_warm_up_skips_the_runner() -> None:
    cli = FakeCli([], runs=[OK_END])
    result, _ = run(cli, cancelled=True)
    assert result.outcome == "stopped"
    assert cli.execs == []
    assert cli.commands() == ["new", "stop"]


# ── the class count of a dataset that only the VM sees ────────────────────────


def _folders(base: Path, split: str, names: list[str]) -> None:
    for name in names:
        (base / split / name).mkdir(parents=True, exist_ok=True)


def test_num_classes_follows_the_remote_train_folders(tmp_path: Path) -> None:
    _folders(tmp_path, "train", [str(i) for i in range(10)])
    cfg: dict[str, Any] = {
        "task": "multiclass",
        "data": {"base_dir": str(tmp_path), "train_dir": "train"},
        "model": {"num_classes": 2},
    }
    assert runner.match_num_classes(cfg) == {"old": 2, "new": 10}
    assert cfg["model"]["num_classes"] == 10


def test_num_classes_already_right_is_left_alone(tmp_path: Path) -> None:
    _folders(tmp_path, "train", ["a", "b", "c"])
    cfg: dict[str, Any] = {
        "data": {"base_dir": str(tmp_path), "train_dir": "train"},
        "model": {"num_classes": 3},
    }
    assert runner.match_num_classes(cfg) is None


def test_a_binary_task_keeps_its_count_and_its_error(tmp_path: Path) -> None:
    _folders(tmp_path, "train", ["a", "b", "c"])
    cfg: dict[str, Any] = {
        "task": "binary",
        "data": {"base_dir": str(tmp_path), "train_dir": "train"},
        "model": {"num_classes": 2},
    }
    assert runner.match_num_classes(cfg) is None
    assert cfg["model"]["num_classes"] == 2


def test_hidden_folders_and_files_are_not_classes(tmp_path: Path) -> None:
    _folders(tmp_path, "train", ["a", "b", ".ipynb_checkpoints"])
    (tmp_path / "train" / "notes.txt").write_text("x", encoding="utf-8")
    cfg: dict[str, Any] = {
        "data": {"base_dir": str(tmp_path), "train_dir": "train"},
        "model": {"num_classes": 5},
    }
    assert runner.match_num_classes(cfg) == {"old": 5, "new": 2}


def test_a_missing_train_split_is_left_to_the_training_error(tmp_path: Path) -> None:
    cfg: dict[str, Any] = {
        "data": {"base_dir": str(tmp_path), "train_dir": "train"},
        "model": {"num_classes": 2},
    }
    assert runner.match_num_classes(cfg) is None


def test_runner_reports_the_class_count_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _folders(tmp_path, "train", [str(i) for i in range(10)])
    monkeypatch.setattr(
        runner,
        "prepare_config",
        lambda config, base, kind: {
            "data": {"base_dir": str(tmp_path), "train_dir": "train"},
            "model": {"num_classes": 2},
            "device": {"kind": "cpu"},
        },
    )
    seen: dict[str, Any] = {}

    def train(cfg: dict[str, Any]) -> dict[str, Any]:
        seen.update(cfg)
        return {}

    events = _main_events(monkeypatch, tmp_path, train)
    change = next(e for e in events if e.get("stage") == "num_classes")
    assert (change["old"], change["new"]) == (2, 10)
    assert seen["model"]["num_classes"] == 10
