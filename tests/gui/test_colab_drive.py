"""Datasets from Google Drive on Colab (ADR-164) against a fake CLI: nothing reaches Google."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.helpers import wait_until
from visionforge.gui.api import colab_connection as cc
from visionforge.gui.api import colab_drive as cd
from visionforge.gui.api import colab_jobs as cj
from visionforge.gui.api import colab_remote_runner as runner

from .test_colab_jobs import OK_END, FakeCli

CONSENT = "https://accounts.google.com/o/oauth2/v2/auth?state=secret-token"
DRIVE_PAYLOAD = {
    "version": "0.21.0",
    "config": {"name": "exp"},
    "source": {"kind": "drive", "path": "/content/drive/MyDrive/ds", "files": 3},
}
# The lines `colab drivemount` prints, as seen in the spikes (design doc, 2026-11-05).
ASK = [
    "[colab] Mounting Google Drive to '/content/drive' on vf-x...\n",
    "[colab] REQUIRED: Google Drive Authorization needed.\n",
    "Please visit:\n",
    f"{CONSENT}\n",
]
OK = [
    "[colab] Authorizing VM...\n",
    "[colab] Credentials propagated. Resuming mount...\n",
    "Mounted at /content/drive\n",
]
REFUSED = [
    "[colab] Authorizing VM...\n",
    "[colab] Error propagating: 400 \n",
    "<!DOCTYPE html>\n",
]
ENTER = "<enter>"  # the fake blocks here until the server writes to stdin
HANG = "<hang>"  # the fake blocks here until it is killed


@pytest.fixture(autouse=True)
def vf_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VISIONFORGE_HOME", str(tmp_path))
    return tmp_path


class FakeStdin:
    def __init__(self) -> None:
        self.written: list[str] = []
        self.got = threading.Event()

    def write(self, text: str) -> int:
        self.written.append(text)
        self.got.set()
        return len(text)

    def flush(self) -> None:
        pass


class FakeMount:
    """`colab drivemount`: prints its lines, waits for Enter or for a kill where told."""

    def __init__(self, script: list[str]) -> None:
        self._script = script
        self.stdin = FakeStdin()
        self.killed = threading.Event()
        self.returncode: int | None = None
        self.pid = 2
        self.stdout = self._stream()

    def _stream(self):  # type: ignore[no-untyped-def]
        for line in self._script:
            if self.killed.is_set():
                return
            if line == ENTER:
                while not (self.stdin.got.wait(0.01) or self.killed.is_set()):
                    pass
                continue
            if line == HANG:
                self.killed.wait(10)
                return
            yield line

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = -9 if self.killed.is_set() else 0
        return self.returncode

    def kill(self) -> None:
        self.killed.set()


class DriveCli(FakeCli):
    """The fake CLI of the phase 2 tests, plus `drivemount` playing given scripts."""

    def __init__(self, mounts: list[list[str]], **kwargs: Any) -> None:
        super().__init__([], runs=[OK_END], **kwargs)
        self.mount_scripts = list(mounts)
        self.mounts: list[FakeMount] = []

    def popen(self, argv: list[str], **kwargs: Any) -> Any:
        if argv[1] != "drivemount":
            return super().popen(argv, **kwargs)
        self.calls.append(argv[1:])
        assert kwargs["stdin"] is subprocess.PIPE
        mount = FakeMount(self.mount_scripts.pop(0))
        self.mounts.append(mount)
        return mount


def run_drive(
    cli: DriveCli,
    grant: bool = True,
    cancel_on_consent: bool = False,
    payload: dict[str, Any] = DRIVE_PAYLOAD,
) -> tuple[cj.RemoteResult, list[dict[str, Any]]]:
    """Run a job; each consent URL is answered the way the GUI would, from another thread."""
    events: list[dict[str, Any]] = []
    consent = cd.DriveConsent()
    cancel = threading.Event()

    def on_event(event: dict[str, Any]) -> None:
        events.append(event)
        if event.get("event") != "drive_consent_required":
            return
        if cancel_on_consent:
            cancel.set()
        elif grant:
            threading.Thread(target=_grant_when_waiting, args=(consent,)).start()

    jobs = cj.ColabJobs(run=cli.run, popen=cli.popen)
    result = jobs.run_job(
        "ana", "T4", payload, on_event, cancel.is_set, consent=consent
    )
    return result, events


def _grant_when_waiting(consent: cd.DriveConsent) -> None:
    wait_until(
        consent.grant,
        timeout=5,
        interval=0.01,
        message="the consent gate to be waiting",
    )


def consent_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in events if e.get("event") == "drive_consent_required"]


# ── the mount, through the GUI ────────────────────────────────────────────────


def test_consent_given_mounts_the_drive_then_trains() -> None:
    cli = DriveCli([[*ASK, ENTER, *OK]])
    result, events = run_drive(cli)

    assert result.outcome == "completed"
    assert cli.commands() == ["new", "exec", "drivemount", "exec", "stop"]
    asked = consent_events(events)
    assert [(e["url"], e["attempt"], e["attempts"]) for e in asked] == [(CONSENT, 1, 3)]
    assert cli.mounts[0].stdin.written == ["\n"]
    stages = [e.get("stage") for e in events if e.get("event") == "remote"]
    assert stages.index("drive_consent_sent") < stages.index("drive_mounted")


def test_a_runtime_already_allowed_mounts_without_asking() -> None:
    cli = DriveCli([["[colab] Credentials propagated. Resuming mount...\n", *OK[2:]]])
    result, events = run_drive(cli)
    assert result.outcome == "completed"
    assert consent_events(events) == []
    assert cli.mounts[0].stdin.written == []


def test_a_400_asks_again_with_a_new_url_and_then_mounts() -> None:
    cli = DriveCli([[*ASK, ENTER, *REFUSED, HANG], [*ASK, ENTER, *OK]])
    result, events = run_drive(cli)

    assert result.outcome == "completed"
    assert [e["attempt"] for e in consent_events(events)] == [1, 2]
    # The CLI hangs after the 400; the server kills it instead of waiting.
    assert cli.mounts[0].killed.is_set()
    assert cli.commands() == ["new", "exec", "drivemount", "drivemount", "exec", "stop"]


def test_three_refusals_give_up_and_release_the_vm() -> None:
    cli = DriveCli([[*ASK, ENTER, *REFUSED, HANG] for _ in range(3)])
    result, events = run_drive(cli)

    assert result.outcome == "failed"
    assert result.error is not None and "3" in result.error
    assert [e["attempt"] for e in consent_events(events)] == [1, 2, 3]
    assert cli.commands() == [
        "new",
        "exec",
        "drivemount",
        "drivemount",
        "drivemount",
        "stop",
    ]


def test_a_consent_never_answered_times_out_on_its_own_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Neither the focus nor "I already allowed" ever arrives.
    monkeypatch.setattr(cj, "DRIVE_CONSENT_TIMEOUT_S", 0.05)
    cli = DriveCli([[*ASK, ENTER, *OK] for _ in range(3)])
    result, events = run_drive(cli, grant=False)

    assert result.outcome == "failed"
    assert len(consent_events(events)) == 3
    assert all(m.stdin.written == [] for m in cli.mounts)
    # The runner was never sent, so its startup watchdog never ran.
    assert "exec" not in cli.commands()[2:]


def test_stopping_while_waiting_for_the_consent_stops_the_job() -> None:
    cli = DriveCli([[*ASK, ENTER, *OK]])
    result, _ = run_drive(cli, cancel_on_consent=True)
    assert result.outcome == "stopped"
    assert cli.commands()[-1] == "stop"
    assert cli.mounts[0].stdin.written == []


def test_a_mount_that_fails_otherwise_says_why() -> None:
    cli = DriveCli([["ValueError: mountpoint must not already contain files\n"]])
    result, _ = run_drive(cli)
    assert result.outcome == "failed"
    assert result.error is not None and "mountpoint" in result.error


def test_the_consent_url_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    from loguru import logger

    handler = logger.add(caplog.handler, level="DEBUG")
    try:
        run_drive(DriveCli([[*ASK, ENTER, *REFUSED, HANG], [*ASK, ENTER, *OK]]))
    finally:
        logger.remove(handler)
    assert "secret-token" not in caplog.text


def test_a_job_without_drive_never_mounts() -> None:
    payload = {**DRIVE_PAYLOAD, "source": {"kind": "url", "url": "https://x/d.zip"}}
    cli = DriveCli([])
    result, _ = run_drive(cli, payload=payload)
    assert result.outcome == "completed"
    assert "drivemount" not in cli.commands()


def test_exporting_to_drive_mounts_it_even_for_another_source() -> None:
    payload = {
        **DRIVE_PAYLOAD,
        "source": {"kind": "url", "url": "https://x/d.zip"},
        "drive_export": True,
    }
    cli = DriveCli([[*ASK, ENTER, *OK]])
    result, _ = run_drive(cli, payload=payload)
    assert result.outcome == "completed"
    assert "drivemount" in cli.commands()


# ── DriveConsent ──────────────────────────────────────────────────────────────


def test_a_grant_with_nobody_waiting_is_refused() -> None:
    consent = cd.DriveConsent()
    assert consent.grant() is False
    # ...so it does not answer the next consent page in advance.
    assert consent.wait(0.05, lambda: False) == "timeout"


def test_wait_returns_granted_cancelled_or_timeout() -> None:
    consent = cd.DriveConsent()
    threading.Thread(target=_grant_when_waiting, args=(consent,)).start()
    assert consent.wait(5, lambda: False) == "granted"
    assert consent.waiting is False
    assert consent.wait(5, lambda: True, poll=0.01) == "cancelled"
    assert consent.wait(0.02, lambda: False, poll=0.01) == "timeout"


# ── the path on this machine ──────────────────────────────────────────────────


def test_consent_url_finds_the_google_page_only() -> None:
    assert cd.consent_url(f"{CONSENT}\n") == CONSENT
    assert cd.consent_url("Please visit:\n") is None
    assert cd.consent_url("https://colab.research.google.com/x\n") is None


def test_a_path_under_a_drive_letter_my_drive_is_translated() -> None:
    remote = cd.remote_path(PureWindowsPath("G:/Meu Drive/datasets/flores"), [])
    assert remote == PurePosixPath("/content/drive/MyDrive/datasets/flores")
    remote = cd.remote_path(PureWindowsPath("H:/my drive/x"), [])
    assert remote == PurePosixPath("/content/drive/MyDrive/x")


def test_a_mirrored_my_drive_under_the_home_folder_is_translated() -> None:
    remote = cd.remote_path(Path.home() / "My Drive" / "ds", [])
    assert remote == PurePosixPath("/content/drive/MyDrive/ds")


def test_a_my_drive_folder_elsewhere_is_not_guessed(tmp_path: Path) -> None:
    assert cd.remote_path(tmp_path / "Meu Drive" / "ds", []) is None
    assert cd.remote_path(PureWindowsPath("D:/datasets/flores"), []) is None


def test_the_manual_root_wins(tmp_path: Path) -> None:
    root = tmp_path / "Mi unidad"
    remote = cd.remote_path(
        root / "a" / "b", cd.my_drive_roots(str(root), lambda: None)
    )
    assert remote == PurePosixPath("/content/drive/MyDrive/a/b")


def test_the_registry_mount_point_gives_both_folder_names() -> None:
    roots = cd.my_drive_roots(None, lambda: "X")
    assert roots == [PureWindowsPath("X:\\My Drive"), PureWindowsPath("X:\\Meu Drive")]
    remote = cd.remote_path(PureWindowsPath("X:/Meu Drive/ds"), roots)
    assert remote == PurePosixPath("/content/drive/MyDrive/ds")


def test_count_files_skips_hidden_ones(tmp_path: Path) -> None:
    (tmp_path / "train" / "a").mkdir(parents=True)
    (tmp_path / "train" / "a" / "1.png").write_bytes(b"x")
    (tmp_path / "train" / ".DS_Store").write_bytes(b"x")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_bytes(b"x")
    assert cd.count_files(tmp_path) == 1
    assert runner.count_files(tmp_path) == 1


# ── the routes ────────────────────────────────────────────────────────────────


def _dataset(root: Path) -> Path:
    folder = root / "Meu Drive" / "flores"
    (folder / "train" / "a").mkdir(parents=True)
    for i in range(3):
        (folder / "train" / "a" / f"{i}.png").write_bytes(b"x")
    return folder


def test_drive_source_translates_the_form_folder(tmp_path: Path) -> None:
    import visionforge.gui.api.routes as routes_mod

    folder = _dataset(tmp_path)
    source = routes_mod._drive_source(
        {"data": {"base_dir": str(folder)}}, str(tmp_path / "Meu Drive")
    )
    assert source == {
        "kind": "drive",
        "path": "/content/drive/MyDrive/flores",
        "files": 3,
    }


def test_colab_run_refuses_a_folder_outside_the_drive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    monkeypatch.setattr(
        routes_mod._COLAB_CLI, "installed_version", lambda: cc.PINNED_VERSION
    )
    monkeypatch.setattr(routes_mod._COLAB_CLI, "logged_in", lambda slug: True)
    monkeypatch.setattr(routes_mod, "my_drive_roots", lambda manual: [])
    body = {
        "config": {"name": "exp", "data": {"base_dir": str(tmp_path / "ds")}},
        "source": {"kind": "drive"},
    }
    response = TestClient(app).post("/api/colab/experiment/run", json=body)
    assert response.status_code == 400
    assert "Google Drive" in response.json()["detail"]


def test_colab_run_refuses_an_empty_drive_folder(tmp_path: Path) -> None:
    from fastapi import HTTPException

    import visionforge.gui.api.routes as routes_mod

    empty = tmp_path / "Meu Drive" / "vazio"
    empty.mkdir(parents=True)
    with pytest.raises(HTTPException) as caught:
        routes_mod._drive_source(
            {"data": {"base_dir": str(empty)}}, str(tmp_path / "Meu Drive")
        )
    assert caught.value.status_code == 400


def test_continue_reaches_only_a_job_that_waits() -> None:
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    client = TestClient(app)
    assert (
        client.post("/api/colab/drive/continue", json={"run_id": "nope"}).status_code
        == 404
    )
    consent = cd.DriveConsent()
    routes_mod._DRIVE_CONSENTS["r-drive"] = consent
    answers: list[str] = []
    waiter = threading.Thread(
        target=lambda: answers.append(consent.wait(5, lambda: False))
    )
    try:
        waiter.start()
        wait_until(
            lambda: consent.waiting,
            timeout=5,
            interval=0.01,
            message="the consent gate to be waiting",
        )
        response = client.post("/api/colab/drive/continue", json={"run_id": "r-drive"})
        assert response.status_code == 200
        waiter.join(5)
        assert answers == ["granted"]
        # Once answered, a second call has nothing to answer.
        again = client.post("/api/colab/drive/continue", json={"run_id": "r-drive"})
        assert again.status_code == 404
    finally:
        routes_mod._DRIVE_CONSENTS.pop("r-drive", None)


# ── the runner on the VM ──────────────────────────────────────────────────────


def test_the_runner_copies_the_drive_folder_and_checks_the_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drive = tmp_path / "MyDrive"
    monkeypatch.setattr(runner, "DRIVE_ROOT", drive)
    folder = _dataset(drive)
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(runner, "emit", emitted.append)

    base = runner.fetch_dataset(
        {"kind": "drive", "path": str(folder), "files": 3}, tmp_path / "vm"
    )
    assert base == tmp_path / "vm" / "dataset"
    assert runner.count_files(base) == 3
    assert emitted[-1]["stage"] == "drive_copy"
    assert (emitted[-1]["files"], emitted[-1]["expected"]) == (3, 3)


def test_the_runner_refuses_a_folder_drive_has_not_finished_syncing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drive = tmp_path / "MyDrive"
    monkeypatch.setattr(runner, "DRIVE_ROOT", drive)
    monkeypatch.setattr(runner, "emit", lambda event: None)
    folder = _dataset(drive)
    with pytest.raises(RuntimeError, match="3 of the 5"):
        runner.fetch_dataset(
            {"kind": "drive", "path": str(folder), "files": 5}, tmp_path / "vm"
        )


def test_the_runner_reads_only_inside_the_mounted_drive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "DRIVE_ROOT", tmp_path / "MyDrive")
    outside = _dataset(tmp_path / "elsewhere")
    with pytest.raises(FileNotFoundError):
        runner.fetch_dataset(
            {"kind": "drive", "path": str(outside), "files": 3}, tmp_path / "vm"
        )


def test_export_to_drive_keeps_a_copy_of_the_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner, "DRIVE_RUNS", tmp_path / "MyDrive" / "VisionForge" / "runs"
    )
    bundle = tmp_path / "run-bundle.zip"
    bundle.write_bytes(b"zip")
    saved = runner.export_to_drive({"path": str(bundle), "size": 3}, "exp")
    assert saved is not None and Path(saved).read_bytes() == b"zip"
    assert Path(saved).parent == tmp_path / "MyDrive" / "VisionForge" / "runs"
    assert runner.export_to_drive(None, "exp") is None
