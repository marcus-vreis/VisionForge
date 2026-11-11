"""Offline mode keeps VisionForge off the network (ADR-189).

With ``VISIONFORGE_OFFLINE=1`` or ``--offline`` the SDKs are told to stay
offline and every socket bound for a non-loopback host is refused. The training
test proves the claim the README makes: one epoch on CPU, with the guard on,
needs no network at all.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import visionforge.__main__ as cli
from visionforge.utils import third_party_privacy as tpp
from visionforge.utils.third_party_privacy import (
    OFFLINE_ENV,
    OFFLINE_SWITCHES,
    NetworkBlockedError,
    enable_offline_mode,
    install_socket_guard,
    offline_requested,
    remove_socket_guard,
)

_SRC = Path(__file__).resolve().parents[2] / "src"
# TEST-NET-3 (RFC 5737): a documentation address that routes nowhere.
_EXTERNAL_IP = "203.0.113.7"
_LOOPBACK = "127.0.0.1"
_EXTERNAL_NAME = "example.com"
_NO_BROWSER = "--no-browser"
_SERVER = "server"
_ENABLED = "enable_offline_mode"


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Offline mode for one test; the environment and sockets come back after."""
    for name in (OFFLINE_ENV, *OFFLINE_SWITCHES):
        monkeypatch.setenv(name, "0")
    enable_offline_mode()
    try:
        yield
    finally:
        remove_socket_guard()


_OFFLINE = pytest.mark.usefixtures("offline")


@contextlib.contextmanager
def _listener() -> Iterator[int]:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((_LOOPBACK, 0))
    server.listen(4)
    try:
        yield server.getsockname()[1]
    finally:
        server.close()


@pytest.mark.parametrize("host", [_EXTERNAL_NAME, _EXTERNAL_IP])
@_OFFLINE
def test_a_connection_to_an_external_host_is_refused(host: str) -> None:
    address = (host, 80)
    with pytest.raises(NetworkBlockedError):
        socket.create_connection(address, timeout=1)


@_OFFLINE
def test_a_datagram_to_an_external_host_is_refused() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        with pytest.raises(NetworkBlockedError):
            sock.sendto(b"x", (_EXTERNAL_IP, 53))


@_OFFLINE
def test_connect_ex_to_an_external_host_is_refused() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(NetworkBlockedError):
            sock.connect_ex((_EXTERNAL_IP, 80))


@_OFFLINE
def test_a_lookup_of_an_external_name_is_refused_before_any_dns_query() -> None:
    with pytest.raises(NetworkBlockedError):
        socket.gethostbyname(_EXTERNAL_NAME)


@pytest.mark.parametrize("host", [_LOOPBACK, "localhost"])
@_OFFLINE
def test_loopback_still_connects(host: str) -> None:
    with _listener() as port:
        with socket.create_connection((host, port), timeout=5):
            pass
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
            assert stream.connect_ex((_LOOPBACK, port)) == 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        assert sock.sendto(b"x", (_LOOPBACK, port)) == 1
    assert socket.gethostbyname("localhost").startswith("127.")
    assert socket.gethostbyname_ex(_LOOPBACK)[2] == [_LOOPBACK]


@_OFFLINE
def test_offline_mode_sets_the_sdk_switches() -> None:
    assert os.environ[OFFLINE_ENV] == "1"
    assert {name: os.environ[name] for name in OFFLINE_SWITCHES} == {
        "HF_HUB_OFFLINE": "1",
        "YOLO_OFFLINE": "1",
        "VISIONFORGE_NO_UPDATE_CHECK": "1",
    }


def test_the_guard_installs_once_and_comes_off() -> None:
    original = socket.getaddrinfo
    install_socket_guard()
    guarded = socket.getaddrinfo
    install_socket_guard()
    try:
        assert socket.getaddrinfo is guarded
        assert guarded is not original
    finally:
        remove_socket_guard()
    remove_socket_guard()
    assert socket.getaddrinfo is original


@pytest.mark.parametrize(
    ("value", "expected"), [("1", True), ("yes", True), ("0", False), ("", False)]
)
def test_offline_is_requested_by_any_value_but_empty_or_zero(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
) -> None:
    monkeypatch.setenv(OFFLINE_ENV, value)
    assert offline_requested() is expected


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        (b"127.0.0.1", True),
        ("::1", True),
        ("LOCALHOST", True),
        ("fe80::1%3", False),
        (_EXTERNAL_NAME, False),
        (80, False),
    ],
)
def test_only_loopback_hosts_count_as_local(host: object, expected: bool) -> None:
    assert tpp._is_loopback_host(host) is expected


def test_a_socket_outside_the_internet_families_is_not_checked() -> None:
    local_socket = SimpleNamespace(family=-1)
    tpp._check_destination(local_socket, "/run/visionforge.sock")  # type: ignore[arg-type]
    assert local_socket.family == -1


def _synthetic_imagefolder(base: Path) -> None:
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        for index, name in enumerate(("class_a", "class_b")):
            folder = base / split / name
            folder.mkdir(parents=True)
            for i in range(3):
                pixels = rng.integers(0, 80, (32, 32, 3), dtype=np.uint8)
                pixels[..., index] = 200
                Image.fromarray(pixels, "RGB").save(folder / f"img{i}.png")


@_OFFLINE
def test_one_epoch_of_classification_on_cpu_trains_under_the_guard(
    tmp_path: Path,
) -> None:
    from visionforge.blocks.classification import ClassificationBlock
    from visionforge.utils.config import ExperimentConfig

    base = tmp_path / "ds"
    _synthetic_imagefolder(base)
    config = ExperimentConfig.model_validate(
        {
            "name": "offline_e2e",
            "task": "multiclass",
            "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
            "training": {"epochs": 1, "batch_size": 2, "seed": 0},
            "data": {
                "base_dir": str(base),
                "num_workers": 0,
                "pin_memory": False,
                "transforms": {"image_size": 32},
            },
            "output": {"models_dir": str(tmp_path / "models")},
            "device": {"kind": "cpu"},
        }
    )
    block = ClassificationBlock()

    block.setup(config)
    block.run()

    record = json.loads(
        next((tmp_path / "models").rglob("run.json")).read_text(encoding="utf-8")
    )
    assert record["status"] == "completed"
    assert record["metrics"]["total_epochs"] == 1


def _gui_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> list[str]:
    """Run ``main()`` for the gui command; return the order of the setup calls."""
    import visionforge.gui.server as server
    import visionforge.utils.logger as vf_logger

    calls: list[str] = []
    monkeypatch.setattr(cli, "_use_utf8_console", lambda: None)  # pytest's streams
    monkeypatch.setattr(vf_logger, "setup_logger", lambda: None)  # pytest's sinks
    monkeypatch.setattr(tpp, "enable_offline_mode", lambda: calls.append(_ENABLED))
    monkeypatch.setattr(server, "start_server", lambda **_kw: calls.append(_SERVER))
    monkeypatch.setattr(sys, "argv", ["visionforge", *argv])
    cli.main()
    return calls


def test_the_offline_flag_turns_offline_mode_on_before_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(OFFLINE_ENV, raising=False)
    calls = _gui_main(monkeypatch, ["gui", "--offline", _NO_BROWSER])
    assert calls == [_ENABLED, _SERVER]


def test_the_offline_variable_turns_offline_mode_on_without_the_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OFFLINE_ENV, "1")
    calls = _gui_main(monkeypatch, ["gui", _NO_BROWSER])
    assert calls == [_ENABLED, _SERVER]


def test_without_offline_mode_the_guard_stays_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(OFFLINE_ENV, raising=False)
    calls = _gui_main(monkeypatch, ["gui", _NO_BROWSER])
    assert calls == [_SERVER]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_LOOPBACK, 0))
        port: int = sock.getsockname()[1]
        return port


def _health(port: int, deadline: float) -> dict[str, str]:
    url = f"http://{_LOOPBACK}:{port}/api/health"  # NOSONAR python:S5332 - loopback
    while True:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310
                body: dict[str, str] = json.loads(response.read())
                return body
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


def test_the_gui_starts_offline_and_answers_health(tmp_path: Path) -> None:
    port = _free_port()
    env = {**os.environ, "PYTHONPATH": str(_SRC), "CUDA_VISIBLE_DEVICES": "-1"}
    env.pop(OFFLINE_ENV, None)
    command = [
        sys.executable,
        "-m",
        "visionforge",
        "gui",
        "--offline",
        _NO_BROWSER,
        "--port",
        str(port),
    ]
    process = subprocess.Popen(  # noqa: S603 - fixed interpreter and arguments
        command,
        cwd=tmp_path,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        body = _health(port, time.monotonic() + 120)
    finally:
        process.terminate()
        process.wait(timeout=30)
    assert "version" in body
