"""Switch off the usage telemetry of the third-party SDKs VisionForge bundles.

The README promises that nothing leaves this computer unless you ask for it
(ADR-120, ADR-170). Some SDKs we depend on report usage on their own, so this
module turns that off, once, before any of them is imported.

``apply_opt_outs`` is idempotent and never overrides a value the user already
set: a user who wants a SDK's telemetry on keeps it on.

``NETWORK_SDKS`` lists every dependency that can reach the network, and says
how it is covered. ``tests/utils/test_third_party_privacy.py`` fails when
``pyproject.toml`` gains a known network SDK that has no entry here.

Offline mode (``--offline`` or ``VISIONFORGE_OFFLINE=1``, ADR-189) goes further:
it tells the SDKs to stay offline and refuses, in this process, every socket and
name lookup bound for a host other than this computer.
"""

from __future__ import annotations

import contextlib
import ipaddress
import os
import socket
import sys
from typing import Any

# Environment switches honoured by the SDKs, set only when the user did not.
_ENV_OPT_OUTS: dict[str, str] = {
    "HF_HUB_DISABLE_TELEMETRY": "1",  # huggingface_hub download telemetry
    "DO_NOT_TRACK": "1",  # community convention, read by several SDKs
    "MLFLOW_DISABLE_TELEMETRY": "true",  # MLflow usage data (3.2+)
    "DVC_NO_ANALYTICS": "1",  # DVC; dvc_export.py also sets it per subprocess
}

# Dependency name -> how it is covered. Not YOLO_OFFLINE: that would also stop
# the weights download the user asks for (offline mode sets it, ADR-189).
NETWORK_SDKS: dict[str, str] = {
    "ultralytics": "events.enabled=False in-process (events for train/val/export)",
    "huggingface-hub": "HF_HUB_DISABLE_TELEMETRY",
    "datasets": "HF_HUB_DISABLE_TELEMETRY (uses huggingface_hub)",
    "mlflow": "MLFLOW_DISABLE_TELEMETRY",
    "dvc": "DVC_NO_ANALYTICS",
    "wandb": "opt-in only: sends nothing until the user asks for it",
    "comet-ml": "opt-in only; auto-logging, Sentry and SDK reports off (ADR-200)",
    "roboflow": "dataset download the user asks for; no telemetry",
    "kaggle": "dataset download the user asks for; no telemetry",
}


def apply_env_opt_outs() -> None:
    """Set only the environment switches; imports no SDK.

    The GUI server calls this on import: it no longer imports the trainers, so
    it must not pay for ultralytics (and torch) just to switch telemetry off.
    """
    for name, value in _ENV_OPT_OUTS.items():
        os.environ.setdefault(name, value)


def apply_opt_outs() -> None:
    """Turn off third-party telemetry; keep any value the user already set."""
    apply_env_opt_outs()
    _disable_ultralytics_sync()


def _disable_ultralytics_sync() -> None:
    """Stop ultralytics' usage events in this process only.

    Its ``settings.update({"sync": False})`` would write the user's global
    ultralytics settings file and switch the events off for every other project
    on the machine too. The events go through one ``Events`` instance whose
    ``enabled`` flag is read on each call, so turning that off here keeps the
    choice inside VisionForge.

    The import goes with stdout pointed at stderr: on a machine without its
    settings file, ultralytics prints "Creating new Ultralytics Settings" on
    stdout, which broke the JSON that ``audit-data`` and ``bundle`` print there.
    """
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from ultralytics.utils.events import events

        events.enabled = False
    except Exception:  # noqa: BLE001 - absent or broken SDK must not stop the app
        return


# --- Offline mode (ADR-189) ---------------------------------------------------

OFFLINE_ENV = "VISIONFORGE_OFFLINE"

# Set, overriding the user, when offline mode is on: it is a guarantee, not a
# default. YOLO_OFFLINE also stops the weights download, which is the point here.
OFFLINE_SWITCHES: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "YOLO_OFFLINE": "1",
    "VISIONFORGE_NO_UPDATE_CHECK": "1",
}

_LOCAL_NAMES = frozenset({"localhost", "localhost.localdomain", ""})
_ORIGINALS: dict[str, Any] = {}


class NetworkBlockedError(ConnectionRefusedError):
    """A connection or name lookup that offline mode refused."""


def offline_requested() -> bool:
    """``VISIONFORGE_OFFLINE`` set to anything but empty or 0."""
    return os.environ.get(OFFLINE_ENV, "").strip() not in ("", "0")


def enable_offline_mode() -> None:
    """Switch the SDKs offline and refuse every non-loopback socket.

    Call it before any SDK is imported: they read their switches on import.
    ``VISIONFORGE_OFFLINE`` is set too, so a child process (the updater's
    relaunch, a worker) starts offline as well; the socket guard itself covers
    only this process.
    """
    os.environ[OFFLINE_ENV] = "1"
    for name, value in OFFLINE_SWITCHES.items():
        os.environ[name] = value
    install_socket_guard()


def install_socket_guard() -> None:
    """Refuse connections and lookups that would leave this computer. Idempotent."""
    if _ORIGINALS:
        return
    _ORIGINALS.update(
        connect=socket.socket.connect,
        connect_ex=socket.socket.connect_ex,
        sendto=socket.socket.sendto,
        getaddrinfo=socket.getaddrinfo,
        gethostbyname=socket.gethostbyname,
        gethostbyname_ex=socket.gethostbyname_ex,
    )
    socket.socket.connect = _guarded_connect  # type: ignore[method-assign,assignment]
    socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign,assignment]
    socket.socket.sendto = _guarded_sendto  # type: ignore[method-assign,assignment]
    socket.getaddrinfo = _guarded_getaddrinfo  # type: ignore[assignment]
    socket.gethostbyname = _guarded_gethostbyname
    socket.gethostbyname_ex = _guarded_gethostbyname_ex


def remove_socket_guard() -> None:
    """Put the original socket functions back. For tests."""
    if not _ORIGINALS:
        return
    socket.socket.connect = _ORIGINALS["connect"]  # type: ignore[method-assign]
    socket.socket.connect_ex = _ORIGINALS["connect_ex"]  # type: ignore[method-assign]
    socket.socket.sendto = _ORIGINALS["sendto"]  # type: ignore[method-assign]
    socket.getaddrinfo = _ORIGINALS["getaddrinfo"]
    socket.gethostbyname = _ORIGINALS["gethostbyname"]
    socket.gethostbyname_ex = _ORIGINALS["gethostbyname_ex"]
    _ORIGINALS.clear()


def _blocked(destination: object) -> NetworkBlockedError:
    return NetworkBlockedError(
        f"VisionForge offline mode blocked a connection to {destination!r}"
    )


def _is_loopback_host(host: object) -> bool:
    """A loopback address, or a name that never needs a lookup to reach one."""
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    if host.lower() in _LOCAL_NAMES:
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _check_destination(sock: socket.socket, address: Any) -> None:
    # AF_UNIX and other local families carry a path, not a network host.
    if sock.family not in (socket.AF_INET, socket.AF_INET6):
        return
    host = address[0] if isinstance(address, tuple) and address else address
    if not _is_loopback_host(host):
        raise _blocked(host)


def _check_lookup(host: object) -> None:
    """A name lookup may go to a DNS server: only literals and local names pass."""
    if host is None or _is_loopback_host(host):
        return
    text = host.decode("ascii", "replace") if isinstance(host, bytes) else str(host)
    try:
        ipaddress.ip_address(text.split("%", 1)[0])
    except ValueError:
        # This machine's own name resolves locally; servers ask for it on bind.
        if text.lower() != socket.gethostname().lower():
            raise _blocked(text) from None


def _guarded_connect(self: socket.socket, address: Any) -> None:
    _check_destination(self, address)
    _ORIGINALS["connect"](self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> int:
    _check_destination(self, address)
    result: int = _ORIGINALS["connect_ex"](self, address)
    return result


def _guarded_sendto(self: socket.socket, data: Any, *args: Any) -> int:
    _check_destination(self, args[-1])
    result: int = _ORIGINALS["sendto"](self, data, *args)
    return result


def _guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
    _check_lookup(host)
    return _ORIGINALS["getaddrinfo"](host, *args, **kwargs)


def _guarded_gethostbyname(hostname: str) -> str:
    _check_lookup(hostname)
    result: str = _ORIGINALS["gethostbyname"](hostname)
    return result


def _guarded_gethostbyname_ex(hostname: str) -> tuple[str, list[str], list[str]]:
    _check_lookup(hostname)
    result: tuple[str, list[str], list[str]] = _ORIGINALS["gethostbyname_ex"](hostname)
    return result
