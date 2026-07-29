"""Environment diagnostics: detect GPU/driver and recommend the correct torch wheel."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from collections.abc import Callable
from importlib.metadata import distribution
from pathlib import Path
from typing import TypedDict

# The PyPI distribution name, which is NOT the import name: plain `visionforge`
# on PyPI belongs to an unrelated project.
_DIST_NAME = "visionforge-studio"

# Supported wheel tags in ascending CUDA capability order.
# Each entry is (min_driver_major*10 + min_driver_minor, tag).
# Rule: pick the highest tag whose threshold is <= driver CUDA (in major*10+minor space).
# driver < 11.8 or None → cpu; 11.8..12.0 → cu118; 12.1..12.3 → cu121;
# 12.4..12.5 → cu124; 12.6+ → cu126.
_WHEEL_THRESHOLDS: list[tuple[int, str]] = [
    (118, "cu118"),
    (121, "cu121"),
    (124, "cu124"),
    (126, "cu126"),
]

_INDEX_BASE = "https://download.pytorch.org/whl"


class TorchProbe(TypedDict):
    """Result of probing whether torch is importable and CUDA-enabled."""

    importable: bool
    version: str
    cuda_available: bool
    # The CUDA the installed wheel was *built* against (torch.version.cuda),
    # which is what names the wheel — not the driver's CUDA.
    cuda_build: str | None


class PythonCheck(TypedDict):
    """Result of checking whether the running Python meets the project minimum."""

    version: str
    ok: bool


def detect_driver_cuda() -> str | None:
    """Run nvidia-smi and return the driver CUDA version string, or None on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        return None

    match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
    if not match:
        return None

    return match.group(1)


def select_wheel_tag(driver_cuda: str | None) -> str:
    """Map a driver CUDA version string (e.g. '12.4') to the nearest wheel tag.

    Returns 'cpu' when driver_cuda is None or below the minimum supported version.
    """
    if driver_cuda is None:
        return "cpu"

    try:
        major, minor = driver_cuda.split(".", 1)
        driver_int = int(major) * 10 + int(minor)
    except (ValueError, AttributeError):
        return "cpu"

    # Walk thresholds in reverse; return the first tag whose threshold ≤ driver.
    selected = "cpu"
    for threshold, tag in _WHEEL_THRESHOLDS:
        if driver_int >= threshold:
            selected = tag
    return selected


def installed_from_source() -> bool:
    """True when this install points at a checkout (``pip install -e .``).

    Editable installs record a ``direct_url.json`` with ``dir_info.editable``.
    Anything else — a wheel from PyPI — has no source tree to install from, so
    telling that user to run ``pip install -e ".[cpu]"`` would fail.
    """
    try:
        raw = distribution(_DIST_NAME).read_text("direct_url.json")
    except Exception:  # noqa: BLE001 - absent metadata just means "not editable"
        return False
    if not raw:
        return False
    try:
        info = json.loads(raw)
    except ValueError:
        return False
    return bool(info.get("dir_info", {}).get("editable"))


def extra_install_hint(extra: str) -> str:
    """The command that installs an optional extra *for this install*.

    Same split as `build_install_command`: `pip install -e ".[x]"` only works
    inside a checkout, so a pip-installed user needs the distribution name.
    Optional features raise ImportError with this string, and a hint that
    cannot be pasted is no hint at all.
    """
    if installed_from_source():
        return f'pip install -e ".[{extra}]"'
    return f'pip install "{_DIST_NAME}[{extra}]"'


def build_install_command(tag: str) -> tuple[str, str]:
    """Return the pip install command and index URL for the given wheel tag."""
    if installed_from_source():
        cmd = f'pip install -e ".[{tag}]"'
    else:
        cmd = f'pip install "{_DIST_NAME}[{tag}]"'
    url = f"{_INDEX_BASE}/{tag}"
    return cmd, url


def probe_torch() -> TorchProbe:
    """Check whether torch is importable and whether CUDA is available.

    Uses importlib to avoid a hard import at module top so CI (no torch) stays clean.
    """
    if importlib.util.find_spec("torch") is None:
        return TorchProbe(
            importable=False,
            version="unknown",
            cuda_available=False,
            cuda_build=None,
        )

    try:
        import torch  # noqa: PLC0415

        return TorchProbe(
            importable=True,
            version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_build=torch.version.cuda,
        )
    except Exception:  # noqa: BLE001
        return TorchProbe(
            importable=False,
            version="unknown",
            cuda_available=False,
            cuda_build=None,
        )


def check_python() -> PythonCheck:
    """Report current Python version and whether it meets the >=3.13 requirement."""
    # Index by position so the function also works when tests patch version_info
    # with a plain tuple (patch.object replaces the namedtuple with a tuple).
    vi = sys.version_info
    major, minor, micro = vi[0], vi[1], vi[2]
    version = f"{major}.{minor}.{micro}"
    ok = (major, minor) >= (3, 13)
    return PythonCheck(version=version, ok=ok)


def _default_confirm(prompt: str) -> bool:
    """Prompt the user for y/N and return True only on an explicit 'y'."""
    answer = input(prompt).strip().lower()
    return answer == "y"


def _run_install(cmd: str, index_url: str) -> int:
    """Shell out to run the pip install command; returns the subprocess exit code."""
    full_cmd = f"{cmd} --index-url {index_url}"
    result = subprocess.run(full_cmd, shell=True, check=False)  # noqa: S602
    return result.returncode


def run_doctor(
    fix: bool = False,
    confirm_fn: Callable[[str], bool] = _default_confirm,
) -> int:
    """Run the full environment diagnostic, print a report, and optionally install.

    Args:
        fix: When True, prompt the user and run the recommended install on confirmation.
        confirm_fn: Callable that receives a prompt string and returns True for 'yes'.
            Injected so tests can control it without shelling out.

    Returns:
        0 if everything looks good, 1 if any issue was detected.
    """
    issues: list[str] = []

    # --- Python version ---
    py = check_python()
    py_marker = "OK" if py["ok"] else "FAIL"
    print(f"[{py_marker}] Python {py['version']}", end="")
    if not py["ok"]:
        print("  (requires >=3.13)")
        issues.append("python-version")
    else:
        print()

    # --- GPU / driver ---
    # Probe torch first: a working CUDA build is stronger evidence of a usable
    # GPU than nvidia-smi, which is simply absent from PATH on plenty of
    # machines. Recommending the CPU wheel to someone whose GPU already works
    # is the worst answer doctor can give, and it used to give it.
    torch_info = probe_torch()
    driver_cuda = detect_driver_cuda()
    torch_says_cuda = torch_info["importable"] and torch_info["cuda_available"]

    if driver_cuda:
        print(f"[OK] CUDA driver version: {driver_cuda}")
    elif torch_says_cuda:
        print(
            "[OK] GPU usable via torch "
            f"(CUDA build {torch_info['cuda_build']}); nvidia-smi not on PATH"
        )
    else:
        print("[INFO] No CUDA-capable GPU detected (nvidia-smi not found or failed)")

    # --- Wheel recommendation (always shown — this is doctor's primary value) ---
    already_working = torch_says_cuda and not driver_cuda
    tag = select_wheel_tag(driver_cuda)
    cmd, index_url = build_install_command(tag)
    print()
    if already_working:
        print("  Recommended wheel: keep the current install")
        print(f"  Reason           : torch {torch_info['version']} already runs on GPU")
    else:
        print(f"  Recommended wheel: {tag}")
        print(f"  Install command  : {cmd}")
        print(f"  Index URL        : --index-url {index_url}")
    print()

    # --- torch probe ---
    if not torch_info["importable"]:
        # Not an INFO: torch is what trains. Reporting "environment looks good"
        # while the one required dependency is missing sends a new user off to
        # discover it on their first run instead of here.
        print(
            "[FAIL] torch is not installed — nothing can train yet.\n"
            f"       Install it with the hardware extra:\n"
            f"         {cmd}"
        )
        issues.append("torch-missing")
    else:
        cuda_ok = torch_info["cuda_available"]
        torch_marker = "OK" if cuda_ok else "WARN"
        print(
            f"[{torch_marker}] torch {torch_info['version']}  cuda_available={cuda_ok}"
        )

        # Flag the silent-failure case: GPU driver present but CPU-only torch wheel.
        # This is exactly the misconfiguration ADR-005/ADR-042 targets.
        if driver_cuda and not cuda_ok:
            print(
                "[FAIL] GPU driver detected but torch.cuda.is_available() is False.\n"
                "       You likely have a CPU-only torch wheel on a CUDA machine.\n"
                "       Reinstall with the GPU extra to fix this mismatch:\n"
                f"         {cmd}\n"
                f"         --index-url {index_url}"
            )
            issues.append("torch-cuda-mismatch")

    # --- Workspace ---
    # Custom models and tasks are picked up from folders *relative to the
    # working directory*, which works but is invisible: a pip-installed user
    # has no repo to look at, and running from a different folder silently
    # loses them. Printing the resolved paths turns that into something you
    # can see and act on.
    print()
    print(f"  Working directory: {Path.cwd()}")
    for label, folder in (
        ("Custom models", "user_models"),
        ("Custom tasks", "user_tasks"),
    ):
        path = Path(folder)
        if path.is_dir():
            n = len(list(path.glob("*.py"))) + len(list(path.glob("*/task.py")))
            print(f"  {label:17}: {path.resolve()}  ({n} found)")
        else:
            print(f"  {label:17}: {path.resolve()}  (create it to add your own)")

    # --- Verdict ---
    print()
    if issues:
        print(f"[FAIL] Verdict: {len(issues)} issue(s) found — see above.")
        exit_code = 1
    else:
        print("[OK]   Verdict: environment looks good.")
        exit_code = 0

    # --- Optional install ---
    if fix:
        prompt = f"\nRun '{cmd} --index-url {index_url}' now? [y/N] "
        if confirm_fn(prompt):
            _run_install(cmd, index_url)

    return exit_code
