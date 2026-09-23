"""Environment diagnostics: detect GPU/driver and recommend the correct torch wheel."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
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
# 12.4..12.5 → cu124; 12.6..12.7 → cu126; 12.8+ → cu128.
#
# cu128 matters beyond "newer is nicer": RTX 50-series (Blackwell) is compute
# capability 12.0, and no wheel before cu128 ships an sm_120 kernel. On those
# cards an earlier build imports fine and reports the GPU, then fails at the
# first kernel launch — the exact silent-misconfiguration this module exists
# to prevent.
_WHEEL_THRESHOLDS: list[tuple[int, str]] = [
    (118, "cu118"),
    (121, "cu121"),
    (124, "cu124"),
    (126, "cu126"),
    (128, "cu128"),
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

    # Driver 6xx renamed the header field to "CUDA UMD Version" (and added a
    # separate KMD one). Matching only the old spelling made every recent driver
    # look like no GPU at all, which recommended the CPU wheel on a CUDA machine.
    match = re.search(r"CUDA(?:\s+UMD)?\s+Version:\s*(\d+\.\d+)", result.stdout)
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


def _cuda_int(version: str | None) -> int | None:
    """``"12.8"`` → ``128``, the same scale as `_WHEEL_THRESHOLDS`; None if unreadable."""
    if not version:
        return None
    try:
        major, minor = version.split(".")[:2]
        return int(major) * 10 + int(minor)
    except ValueError:
        return None


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


def missing_package_hint(package: str) -> str:
    """What to run when a bundled dependency turns out to be missing.

    Since ADR-106 these ship with the package, so an ImportError means the
    install is damaged or partial — not that an extra was skipped. Telling
    someone to "add the optional extra" would send them looking for a flag that
    no longer exists.

    An editable checkout reinstalls from the source tree, a wheel install from
    the distribution name — `pip install -e .` only works inside a checkout.
    """
    if installed_from_source():
        return f'pip install -e "." --force-reinstall  (missing: {package})'
    return f'pip install --force-reinstall "{_DIST_NAME}"  (missing: {package})'


def build_install_command(tag: str) -> tuple[list[str], str]:
    """Return the commands that put the ``tag`` build of torch in place, and its index.

    Only torch and torchvision — the package itself is already installed, that
    is how doctor is running. And two steps, because since ADR-106 there is
    always a torch here already: `ultralytics` pulls one from PyPI on the first
    install, and on Windows PyPI only carries CPU builds. Asking pip to install
    `visionforge-studio[cu128]` against the CUDA index then changed nothing —
    the CPU torch already satisfies `torch>=2.3`, so pip printed "Requirement
    already satisfied" and stopped (ADR-107).

    `--upgrade` does not rescue it either: the CUDA index trails PyPI (2.11+cu128
    against 2.14+cpu when this was written), so the CUDA build looks *older* and
    pip keeps what is installed. Removing the CPU build first is what leaves the
    CUDA index as the only candidate.
    """
    url = f"{_INDEX_BASE}/{tag}"
    return (
        [
            "pip uninstall -y torch torchvision",
            f"pip install torch torchvision --index-url {url}",
        ],
        url,
    )


def torch_swap_steps(tag: str) -> list[list[str]]:
    """The argv lists `--fix` runs, aimed at *this* interpreter's environment.

    `sys.executable -m pip`, not whatever `pip` is first on PATH: with the venv
    not activated, that is another Python and the install lands where nothing
    will import it. An environment made by `uv venv` has no pip at all, so there
    the same two steps go through `uv pip --python`.
    """
    url = f"{_INDEX_BASE}/{tag}"
    uv = shutil.which("uv")
    if importlib.util.find_spec("pip") is None and uv:
        target = ["--python", sys.executable]
        return [
            [uv, "pip", "uninstall", "torch", "torchvision", *target],
            [uv, "pip", "install", "torch", "torchvision", "--index-url", url, *target],
        ]
    pip = [sys.executable, "-m", "pip"]
    return [
        [*pip, "uninstall", "-y", "torch", "torchvision"],
        [*pip, "install", "torch", "torchvision", "--index-url", url],
    ]


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


def _run_install(tag: str) -> int:
    """Replace this environment's torch with the ``tag`` build; returns the exit code.

    The uninstall's exit code is not the answer — removing a torch that is not
    there is fine. The install's is.
    """
    uninstall, install = torch_swap_steps(tag)
    print("  $ " + " ".join(uninstall))
    subprocess.run(uninstall, check=False)
    print("  $ " + " ".join(install))
    return subprocess.run(install, check=False).returncode


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
    tag = select_wheel_tag(driver_cuda)
    # "Runs on GPU" alone is not enough to keep an install: an older CUDA build on
    # an RTX 50 imports, reports the GPU, and fails at the first kernel. What
    # settles it is the build — as new as the one this driver calls for, keep it.
    # This matters more since ADR-107, because the fix now *uninstalls* torch;
    # printing that to someone whose setup is already right would be sabotage.
    installed_build = _cuda_int(torch_info["cuda_build"])
    wanted_build = _cuda_int(tag[2:4] + "." + tag[4:]) if tag.startswith("cu") else None
    already_working = torch_says_cuda and (
        not driver_cuda
        or (
            installed_build is not None
            and wanted_build is not None
            and installed_build >= wanted_build
        )
    )
    commands, _index_url = build_install_command(tag)
    steps = "\n".join(f"         {line}" for line in commands)
    print()
    if already_working:
        print("  Recommended wheel: keep the current install")
        print(f"  Reason           : torch {torch_info['version']} already runs on GPU")
    else:
        print(f"  Recommended wheel: {tag}")
        print("  Install commands :")
        for line in commands:
            print(f"    {line}")
    print()

    # --- torch probe ---
    if not torch_info["importable"]:
        # Not an INFO: torch is what trains. Reporting "environment looks good"
        # while the one required dependency is missing sends a new user off to
        # discover it on their first run instead of here.
        print(
            "[FAIL] torch is not installed — nothing can train yet.\n"
            "       Install the build for this machine (or run "
            "`visionforge doctor --fix`):\n"
            f"{steps}"
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
                "       You have a CPU-only torch on a CUDA machine — training\n"
                "       would run, slowly, on the CPU. Reinstall torch from the\n"
                "       CUDA index to fix this mismatch (`visionforge doctor "
                "--fix` does both):\n"
                f"{steps}"
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
        prompt = (
            f"\nReplace torch with the {tag} build now? This runs:\n"
            + "\n".join(f"  {line}" for line in commands)
            + "\n[y/N] "
        )
        if confirm_fn(prompt):
            _run_install(tag)

    return exit_code
