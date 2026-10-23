"""Shared phase boundaries, failure reports and CPU guard for benchmark runs (ADR-148).

Importing this module must not load torch: the paired CLI calls
``prepare_cpu_process`` before any accelerator library is imported.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any

CONTRACT_VERSION = 2
CONTRACT_SCOPE = "phase_contract_v2"
LEGACY_SCOPE = "legacy_implementation_phases"
PHASES = ("setup", "training", "evaluation", "evidence")
BOUNDARIES = {
    "setup": "Starts after shared untimed preflight and output creation; ends with "
    "loaders and the model holding the initial weights ready.",
    "training": "Starts at seeding or the training call; ends when the best "
    "checkpoint is on disk.",
    "evaluation": "Starts by reloading the best checkpoint; ends with test targets "
    "and predictions in memory.",
    "evidence": "Starts after evaluation; ends when the best-checkpoint hash and "
    "report are assembled. The single report write is untimed.",
}
CACHE_CONDITIONS = {
    "dataset": "local files staged before the run",
    "initial_weights": "local file supplied by the operator",
    "downloads": "none",
    "os_file_cache": "not controlled; earlier phases or runs may warm it",
}
_WINDOWS_PRIORITY = {0x40: "idle", 0x4000: "below_normal", 0x20: "normal"}
_BELOW_NORMAL_PRIORITY_CLASS = 0x4000


class PhaseFailure(RuntimeError):
    """A contract phase failed after its partial report was written."""

    def __init__(self, workflow: str, phase: str, report: dict[str, Any]) -> None:
        super().__init__(f"{workflow} failed during {phase}: {report['error']}")
        self.workflow = workflow
        self.phase = phase
        self.report = report


class PhaseRecorder:
    """Time contract phases in order and write one report, complete or partial."""

    def __init__(
        self,
        workflow: str,
        report_path: Path,
        includes: Mapping[str, Sequence[str]],
        conditions: Mapping[str, Any],
    ) -> None:
        missing = [name for name in PHASES if not includes.get(name)]
        if missing:
            raise ValueError(f"includes must describe every phase; missing {missing}")
        self._path = Path(report_path)
        self._includes = includes
        self.report: dict[str, Any] = {
            "workflow": workflow,
            "status": "running",
            "failed_phase": None,
            "phase_contract": contract_header(),
            "conditions": dict(conditions),
            "phases": {},
        }

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time one phase; on failure write the partial report and raise PhaseFailure."""
        done = len(self.report["phases"])
        if done >= len(PHASES) or PHASES[done] != name:
            expected = PHASES[done] if done < len(PHASES) else "none"
            raise ValueError(f"expected phase {expected!r}, got {name!r}")
        entry: dict[str, Any] = {"status": "running", "seconds": None}
        entry["includes"] = list(self._includes[name])
        self.report["phases"][name] = entry
        start = time.perf_counter()
        try:
            yield
        except Exception as exc:
            entry.update(status="failed", seconds=time.perf_counter() - start)
            self.report.update(
                status="failed",
                failed_phase=name,
                error=f"{type(exc).__name__}: {exc}",
            )
            write_report_once(self._path, self.report)
            raise PhaseFailure(self.report["workflow"], name, self.report) from exc
        entry.update(status="completed", seconds=time.perf_counter() - start)

    def complete(self, results: Mapping[str, Any]) -> dict[str, Any]:
        """Merge results after the last phase and write the completed report."""
        if list(self.report["phases"]) != list(PHASES):
            raise ValueError("cannot complete an incomplete phase sequence")
        self.report.update(results)
        self.report["status"] = "completed"
        write_report_once(self._path, self.report)
        return self.report


def contract_header() -> dict[str, Any]:
    """Return the contract description stored in every contract report."""
    return {
        "version": CONTRACT_VERSION,
        "phases": list(PHASES),
        "boundaries": dict(BOUNDARIES),
        "timing_kind": "machine_wall_clock",
        "human_effort": "not_measured",
    }


def write_report_once(path: Path, report: Mapping[str, Any]) -> None:
    """Create the report file exclusively; an existing file is never replaced."""
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    """Hash a file in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_scope(report: Mapping[str, Any]) -> str:
    """Classify a loaded report without rewriting or reinterpreting its timings."""
    contract = report.get("phase_contract")
    if contract is None:
        return LEGACY_SCOPE
    version = contract.get("version")
    return (
        CONTRACT_SCOPE if version == CONTRACT_VERSION else f"unknown_contract_{version}"
    )


def load_report_scope(path: Path) -> str:
    """Read a report file and return its scope label."""
    return report_scope(json.loads(Path(path).read_text(encoding="utf-8")))


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def capture_conditions(concurrent_workload: str) -> dict[str, Any]:
    """Record cache state, versions, threads, CUDA mask, priority and declared load."""
    import torch

    return {
        "cache": dict(CACHE_CONDITIONS),
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{
                name: _package_version(name)
                for name in ("torch", "torchvision", "numpy", "pillow")
            },
            "visionforge": _package_version("visionforge-studio"),
        },
        "cpu_threads": torch.get_num_threads(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "process_priority": process_priority(),
        "concurrent_workload": concurrent_workload,
    }


def _kernel32() -> Any:
    """Return kernel32 with pointer-sized process handles declared."""
    # sys.platform, which mypy narrows on: on Linux the rest is unreachable and
    # ctypes.WinDLL, which does not exist there, is not type-checked.
    if sys.platform != "win32":
        raise OSError("kernel32 exists only on Windows")
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetPriorityClass.argtypes = [wintypes.HANDLE]
    kernel32.GetPriorityClass.restype = wintypes.DWORD
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.SetPriorityClass.restype = wintypes.BOOL
    return kernel32


def process_priority() -> str:
    """Describe the current process scheduling priority."""
    if sys.platform != "win32":
        return f"nice={os.getpriority(os.PRIO_PROCESS, 0)}"
    kernel32 = _kernel32()
    value = kernel32.GetPriorityClass(kernel32.GetCurrentProcess())
    return _WINDOWS_PRIORITY.get(value, f"class=0x{value:x}")


def prepare_cpu_process() -> None:
    """Hide CUDA, allow one CPU thread and lower priority; call before importing torch."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    if sys.platform != "win32":
        os.nice(10)
        return
    kernel32 = _kernel32()
    if not kernel32.SetPriorityClass(
        kernel32.GetCurrentProcess(), _BELOW_NORMAL_PRIORITY_CLASS
    ):
        raise OSError("SetPriorityClass failed to lower process priority")
