"""How many DataLoader workers this machine can actually afford.

The obvious heuristic — scale with the model or the GPU — measures the wrong
thing. A worker never holds the model: it loads and transforms images while the
model lives in the main process, on the GPU. What a worker costs is *commit*:
on Windows, `spawn` starts a fresh interpreter that re-imports torch and its
CUDA DLLs, on the order of a gigabyte of memory each, and a run keeps one pool
per loader (train and val, sometimes test). VRAM and parameter count do not
enter into it.

That is the failure this exists to prevent: WinError 1455, "the paging file is
too small", which is Windows refusing to commit memory that was never there
(ADR-081). ADR-081 declined to cap workers because the per-worker cost was a
guess. It no longer has to be — the commit budget is readable, so the cap is
arithmetic rather than superstition (ADR-098).
"""

from __future__ import annotations

import ctypes
import os
import sys

# Measured on Windows with a CUDA build: a spawned worker re-importing torch
# lands near 1 GB of commit. Deliberately not tuned finer — the number is a
# budget, and being wrong by 20% here costs one worker either way.
_COMMIT_PER_WORKER_BYTES = 1024**3

# Even with memory to spare, more workers than this stop paying for themselves
# on image pipelines and start costing startup time.
_MAX_WORKERS = 8


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_commit_bytes() -> int | None:
    """Memory this process could still commit, or None when unknowable.

    Windows answers exactly (``ullAvailPageFile`` is RAM plus page file minus
    what is already committed), which is the platform the crash belongs to.
    Elsewhere this returns None and the caller falls back to the CPU count —
    Linux overcommits, so the same arithmetic would not mean the same thing.
    """
    if sys.platform != "win32":
        return None
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullAvailPageFile)


def suggested_workers(*, loader_pools: int = 2) -> int:
    """Workers per loader this machine can afford, given how many pools run.

    Args:
        loader_pools: how many DataLoaders keep workers alive at once. A
            classification run holds train and val, so two.

    Returns:
        A count from 0 upward, capped by the CPU count and ``_MAX_WORKERS``.
        Zero means "load in the main process", which is always safe.
    """
    cpu_cap = min(os.cpu_count() or 1, _MAX_WORKERS)
    commit = available_commit_bytes()
    if commit is None:
        return cpu_cap
    pools = max(loader_pools, 1)
    # Half the free commit, not all of it: the training process itself grows,
    # and a cap that consumes the entire budget only moves the crash later.
    affordable = int(commit * 0.5) // (_COMMIT_PER_WORKER_BYTES * pools)
    return max(0, min(cpu_cap, affordable))


__all__ = ["available_commit_bytes", "suggested_workers"]
