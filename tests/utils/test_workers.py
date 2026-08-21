"""A worker budget measured in memory, because that is what a worker costs.

ADR-081 declined to cap `num_workers` because the per-worker cost was a guess.
It is not a guess on Windows: the commit budget is readable, and a spawned
worker re-importing torch's CUDA DLLs lands near a gigabyte of it. Scaling with
the model or the GPU would measure the wrong thing entirely — the worker never
holds the model (ADR-098).
"""

from __future__ import annotations

import pytest

from visionforge.utils import workers as w


class TestBudget:
    def test_more_pools_mean_fewer_workers_each(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(w, "available_commit_bytes", lambda: 16 * 1024**3)
        monkeypatch.setattr(w.os, "cpu_count", lambda: 32)

        # Half of 16 GB is 8 GB; at 1 GB a worker that is 8, 4 and 2 per pool.
        assert w.suggested_workers(loader_pools=1) == 8
        assert w.suggested_workers(loader_pools=2) == 4
        assert w.suggested_workers(loader_pools=3) == 2

    def test_a_starved_machine_gets_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Zero is a valid answer: the main process always loads its own data."""
        monkeypatch.setattr(w, "available_commit_bytes", lambda: 1 * 1024**3)
        monkeypatch.setattr(w.os, "cpu_count", lambda: 32)

        assert w.suggested_workers(loader_pools=3) == 0

    def test_the_cpu_count_still_caps_a_roomy_machine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(w, "available_commit_bytes", lambda: 512 * 1024**3)
        monkeypatch.setattr(w.os, "cpu_count", lambda: 2)

        assert w.suggested_workers(loader_pools=1) == 2

    def test_the_ceiling_holds_even_with_memory_to_burn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(w, "available_commit_bytes", lambda: 512 * 1024**3)
        monkeypatch.setattr(w.os, "cpu_count", lambda: 128)

        assert w.suggested_workers(loader_pools=1) == w._MAX_WORKERS

    def test_without_a_readable_budget_it_falls_back_to_cpus(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Linux overcommits, so the same arithmetic would not mean the same."""
        monkeypatch.setattr(w, "available_commit_bytes", lambda: None)
        monkeypatch.setattr(w.os, "cpu_count", lambda: 4)

        assert w.suggested_workers(loader_pools=3) == 4

    def test_the_reported_crash_would_have_been_capped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """10.7 GB free, three pools, 8 requested — the WinError 1455 run."""
        monkeypatch.setattr(w, "available_commit_bytes", lambda: int(10.7 * 1024**3))
        monkeypatch.setattr(w.os, "cpu_count", lambda: 12)

        assert w.suggested_workers(loader_pools=3) == 1
