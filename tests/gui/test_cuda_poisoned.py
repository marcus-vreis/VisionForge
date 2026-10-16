"""A job is refused, with a reason, once the server's CUDA context is broken.

A device-side assert (a segmentation mask id beyond ``num_classes`` did it in the
2026-10-16 validation) poisons the whole process's CUDA context: every later job
failed within a second, CPU ones included, while the GUI kept accepting them.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from visionforge.gui.api import routes
from visionforge.utils.messages import using_lang


def _submit() -> object:
    async def start() -> None:
        return None

    return routes._submit_job("r1", "label", "classification", "single", start)


def test_a_broken_cuda_context_refuses_new_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes, "cuda_context_broken", lambda: True)
    submitted: list[object] = []
    monkeypatch.setattr(routes._RUN_QUEUE, "submit", submitted.append)

    with using_lang("en"), pytest.raises(HTTPException) as raised:
        _submit()

    assert raised.value.status_code == 503
    assert "reopen visionforge" in str(raised.value.detail).lower()
    assert submitted == []


def test_a_healthy_server_queues_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes, "cuda_context_broken", lambda: False)
    monkeypatch.setattr(routes._RUN_QUEUE, "submit", lambda job: "running")

    assert _submit().status == "running"  # type: ignore[attr-defined]


def test_the_probe_never_starts_cuda_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from visionforge.gui.api.gpu_health import cuda_context_broken

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    def must_not_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("probed a CUDA context nobody had started")

    monkeypatch.setattr(torch, "zeros", must_not_run)
    assert cuda_context_broken() is False


def test_the_probe_reads_a_cuda_error_as_broken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from visionforge.gui.api.gpu_health import cuda_context_broken

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    def poisoned(*args: object, **kwargs: object) -> None:
        raise RuntimeError("CUDA error: device-side assert triggered")

    monkeypatch.setattr(torch, "zeros", poisoned)
    assert cuda_context_broken() is True
