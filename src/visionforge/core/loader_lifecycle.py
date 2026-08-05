"""Deterministic shutdown for DataLoader worker pools.

A `persistent_workers=True` pool stays alive until the DataLoader's internal
iterator is garbage-collected. Inside a long-lived process — `visionforge gui`
serves runs one after another — that is not good enough: when a run raises, the
exception's traceback holds the frames that hold the loaders, so the pool
outlives the run that created it. Every failed attempt then leaves its workers
behind, and on Windows each one has re-imported torch and its CUDA DLLs, so the
leak is measured in gigabytes of commit charge rather than megabytes.

Observed in the wild: two failed classification attempts left 19 worker
processes holding ~22 GB, after which the next spawn died with
``OSError: [WinError 1455] The paging file is too small``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from loguru import logger

_LoaderT = TypeVar("_LoaderT")


def shutdown_loader(loader: Any) -> None:
    """Stop a DataLoader's worker processes now, instead of at collection time.

    Safe to call on a loader with no workers, on one that was never iterated,
    and more than once.
    """
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is None:
        return
    try:
        shutdown()
    except Exception as exc:  # noqa: BLE001 - teardown must never mask the real error
        logger.debug(
            "DataLoader worker shutdown raised {}: {}", type(exc).__name__, exc
        )
    finally:
        loader._iterator = None


def shutdown_loaders(*loaders: Any) -> None:
    """Shut down several loaders, skipping any that are None."""
    for loader in loaders:
        if loader is not None:
            shutdown_loader(loader)


_PAGING_FILE_WINERROR = 1455


def describe_worker_spawn_failure(exc: BaseException) -> str | None:
    """Turn the Windows paging-file error into something a researcher can act on.

    Returns None for anything else, so callers can re-raise untouched.

    Each DataLoader worker is a fresh process that re-imports torch and its CUDA
    DLLs — on Windows the start method is spawn, not fork — so the commit charge
    is roughly a gigabyte per worker, per loader. Windows reports the shortfall
    as WinError 1455, naming whichever DLL happened to be loading, which points
    at torch and hides the real cause.
    """
    winerror = getattr(exc, "winerror", None)
    if winerror != _PAGING_FILE_WINERROR:
        return None
    return (
        "Windows ficou sem espaço de paginação ao criar os processos de leitura "
        "de dados (WinError 1455). Cada worker é um processo novo que recarrega "
        "o torch e as DLLs da CUDA, ~1 GB cada.\n"
        "  - Reduza data.num_workers (2, ou 0 para desligar).\n"
        "  - Feche runs anteriores que tenham ficado presos e o que estiver "
        "ocupando memória.\n"
        "  - Ou aumente o arquivo de paginação do Windows "
        "(Sistema → Configurações avançadas → Desempenho → Memória virtual)."
    )


class LoaderCache:
    """Holds one DataLoader per split so a data module can shut them all down.

    Building a loader per call would give the same split a second worker pool,
    and handing the object out without keeping a reference leaves nothing to
    stop. Every task's data module owns one of these.
    """

    def __init__(self) -> None:
        self._loaders: dict[str, Any] = {}

    def cached(self, split: str, factory: Callable[[], _LoaderT]) -> _LoaderT:
        """Return the split's loader, building it on first request."""
        loader = self._loaders.get(split)
        if loader is None:
            loader = factory()
            self._loaders[split] = loader
        return loader  # type: ignore[no-any-return]

    def close(self) -> None:
        """Stop every worker pool started through this cache."""
        shutdown_loaders(*self._loaders.values())
        self._loaders.clear()
