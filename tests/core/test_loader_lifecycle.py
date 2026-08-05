"""Worker pools must not outlive the run that created them."""

from __future__ import annotations

from typing import Any

import pytest

from visionforge.core.loader_lifecycle import (
    LoaderCache,
    describe_worker_spawn_failure,
    shutdown_loader,
    shutdown_loaders,
)


class FakeIterator:
    """Stands in for a DataLoader's _MultiProcessingDataLoaderIter."""

    def __init__(self, raises: bool = False) -> None:
        self.shutdown_calls = 0
        self._raises = raises

    def _shutdown_workers(self) -> None:
        self.shutdown_calls += 1
        if self._raises:
            raise RuntimeError("worker refused to die")


class FakeLoader:
    def __init__(self, iterator: Any = None) -> None:
        self._iterator = iterator


class TestShutdownLoader:
    def test_stops_the_worker_pool(self) -> None:
        it = FakeIterator()
        loader = FakeLoader(it)

        shutdown_loader(loader)

        assert it.shutdown_calls == 1
        assert loader._iterator is None

    def test_loader_that_was_never_iterated_is_a_no_op(self) -> None:
        loader = FakeLoader(None)
        shutdown_loader(loader)  # must not raise
        assert loader._iterator is None

    def test_second_call_is_harmless(self) -> None:
        it = FakeIterator()
        loader = FakeLoader(it)

        shutdown_loader(loader)
        shutdown_loader(loader)

        assert it.shutdown_calls == 1

    def test_a_failing_shutdown_does_not_mask_the_real_error(self) -> None:
        """Teardown runs in a `finally`; raising there would hide the run's error."""
        it = FakeIterator(raises=True)
        loader = FakeLoader(it)

        shutdown_loader(loader)  # must swallow

        assert loader._iterator is None

    def test_shutdown_loaders_skips_none(self) -> None:
        it = FakeIterator()
        loader = FakeLoader(it)

        shutdown_loaders(loader, None)

        assert it.shutdown_calls == 1


class TestLoaderCache:
    def test_builds_once_and_reuses(self) -> None:
        """A second pool for the same split is a second set of worker processes."""
        calls = []

        def factory() -> FakeLoader:
            calls.append(1)
            return FakeLoader(FakeIterator())

        cache = LoaderCache()
        first = cache.cached("train", factory)
        second = cache.cached("train", factory)

        assert first is second
        assert len(calls) == 1

    def test_close_stops_every_split(self) -> None:
        cache = LoaderCache()
        iters = {}
        for split in ("train", "val", "test"):
            it = FakeIterator()
            iters[split] = it
            cache.cached(split, lambda it=it: FakeLoader(it))  # type: ignore[misc]

        cache.close()

        assert all(it.shutdown_calls == 1 for it in iters.values())

    def test_close_is_idempotent(self) -> None:
        it = FakeIterator()
        cache = LoaderCache()
        cache.cached("train", lambda: FakeLoader(it))

        cache.close()
        cache.close()

        assert it.shutdown_calls == 1


class TestBlockClosesOnFailure:
    """The leak that started this: a raising run left its workers behind."""

    def test_finally_closes_even_when_the_run_raises(self) -> None:
        cache = LoaderCache()
        it = FakeIterator()
        cache.cached("train", lambda: FakeLoader(it))

        with pytest.raises(RuntimeError, match="training blew up"):
            try:
                raise RuntimeError("training blew up")
            finally:
                cache.close()

        assert it.shutdown_calls == 1


class TestWorkerSpawnFailure:
    """WinError 1455 names a torch DLL, which sends the reader after the wrong thing."""

    def _oserror(self, winerror: int) -> OSError:
        exc = OSError("Error loading curand64_10.dll or one of its dependencies")
        exc.winerror = winerror  # type: ignore[attr-defined]
        return exc

    def test_paging_file_error_is_explained(self) -> None:
        msg = describe_worker_spawn_failure(self._oserror(1455))

        assert msg is not None
        assert "num_workers" in msg
        assert "paginação" in msg

    def test_other_oserrors_pass_through(self) -> None:
        assert describe_worker_spawn_failure(self._oserror(2)) is None

    def test_non_windows_errors_pass_through(self) -> None:
        assert describe_worker_spawn_failure(ValueError("nada a ver")) is None
