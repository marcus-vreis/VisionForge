"""Cancelling keeps what the run earned; that is what makes it usable."""

from __future__ import annotations

import threading

from visionforge.core.cancellation import CancellationToken, is_cancelled


class TestCancellationToken:
    def test_starts_uncancelled(self) -> None:
        assert CancellationToken().cancelled is False

    def test_cancel_is_one_way(self) -> None:
        token = CancellationToken()

        token.cancel()
        token.cancel()

        assert token.cancelled is True

    def test_truthiness_reads_as_was_it_cancelled(self) -> None:
        """`if token:` must not mean "does a token exist" — that inverts the check."""
        token = CancellationToken()

        assert not token
        token.cancel()
        assert token

    def test_is_visible_across_threads(self) -> None:
        """The GUI sets it on the request thread; the trainer reads it on another."""
        token = CancellationToken()
        seen = threading.Event()

        def watcher() -> None:
            while not token.cancelled:
                pass
            seen.set()

        t = threading.Thread(target=watcher, daemon=True)
        t.start()
        token.cancel()

        assert seen.wait(timeout=5.0) is True


class TestIsCancelled:
    def test_absent_token_is_never_cancelled(self) -> None:
        """The CLI has nobody to press a button and passes None."""
        assert is_cancelled(None) is False

    def test_reads_a_present_token(self) -> None:
        token = CancellationToken()

        assert is_cancelled(token) is False
        token.cancel()
        assert is_cancelled(token) is True
