"""A credential store is only worth having if it never leaks and never lies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionforge.utils.credentials import (
    PROVIDERS,
    credential_status,
    forget_credential,
    load_credential,
    mask,
    resolve_credential,
    save_credential,
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never touch the developer's real ~/.visionforge while testing."""
    monkeypatch.setenv("VISIONFORGE_HOME", str(tmp_path / "home"))


class TestMasking:
    def test_keeps_the_last_four_so_you_can_tell_which_key_it_is(self) -> None:
        assert mask("rf_ABCDEFGH1234").endswith("1234")

    def test_hides_everything_else(self) -> None:
        masked = mask("rf_ABCDEFGH1234")
        assert "ABCDEFGH" not in masked
        assert masked.startswith("•")

    def test_a_short_key_reveals_nothing(self) -> None:
        assert mask("abcd") == "••••"

    def test_empty_stays_empty(self) -> None:
        assert mask("") == ""


class TestRoundTrip:
    def test_save_then_load(self) -> None:
        save_credential("roboflow", "rf_secret_value")
        assert load_credential("roboflow") == "rf_secret_value"

    def test_saving_again_replaces(self) -> None:
        save_credential("roboflow", "first")
        save_credential("roboflow", "second")
        assert load_credential("roboflow") == "second"

    def test_whitespace_is_trimmed(self) -> None:
        save_credential("kaggle", "  user:key  ")
        assert load_credential("kaggle") == "user:key"

    def test_forget_removes_it(self) -> None:
        save_credential("roboflow", "x")
        assert forget_credential("roboflow") is True
        assert load_credential("roboflow") is None

    def test_forgetting_what_was_never_saved_is_not_an_error(self) -> None:
        assert forget_credential("roboflow") is False

    def test_providers_do_not_collide(self) -> None:
        save_credential("roboflow", "rf")
        save_credential("kaggle", "kg")
        assert load_credential("roboflow") == "rf"
        assert load_credential("kaggle") == "kg"


class TestValidation:
    def test_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            save_credential("dropbox", "x")

    def test_blank_value_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            save_credential("roboflow", "   ")


class TestStatus:
    def test_reports_every_provider_even_when_empty(self) -> None:
        status = credential_status()
        assert set(status) == set(PROVIDERS)
        assert all(not v["saved"] for v in status.values())

    def test_status_never_carries_the_real_value(self) -> None:
        """The status is what reaches the browser — a screenshot of the panel
        must not be a leak."""
        save_credential("roboflow", "rf_SUPERSECRET99")
        blob = json.dumps(credential_status())
        assert "SUPERSECRET" not in blob
        assert "rf_SUPERSECRET99" not in blob


class TestResolve:
    def test_explicit_value_wins_over_stored(self) -> None:
        """A one-off key must not overwrite what is saved."""
        save_credential("roboflow", "stored")
        assert resolve_credential("roboflow", "one_off") == "one_off"
        assert load_credential("roboflow") == "stored"

    def test_falls_back_to_stored_when_blank(self) -> None:
        save_credential("roboflow", "stored")
        assert resolve_credential("roboflow", "") == "stored"
        assert resolve_credential("roboflow", None) == "stored"

    def test_none_when_nothing_is_available(self) -> None:
        assert resolve_credential("roboflow", None) is None


class TestResilience:
    def test_a_corrupt_store_reads_as_empty_instead_of_crashing(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "home" / "credentials.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        assert load_credential("roboflow") is None
        # And saving over it recovers.
        save_credential("roboflow", "fresh")
        assert load_credential("roboflow") == "fresh"
