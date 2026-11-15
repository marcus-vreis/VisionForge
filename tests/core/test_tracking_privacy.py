"""Which config keys ``public_parameters`` leaves out (ADR-170, ADR-172)."""

from __future__ import annotations

import pytest

from visionforge.core.tracking_privacy import public_parameters


@pytest.mark.parametrize(
    "key",
    [
        "api_token",
        "Password",
        "client_secret",
        "credentials",
        "weights_path",
        "tracking_uri",
        "base_url",
        "dir",
        "dirs",
        "base_dir",
        "cache_dirs",
    ],
)
def test_private_keys_are_left_out(key: str) -> None:
    assert public_parameters({key: "x"}) == {}


@pytest.mark.parametrize("key", ["epochs", "direction", "dirty", "dir_mode", "lr"])
def test_other_keys_are_kept(key: str) -> None:
    assert public_parameters({key: 1}) == {key: 1}


def test_a_private_key_hides_its_whole_subtree() -> None:
    config = {"data": {"base_dir": "C:/x", "split": 0.2}, "output_dir": {"a": 1}}
    assert public_parameters(config) == {"data.split": 0.2}
