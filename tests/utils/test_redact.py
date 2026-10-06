"""redact_secrets: a provider's error message must be safe to show and to log.

The Roboflow client puts the API key in the URL it posts to, so a network
failure raises an exception whose text contains `?api_key=<KEY>`. These tests
pin what the helper masks, and, as much, what it leaves alone: an error message
that has been mangled into nothing is not an improvement.
"""

from __future__ import annotations

from urllib.parse import quote

import pytest

from visionforge.utils.redact import MASK, redact_secrets

SECRET = "SECRET123"


class TestPatterns:
    def test_the_roboflow_connection_error_loses_its_key(self) -> None:
        text = (
            "HTTPSConnectionPool(host='api.roboflow.com', port=443): Max retries "
            f"exceeded with url: /?api_key={SECRET} (Caused by NameResolutionError)"
        )

        out = redact_secrets(text)

        assert SECRET not in out
        # The rest is what makes the message useful, so it must survive.
        assert "api.roboflow.com" in out
        assert "Max retries exceeded with url: /?api_key=" in out
        assert "(Caused by NameResolutionError)" in out

    @pytest.mark.parametrize(
        "name", ["api_key", "API_KEY", "apikey", "api-key", "token", "access_token"]
    )
    def test_query_parameters_are_masked(self, name: str) -> None:
        out = redact_secrets(f"GET https://x.test/v1?a=1&{name}={SECRET}&b=2 failed")

        assert SECRET not in out
        assert f"{name}={MASK}" in out
        # Neighbouring parameters are not part of the secret.
        assert "a=1&" in out
        assert "&b=2 failed" in out

    def test_a_json_style_pair_is_masked(self) -> None:
        out = redact_secrets(f'payload {{"api_key": "{SECRET}", "version": 3}}')

        assert SECRET not in out
        assert '"version": 3' in out

    def test_an_authorization_header_is_masked(self) -> None:
        out = redact_secrets(f"Authorization: Bearer {SECRET}")

        assert SECRET not in out
        assert out.lower().startswith("authorization")

    def test_a_header_in_a_dict_repr_is_masked(self) -> None:
        out = redact_secrets(
            f"headers={{'Authorization': 'Bearer {SECRET}', 'X': 'y'}}"
        )

        assert SECRET not in out
        assert "'X': 'y'" in out

    def test_a_basic_authorization_value_is_masked(self) -> None:
        assert "dXNlcjpwYXNz" not in redact_secrets("Authorization: Basic dXNlcjpwYXNz")

    def test_a_bare_bearer_token_is_masked(self) -> None:
        out = redact_secrets(f"rejected: Bearer {SECRET} is not valid")

        assert SECRET not in out
        assert "is not valid" in out

    def test_a_password_in_a_url_is_masked(self) -> None:
        out = redact_secrets(f"proxy http://user:{SECRET}@10.0.0.1:3128 refused")

        assert SECRET not in out
        assert "10.0.0.1:3128 refused" in out

    def test_a_multiline_traceback_is_masked_line_by_line(self) -> None:
        text = (
            "Traceback (most recent call last):\n"
            f'  File "x.py", line 1, in <module>\n    url = "/?api_key={SECRET}"\n'
            f"requests.exceptions.ConnectionError: /?token={SECRET}\n"
        )

        out = redact_secrets(text)

        assert SECRET not in out
        assert out.count("\n") == text.count("\n")


class TestKnownSecrets:
    def test_the_literal_value_is_masked_wherever_it_appears(self) -> None:
        out = redact_secrets(f"the server said {SECRET!r} is invalid", SECRET)

        assert SECRET not in out
        assert "is invalid" in out

    def test_every_occurrence_is_masked(self) -> None:
        assert SECRET not in redact_secrets(f"{SECRET} and {SECRET} again", SECRET)

    def test_several_secrets(self) -> None:
        out = redact_secrets(
            "a=ROBOFLOW-KEY b=KAGGLE-TOKEN", "ROBOFLOW-KEY", "KAGGLE-TOKEN"
        )

        assert "ROBOFLOW-KEY" not in out
        assert "KAGGLE-TOKEN" not in out

    def test_the_url_encoded_form_is_masked_too(self) -> None:
        secret = "abc/def+ghi==jkl"
        encoded = quote(secret, safe="")
        assert encoded != secret

        out = redact_secrets(f"GET /download/{encoded} failed", secret)

        assert encoded not in out

    def test_a_secret_that_contains_another_is_masked_whole(self) -> None:
        out = redact_secrets("value=KEY-12345-LONG", "KEY-12345", "KEY-12345-LONG")

        assert "LONG" not in out

    def test_none_and_blank_values_are_ignored(self) -> None:
        text = "nothing to hide here"

        assert redact_secrets(text, None, "", "   ") == text  # type: ignore[arg-type]

    def test_a_value_too_short_to_be_a_key_does_not_mangle_the_message(self) -> None:
        # A one-character "secret" would otherwise turn every letter into a mask.
        assert redact_secrets("a plain message", "a") == "a plain message"

    def test_surrounding_whitespace_in_the_secret_is_not_part_of_it(self) -> None:
        assert SECRET not in redact_secrets(f"bad key {SECRET}", f"  {SECRET}\n")


class TestWhatIsLeftAlone:
    @pytest.mark.parametrize(
        "text",
        [
            "ConnectionError: [Errno 11001] getaddrinfo failed",
            "Kaggle token: expired, create a new one",
            "max_tokens=5 and the tokenizer is fine",
            "HTTP 401: Unauthorized for url: https://huggingface.co/api/datasets/a/b",
            "the bearer of bad news",
            "",
        ],
    )
    def test_text_without_a_secret_is_unchanged(self, text: str) -> None:
        assert redact_secrets(text) == text

    def test_redacting_twice_changes_nothing_more(self) -> None:
        once = redact_secrets(f"/?api_key={SECRET}&token={SECRET}", SECRET)

        assert redact_secrets(once, SECRET) == once
