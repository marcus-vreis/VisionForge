"""redact_secrets: a provider's error message must be safe to show and to log.

The Roboflow client puts the API key in the URL it posts to, so a network
failure raises an exception whose text contains `?api_key=<KEY>`. These tests
pin what the helper masks, and, as much, what it leaves alone: an error message
that has been mangled into nothing is not an improvement.
"""

from __future__ import annotations

import time
from urllib.parse import quote

import pytest

from visionforge.utils.redact import (
    _PATTERNS,
    MASK,
    _mask_value,
    redact_secrets,
)

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


class TestSignedLinks:
    """Roboflow's export URL carries its credential as `?key=<token>`."""

    def test_the_roboflow_export_link_loses_its_key(self) -> None:
        out = redact_secrets("Not Found for url: /ds/n9QwXwUK42?key=NnVCe2yMxP")

        assert "NnVCe2yMxP" not in out
        # Where it failed is the useful part.
        assert "/ds/n9QwXwUK42?key=" in out

    @pytest.mark.parametrize(
        "name", ["key", "sig", "signature", "X-Goog-Signature", "X-Amz-Signature"]
    )
    def test_signing_parameters_are_masked_after_a_question_mark_or_ampersand(
        self, name: str
    ) -> None:
        for lead in ("?", "?a=1&"):
            out = redact_secrets(f"GET https://x.test/f{lead}{name}=ABCDEF123456&b=2")

            assert "ABCDEF123456" not in out
            assert f"{name}={MASK}&b=2" in out

    @pytest.mark.parametrize(
        "text",
        [
            "key=value pairs are required",
            "the primary key is missing",
            "set key=1 and sig=2 in the config",
            "dictionary key=name",
            "bad request?a=1 key",
        ],
    )
    def test_prose_that_mentions_key_is_left_alone(self, text: str) -> None:
        assert redact_secrets(text) == text


class TestQuotedAndEncodedForms:
    def test_a_quoted_keyword_argument_is_masked_with_its_quotes_kept(self) -> None:
        # What a pydantic repr or `Roboflow(api_key='…')` looks like.
        for quote_char in ("'", '"'):
            out = redact_secrets(
                f"DatasetDownloadRequest(api_key={quote_char}rf_ABCdef123456{quote_char}, version=1)"
            )

            assert "rf_ABCdef123456" not in out
            assert f"api_key={quote_char}{MASK}{quote_char}, version=1)" in out

    def test_spaces_around_the_equals_sign_do_not_hide_it(self) -> None:
        out = redact_secrets("api_key = 'rf_ABCdef123456'")

        assert "rf_ABCdef123456" not in out

    def test_the_url_encoded_equals_sign_is_masked(self) -> None:
        out = redact_secrets("redirect=/export%3Fapi_key%3Drf_ABCdef123456%26v%3D1")

        assert "rf_ABCdef123456" not in out
        assert "%26v%3D1" in out

    @pytest.mark.parametrize(
        "text",
        ["password: hunter22", "Password:   hunter22", "api_key: hunter22extra"],
    )
    def test_the_colon_form_after_a_credential_name_is_masked(self, text: str) -> None:
        out = redact_secrets(text)

        assert "hunter22" not in out
        assert MASK in out

    def test_a_compound_token_name_works_with_a_colon_too(self) -> None:
        assert "tok12345" not in redact_secrets("access_token: tok12345")

    @pytest.mark.parametrize(
        "text",
        [
            "Kaggle token: expired, create a new one",
            "invalid token: the format is wrong",
            "Error: user not found",
            "Note: keep this somewhere safe",
        ],
    )
    def test_the_colon_form_is_not_applied_to_ordinary_words(self, text: str) -> None:
        assert redact_secrets(text) == text

    def test_every_new_form_is_idempotent(self) -> None:
        text = (
            "api_key='rf_ABCdef123456' /x?key=NnVCe2yMxP&sig=ABCDEF123 "
            "api_key%3Drf_ABCdef123456 password: hunter22"
        )
        once = redact_secrets(text)

        assert redact_secrets(once) == once


class TestTheSchemeWordIsNotTheSecret:
    """`Bearer` is the kind of credential, not the credential.

    A value pattern that took the first word after the separator masked
    "Bearer" and left the key itself in clear.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("x-auth-token: Bearer abcdef123456", "x-auth-token: Bearer ***"),
            ("X-Api-Key: Bearer abcdef123456", "X-Api-Key: Bearer ***"),
            ("access_token: Bearer abcdef123456", "access_token: Bearer ***"),
            ("password: Basic dXNlcjpwYXNz", "password: Basic ***"),
            ("api_key=Bearer abcdef123456", "api_key=Bearer ***"),
            ("token=Bearer abcdef123456", "token=Bearer ***"),
            ("api_key='Bearer abcdef123456'", "api_key='Bearer ***'"),
            ("Authorization=Basic abcdef123456", "Authorization=Basic ***"),
            ("Authorization: Bearer abcdef123456", "Authorization: Bearer ***"),
            ("?key=Bearer abcdef123456", "?key=Bearer ***"),
            ("api_key%3DBearer%20abcdef123456", "api_key%3D***"),
        ],
    )
    def test_the_scheme_stays_and_only_the_credential_is_masked(
        self, text: str, expected: str
    ) -> None:
        out = redact_secrets(text)

        assert "abcdef123456" not in out
        assert "dXNlcjpwYXNz" not in out
        assert out == expected

    def test_a_value_that_is_only_a_scheme_word_is_still_masked(self) -> None:
        assert redact_secrets("api_key=Bearer") == "api_key=***"


class TestOneLineAtATime:
    """A name at the end of a line must not reach for a value on the next one."""

    @pytest.mark.parametrize(
        "text",
        [
            "api_key =\nvalue12345",
            "api_key=\nvalue12345",
            "password:\nvalue12345",
            "password: \n  value12345",
        ],
    )
    def test_a_value_on_the_next_line_is_not_taken(self, text: str) -> None:
        assert redact_secrets(text) == text

    def test_a_tab_before_the_value_is_still_the_same_line(self) -> None:
        assert "value12345" not in redact_secrets("api_key =\tvalue12345")


class TestEscapedQuotes:
    """A message that contains JSON or a dict repr carries its quotes escaped."""

    def test_an_escaped_quoted_value_is_masked_inside_its_escaped_quotes(self) -> None:
        out = redact_secrets('api_key=\\"rf_abc123\\"')

        assert out == 'api_key=\\"***\\"'

    def test_the_colon_form_with_escaped_quotes(self) -> None:
        assert redact_secrets('password: \\"hunter22\\"') == 'password: \\"***\\"'

    def test_a_json_pair_inside_a_string(self) -> None:
        out = redact_secrets('body={\\"api_key\\": \\"rf_abc123\\", \\"v\\": 1}')

        assert "rf_abc123" not in out
        assert '\\"v\\": 1' in out

    def test_an_unterminated_escaped_quote_still_masks_the_value(self) -> None:
        assert "rf_abc123" not in redact_secrets('api_key=\\"rf_abc123')

    def test_a_backslash_ends_an_unquoted_value(self) -> None:
        # `\n` in a repr is two characters; the text after it is not the secret.
        out = redact_secrets("?api_key=rf_abc123\\nnext line")

        assert out == "?api_key=***\\nnext line"

    @pytest.mark.parametrize("letter", ["n", "r", "t"])
    def test_an_escape_letter_ends_the_value(self, letter: str) -> None:
        out = redact_secrets(f"api_key=abc12345\\{letter}next")

        assert out == f"api_key=***\\{letter}next"

    def test_a_backslash_that_escapes_a_quote_ends_the_value(self) -> None:
        # The backslash belongs to the closing `\"`, not to the secret.
        assert redact_secrets('x=\\"api_key=abc12345\\"') == 'x=\\"api_key=***\\"'


class TestABackslashInsideAValueIsPartOfIt:
    """Only an escape (`\\n`, `\\"`…) ends a value; any other backslash is the secret's."""

    def test_a_backslash_in_the_middle_does_not_leak_the_tail(self) -> None:
        assert redact_secrets("password=pa\\ss12345") == "password=***"

    def test_a_value_may_start_with_a_backslash(self) -> None:
        assert redact_secrets("api_key=\\abc12345") == "api_key=***"

    def test_the_colon_form_too(self) -> None:
        assert redact_secrets("password: pa\\ss12345") == "password: ***"

    def test_the_signed_link_form_too(self) -> None:
        assert redact_secrets("/ds/x?key=ab\\cd12345&v=1") == "/ds/x?key=***&v=1"

    def test_the_percent_encoded_form_too(self) -> None:
        assert (
            redact_secrets("api_key%3Dab\\cd12345%26v%3D1") == "api_key%3D***%26v%3D1"
        )


class TestAnEmptyQuotedValueIsLeftAlone:
    """`token=""` has nothing to mask, and must not eat the text up to the next quote."""

    @pytest.mark.parametrize(
        "text",
        [
            'download(token="", dataset="owner/ds")',
            "api_key='' for workspace 'coffee'",
            "password: \"\" and 'x'",
            'token=\\"\\", x=\\"y\\"',
            'token="" token=""',
        ],
    )
    def test_the_text_is_unchanged(self, text: str) -> None:
        assert redact_secrets(text) == text

    def test_a_filled_value_next_to_an_empty_one_is_masked_alone(self) -> None:
        out = redact_secrets('f(token="abc12345", dataset="owner/ds", api_key="")')

        assert out == 'f(token="***", dataset="owner/ds", api_key="")'


class TestQuotedValuesWithSpaces:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('password: "my long phrase"', 'password: "***"'),
            ("password: 'my long phrase' and more", "password: '***' and more"),
            ("api_key='two words'", "api_key='***'"),
            ('secret = "a b c", next=1', 'secret = "***", next=1'),
        ],
    )
    def test_everything_up_to_the_closing_quote_is_masked(
        self, text: str, expected: str
    ) -> None:
        assert redact_secrets(text) == expected

    def test_an_unterminated_quote_masks_the_first_word_at_least(self) -> None:
        assert "abc" not in redact_secrets('password: "abc def')

    def test_the_closing_quote_must_be_on_the_same_line(self) -> None:
        out = redact_secrets('password: "abc\ndef"')

        assert "abc" not in out
        assert out.endswith('\ndef"')

    def test_a_string_literal_that_ends_in_the_name_is_not_a_value(self) -> None:
        # What a traceback prints for the Roboflow client's own request line.
        line = 'response = requests.post(API_URL + "/?api_key=" + api_key + "/x")'

        assert redact_secrets(line) == line


# One adversarial line per way the patterns can backtrack. Each is ~50k characters:
# long enough that a quadratic pattern needs seconds, short enough that a linear
# one needs milliseconds.
_N = 50_000
_ADVERSARIAL = {
    "scheme-like run": "a-" * (_N // 2),
    "bare name": "token" * (_N // 5),
    "named pair repeated": "api_key=" * (_N // 8),
    "name then spaces": "api_key" + " " * _N,
    "colon then spaces": "password:" + " " * _N,
    "quotes opened and never closed": "password: '" * (_N // 11),
    "one quote then a long line": "password: '" + "x " * (_N // 2),
    "mixed quote kinds": "password:'password:\"api_key=\\'x" * (_N // 31),
    "escaped quotes": 'api_key=\\"' * (_N // 10),
    "signed link repeated": "?key=" * (_N // 5),
    "encoded pair repeated": "api_key%3D" * (_N // 10),
    "json pair repeated": '"api_key": "' * (_N // 12),
    "json value never closed": '"api_key": "' + "x" * _N,
    "header then spaces": "authorization" + " " * _N,
    "header repeated": "authorization: " * (_N // 15),
    "bearer then a long word": "bearer " + "a" * _N,
    "bearer repeated": "bearer " * (_N // 7),
    "userinfo colons": "x://" + "b:" * (_N // 2),
    "userinfo user": "a://" + "b" * _N,
    "scheme words": "bearer basic token " * (_N // 19),
    "only colons": ":" * _N,
    "only equals": "=" * _N,
    "only backslashes": "\\" * _N,
    "a value of backslash pairs": "api_key=" + "\\a" * (_N // 2),
    "a value of escaped quotes": "api_key=" + '\\"' * (_N // 2),
    "a value of escape letters": "password: " + "\\n" * (_N // 2),
    "empty quoted values": 'token="" ' * (_N // 9),
    "only quotes": "'" * _N,
}


class TestLinearTime:
    """Redaction runs on traceback text inside a request, so it must stay linear.

    The URL-userinfo pattern took ~10 s on 50k characters of `a-`.
    """

    @pytest.mark.parametrize("name", list(_ADVERSARIAL))
    def test_every_pattern_is_fast_on_an_adversarial_line(self, name: str) -> None:
        text = _ADVERSARIAL[name]
        assert len(text) >= _N * 9 // 10
        for pattern in _PATTERNS:
            started = time.perf_counter()
            pattern.sub(_mask_value, text)
            elapsed = time.perf_counter() - started
            assert elapsed < 0.5, (
                f"{pattern.pattern[:70]!r} took {elapsed:.2f}s on {name!r}"
            )

    @pytest.mark.parametrize("name", list(_ADVERSARIAL))
    def test_the_whole_redaction_is_fast_with_a_known_secret(self, name: str) -> None:
        text = _ADVERSARIAL[name]
        started = time.perf_counter()
        redact_secrets(text, "KNOWN-SECRET-VALUE")
        assert time.perf_counter() - started < 1.0


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

    def test_a_seven_character_value_is_not_masked_as_a_bare_substring(self) -> None:
        # A junk saved value like "test" used to turn trace paths into
        # "***s/foo/***_x.py". Real keys are 20+ characters.
        text = "File tests/foo/test_x.py, line 3"

        assert redact_secrets(text, "tests/f") == text
        assert redact_secrets(text, "test") == text

    def test_an_eight_character_value_is_masked(self) -> None:
        out = redact_secrets("bad key abcd1234 here", "abcd1234")

        assert "abcd1234" not in out

    def test_a_short_value_is_still_caught_by_shape(self) -> None:
        assert "abc123" not in redact_secrets("?api_key=abc123", "abc123")

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
