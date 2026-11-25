import logging
import re
import time

import pytest

from mobilerun.utils.redaction import _redact_secrets, secret_scope


@pytest.mark.parametrize(
    ("value", "display"),
    [
        ("4111111111111111", "4111 1111 1111 1111"),
        ("12/27", "12 / 27"),
        ("1227", "12 / 27"),
        ("4826", "4 - 8.2 / 6"),
        ("4111111111111111", "4111  1111 1111 1111"),
        ("1227", "12  /  27"),
    ],
)
def test_formatted_digits_redacted_in_log(caplog, value, display):
    with secret_scope(value):
        logging.getLogger("variant-test").warning("value: %s", display)
    assert caplog.records[-1].getMessage() == "value: ***"


@pytest.mark.parametrize("value", ["id:234567", "api234567"])
def test_alphanumeric_secrets_are_literal_only(value):
    with secret_scope(value):
        assert _redact_secrets(f"{value}; order 234-567") == "***; order 234-567"


@pytest.mark.parametrize("reverse", [False, True])
def test_overlapping_secret_variants_leave_no_suffix(reverse):
    values = ("4111", "4111111111111111", "label", "label-private")
    with secret_scope(*(values[::-1] if reverse else values)):
        assert _redact_secrets("4111 1111 1111 1111 label-private") == "*** ***"


def test_short_numbers_and_longer_unrelated_digit_runs():
    with secret_scope("1/2", "4826"):
        message = "Order 12; transaction 948267; other 94-8267"
        assert _redact_secrets(message) == message
        assert _redact_secrets("literal 1/2") == "literal ***"


def test_lone_surrogate_retains_literal_and_escaped_protection():
    value = "test\ud800"
    with secret_scope(value):
        assert _redact_secrets(value + " " + repr(value)[1:-1]) == "*** ***"


def test_no_active_scope_returns_message_unchanged():
    message = "Order 12; 4111 1111 1111 1111"
    assert _redact_secrets(message) is message


@pytest.mark.parametrize(
    ("value", "message", "expected"),
    [
        ("12/27", "dates 2026/12/27 and 12/27/99", "dates 2026/12/27 and 12/27/99"),
        (
            "1227",
            "dates 2026 / 12 / 27 and 12.27-99",
            "dates 2026 / 12 / 27 and 12.27-99",
        ),
        (
            "123456",
            "order 99-123-456 and 123.456.77",
            "order 99-123-456 and 123.456.77",
        ),
        ("123456", "code 123456, ID 123-456.", "code ***, ID ***."),
        ("12/27", "expires 12 / 27.", "expires ***."),
        ("4826", "PIN 4826. 2 attempts remain", "PIN ***. 2 attempts remain"),
        ("12/27", "12.27.99 and 12/27.", "12.27.99 and ***."),
        ("4826", "items 2  4826  99", "items 2  ***  99"),
        (
            "4111111111111111",
            "order 9 card 4111 1111 1111 1111 2 items",
            "order 9 card *** 2 items",
        ),
    ],
)
def test_numeric_candidate_boundaries(value, message, expected):
    with secret_scope(value):
        assert _redact_secrets(message) == expected


def test_secret_patterns_do_not_enter_process_regex_cache():
    marker = "regexcachecanary723819"
    with secret_scope(marker):
        assert _redact_secrets(marker) == "***"
    assert not any(marker in str(key) for key in re._cache)


def test_large_active_scope_scan_budget():
    text = ("trace message with id 19. " * 2600)[:65536]
    with secret_scope(
        *(str(200000 + i) if i % 2 else f"label-{200000+i}" for i in range(128))
    ):
        start = time.process_time()
        for _ in range(4):
            assert _redact_secrets(text) is text
        assert time.process_time() - start < 0.5
