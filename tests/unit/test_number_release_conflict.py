"""Number release conflict diagnostics never expose server messages."""

import httpx
import pytest
from mobilerun_sdk import APIStatusError

from mobilerun.clients.cloud.errors import cloud_error
from mobilerun.errors import CloudConflict
from mobilerun.errors.base import is_value_safe


def sdk_conflict(body):
    response = httpx.Response(
        409, request=httpx.Request("DELETE", "https://example.test")
    )
    return APIStatusError("server secret", response=response, body=body)


def test_not_cancellable_number_release_has_specific_safe_message():
    error = cloud_error(
        sdk_conflict({"code": "not_cancellable", "message": "server secret"}),
        "Release number",
    )
    assert isinstance(error, CloudConflict)
    assert str(error) == (
        "Release number failed (HTTP 409). This number can't be released: "
        "only unpaid pending purchases and active rented numbers can be cancelled."
    )
    assert is_value_safe(error)


@pytest.mark.parametrize("body", [None, "invalid json", [], {}, {"code": "other"}])
def test_other_conflicts_keep_generic_message(body):
    error = cloud_error(sdk_conflict(body), "Release number")
    assert type(error) is CloudConflict
    assert str(error) == (
        "Release number failed (HTTP 409). "
        "Wait for the current operation to finish and retry."
    )


def test_not_cancellable_code_for_other_operation_stays_generic():
    error = cloud_error(sdk_conflict({"code": "not_cancellable"}), "Release mailbox")
    assert type(error) is CloudConflict
