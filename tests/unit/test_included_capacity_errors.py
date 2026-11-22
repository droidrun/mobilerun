import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from mobilerun_sdk import APIStatusError

from mobilerun.clients.cloud import mailboxes, numbers
from mobilerun.errors import BillingError, CloudConflict
from mobilerun.errors.base import is_value_safe
from mobilerun.models.billing import Billing


@pytest.mark.parametrize(
    "module,resource,plural",
    [(numbers, "numbers", "phone numbers"), (mailboxes, "mailboxes", "mailboxes")],
)
@pytest.mark.parametrize("code", ["included_capacity_unavailable", "other_conflict"])
def test_create_capacity_conflict(module, resource, plural, code):
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(409, request=request)
    error = APIStatusError("conflict", response=response, body={"code": code})
    sdk = SimpleNamespace(
        **{resource: SimpleNamespace(create=AsyncMock(side_effect=error))}
    )
    with pytest.raises(
        BillingError if code == "included_capacity_unavailable" else CloudConflict
    ) as caught:
        asyncio.run(
            module.create(
                billing=Billing.PAID,
                client=sdk,
                **({"country": "US"} if resource == "numbers" else {}),
            )
        )
    assert is_value_safe(caught.value)
    if code == "included_capacity_unavailable":
        assert f"No included {plural} left" in str(caught.value)
        if resource == "numbers":
            assert "No included phone numbers left for US." in str(caught.value)
        assert 'plan="auto"' in str(caught.value)
        assert 'plan="rent"' in str(caught.value)
        assert 'billing="paid"' in str(caught.value)
    else:
        assert "Wait for the current operation" in str(caught.value)


@pytest.mark.parametrize(
    "country,expected",
    [
        ("us", "US"),
        ("gB", "GB"),
        ("US", "US"),
        ("PRIVATE-CANARY", None),
        ("UＳ", None),
        ("ß", None),
        (" US ", None),
        (123, None),
        (None, None),
    ],
)
def test_capacity_country_is_validated_before_rendering(country, expected):
    from mobilerun.errors import IncludedCapacityUnavailable

    error = IncludedCapacityUnavailable("number", country=country)
    assert is_value_safe(error)
    assert error.country == expected
    assert (f"for {expected}." if expected else "for the requested country.") in str(
        error
    )
    assert "PRIVATE-CANARY" not in str(error)


def test_capacity_country_from_client_never_renders_arbitrary_text():
    request = httpx.Request("POST", "https://example.test")
    error = APIStatusError(
        "PRIVATE-BACKEND",
        response=httpx.Response(409, request=request),
        body={"code": "included_capacity_unavailable"},
    )
    sdk = SimpleNamespace(numbers=SimpleNamespace(create=AsyncMock(side_effect=error)))
    with pytest.raises(BillingError) as caught:
        asyncio.run(
            numbers.create(
                billing=Billing.INCLUDED, client=sdk, country="PRIVATE-COUNTRY"
            )
        )
    assert is_value_safe(caught.value)
    assert "PRIVATE" not in str(caught.value)


def test_capacity_error_rejects_unknown_resource_without_echoing_it():
    from mobilerun.errors import IncludedCapacityUnavailable

    with pytest.raises(ValueError) as caught:
        IncludedCapacityUnavailable("PRIVATE-RESOURCE")
    assert "PRIVATE" not in str(caught.value)
