"""Capability lookup preserves billing, transport selection and typed errors."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest
from mobilerun_sdk import APIStatusError

from mobilerun.clients.cloud import devices
from mobilerun.errors import CloudPermissionDenied
from mobilerun.models import Billing


def test_capabilities_uses_given_client():
    retrieve = AsyncMock(return_value=NS(capabilities=NS(telephony=True)))
    result = asyncio.run(
        devices.capabilities(
            "phone",
            client=NS(devices=NS(retrieve_capabilities=retrieve)),
            billing=Billing.NO_COMMERCE,
        )
    )
    assert result.capabilities.telephony
    retrieve.assert_awaited_once_with("phone")


def test_capabilities_opens_client_when_needed(monkeypatch):
    retrieve = AsyncMock(return_value="capabilities")
    context = AsyncMock()
    context.__aenter__.return_value = NS(devices=NS(retrieve_capabilities=retrieve))

    def factory(*args, **kwargs):
        return context

    monkeypatch.setattr(devices, "cloud_client", factory)
    assert (
        asyncio.run(
            devices.capabilities(
                "phone",
                billing=Billing.NO_COMMERCE,
                api_key="fake",
                base_url="https://example.test",
            )
        )
        == "capabilities"
    )
    retrieve.assert_awaited_once_with("phone")
    context.__aexit__.assert_awaited_once()


def test_capabilities_maps_sdk_error():
    error = APIStatusError(
        "private response",
        response=httpx.Response(
            403, request=httpx.Request("GET", "https://example.test")
        ),
        body={},
    )
    retrieve = AsyncMock(side_effect=error)
    with pytest.raises(CloudPermissionDenied) as raised:
        asyncio.run(
            devices.capabilities(
                "phone",
                billing=Billing.NO_COMMERCE,
                client=NS(devices=NS(retrieve_capabilities=retrieve)),
            )
        )
    assert raised.value.__cause__ is error
    assert "private response" not in str(raised.value)
