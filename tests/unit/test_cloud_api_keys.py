"""API validation maps status, transport errors and causes without secrets."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from mobilerun_sdk import APIConnectionError, APIStatusError

from mobilerun.clients.cloud import api_keys
from mobilerun.errors.cloud import (
    CloudAuthenticationError,
    CloudConnectionError,
    CloudPermissionDenied,
    CloudServiceUnavailable,
)


def fake_client(monkeypatch, error=None):
    endpoint = AsyncMock(side_effect=error)
    context = AsyncMock()
    context.__aenter__.return_value = SimpleNamespace(
        devices=SimpleNamespace(list=endpoint)
    )

    def factory(**kwargs):
        return context

    monkeypatch.setattr(api_keys, "cloud_client", factory)
    return endpoint


def test_valid_key_uses_sdk(monkeypatch):
    endpoint = fake_client(monkeypatch)
    asyncio.run(api_keys.validate_api_key("example-key", billing="included"))
    assert endpoint.await_args.kwargs["page_size"] == 1


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, CloudAuthenticationError),
        (403, CloudPermissionDenied),
        (500, CloudServiceUnavailable),
    ],
)
def test_http_errors(monkeypatch, status, expected):
    response = httpx.Response(
        status, request=httpx.Request("GET", "https://cloud.example/devices")
    )
    error = APIStatusError("private response", response=response, body=None)
    fake_client(monkeypatch, error)
    with pytest.raises(expected, match=f"HTTP {status}") as raised:
        asyncio.run(api_keys.validate_api_key("key", billing="included"))
    assert raised.value.__cause__ is error
    assert "private response" not in str(raised.value)


def test_network_error(monkeypatch):
    error = APIConnectionError(request=httpx.Request("GET", "https://example.com"))
    fake_client(monkeypatch, error)
    with pytest.raises(CloudConnectionError):
        asyncio.run(api_keys.validate_api_key("key", billing="included"))
