import asyncio
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest
from mobilerun_sdk import APIStatusError
from mobilerun_sdk.types.device_retrieve_capabilities_response import Capabilities

from mobilerun.clients.cloud import apps, devices, proxies, tasks
from mobilerun.core.devices import CloudDevice
from mobilerun.errors import CloudValidationError, DeviceOperationNotSupported
from mobilerun.models.apps import App
from mobilerun.models.billing import Billing


@pytest.mark.parametrize("supported", [False, True, None])
def test_proxy_capability_controls_adoption_and_operations(monkeypatch, supported):
    phone = CloudDevice("phone", api_key="test")
    info = NS(capabilities=NS(telephony=False, proxy=supported))
    monkeypatch.setattr(devices, "capabilities", AsyncMock(return_value=info))
    status = AsyncMock(return_value=NS(connected=False, name=None, protocol=None))
    monkeypatch.setattr(proxies, "device_status", status)
    asyncio.run(phone.connect())
    assert status.await_count == int(supported is not False)
    assert ("proxy_switch" in phone.supported_operations) == (supported is not False)


@pytest.mark.parametrize("code", ["DEVICE_TOOL_UNSUPPORTED", "OTHER"])
def test_missing_proxy_capability_only_tolerates_unsupported(monkeypatch, code):
    phone = CloudDevice("phone", api_key="test")
    info = NS(capabilities=Capabilities.construct(telephony=False))
    monkeypatch.setattr(devices, "capabilities", AsyncMock(return_value=info))
    sdk_error = APIStatusError(
        "invalid request",
        response=httpx.Response(
            400, request=httpx.Request("GET", "https://example.test")
        ),
        body={"code": code},
    )
    status = AsyncMock(side_effect=sdk_error)
    client = NS(devices=NS(proxy=NS(status=status)))
    monkeypatch.setattr(proxies, "cloud_client", lambda *a, **kw: nullcontext(client))
    if code == "DEVICE_TOOL_UNSUPPORTED":
        asyncio.run(phone.connect())
        assert "proxy_switch" not in phone.supported_operations
    else:
        with pytest.raises(CloudValidationError):
            asyncio.run(phone.connect())
    status.assert_awaited_once()


def test_unknown_proxy_handles_typed_error_without_cause(monkeypatch):
    phone = CloudDevice("phone", api_key="test")
    info = NS(capabilities=NS(telephony=False, proxy=None))
    monkeypatch.setattr(devices, "capabilities", AsyncMock(return_value=info))
    monkeypatch.setattr(
        proxies,
        "device_status",
        AsyncMock(side_effect=DeviceOperationNotSupported("Unsupported proxy status")),
    )
    asyncio.run(phone.connect())
    assert "proxy_switch" not in phone.supported_operations


def test_install_polls_background_request(monkeypatch):
    phone = CloudDevice("phone", api_key="test")
    app = App("pkg", "1", "Example", id="lib")

    async def submit(*args, background=False, **kwargs):
        if not background:
            raise TimeoutError("foreground install exceeded SDK timeout")

    monkeypatch.setattr(apps, "install", submit)
    attempt = NS(app_id="pkg", started_at=1, status="succeeded")
    polls = AsyncMock(side_effect=[NS(installs=[]), NS(installs=[attempt])])
    monkeypatch.setattr(apps, "list_installs", polls)
    monkeypatch.setattr(
        phone,
        "list_apps",
        AsyncMock(return_value=[{"package_name": "pkg", "version_name": "1"}]),
    )
    assert asyncio.run(phone.install_app(app)) == app
    assert polls.await_count == 2


def test_progress_disconnect_polls_without_stopping(monkeypatch):
    monkeypatch.setattr(tasks, "run", AsyncMock(return_value=NS(id="id")))
    monkeypatch.setattr(
        tasks, "_progress", AsyncMock(side_effect=httpx.ReadError("disconnect"))
    )
    status = AsyncMock(
        return_value=NS(status="completed", succeeded=True, message="ok", steps=1)
    )
    monkeypatch.setattr(tasks, "status", status)
    stop = AsyncMock()
    monkeypatch.setattr(tasks, "_stop", stop)
    result = asyncio.run(
        tasks.run_task(None, device_id="phone", task="fake", billing=Billing.PAID)
    )
    assert result.success
    status.assert_awaited_once()
    stop.assert_not_awaited()
