import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from mobilerun_sdk import AsyncMobilerun

from mobilerun.clients.cloud import apps
from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.models.billing import Billing


def test_uninstall_sends_required_empty_json_body():
    requests = []

    async def respond(request):
        requests.append(request)
        return httpx.Response(204, request=request)

    async def run():
        http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        async with AsyncMobilerun(
            api_key="test", base_url="https://example.test/v1", http_client=http
        ) as client:
            await apps.uninstall(
                client, "ai.example.app", billing=Billing.NO_COMMERCE, device_id="phone"
            )

    asyncio.run(run())
    assert len(requests) == 1
    assert requests[0].method == "DELETE"
    assert requests[0].url.path == "/v1/devices/phone/apps/ai.example.app"
    assert requests[0].content == b"{}"


def test_cloud_stop_uses_sdk_and_display_without_clearing_data():
    device = CloudDevice(device_id="phone", api_key="test", display_id=7)
    device._ensure_connected = AsyncMock()
    stop = AsyncMock()
    device._write_client = SimpleNamespace(
        devices=SimpleNamespace(apps=SimpleNamespace(stop=stop))
    )

    assert asyncio.run(device.stop_app("ai.example.app")) == "Stopped ai.example.app"
    stop.assert_awaited_once_with(
        "ai.example.app", device_id="phone", x_device_display_id=7
    )
    assert "stop_app" in device.supported_operations
