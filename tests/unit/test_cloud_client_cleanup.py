import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.core.devices.components import cloud_lifecycle as lifecycle


@pytest.mark.parametrize(
    "existing,keep,fail",
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (False, False, True),
    ],
)
def test_cloud_close_closes_owned_client(monkeypatch, existing, keep, fail):
    from mobilerun_core_local.driver import cloud

    client = SimpleNamespace(close=AsyncMock())
    client.with_options = lambda **kwargs: client
    monkeypatch.setattr(cloud, "AsyncMobilerun", lambda **kwargs: client)
    terminate = AsyncMock(
        side_effect=RuntimeError("terminate failed") if fail else None
    )
    monkeypatch.setattr(lifecycle.devices, "terminate", terminate)
    device = CloudDevice("existing" if existing else None, api_key="test", keep=keep)
    if not existing:
        device.device_id = "created"
        device._owned = True

    async def run():
        if fail:
            with pytest.raises(RuntimeError, match="terminate failed"):
                await device.close()
            client.close.assert_not_awaited()
        else:
            await device.close()
            client.close.assert_awaited_once()

    asyncio.run(run())


def test_failed_termination_can_retry_with_same_owned_sdk_client(monkeypatch):
    import httpx
    from mobilerun_sdk import AsyncMobilerun
    from mobilerun_core_local.driver import cloud

    attempts = []

    def respond(request):
        attempts.append(request)
        return httpx.Response(400, json={"error": "failed"}) if len(attempts) == 1 else httpx.Response(204)

    sdk = AsyncMobilerun(api_key="test", base_url="https://example.test", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(cloud, "AsyncMobilerun", lambda **kwargs: sdk)
    device = CloudDevice(api_key="test")
    device.device_id = "created"
    device._owned = True

    async def run():
        with pytest.raises(Exception):
            await device.close()
        await device.close()
        assert len(attempts) == 2
        assert sdk.is_closed()

    asyncio.run(run())
