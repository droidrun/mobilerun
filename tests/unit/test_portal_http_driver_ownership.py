import httpx
import pytest
from mobilerun_core_local.driver.android import AndroidPortalHttpDriver

from mobilerun.cli.devices.base import selection
from mobilerun.config.user_config import MobileConfig
from mobilerun.core.devices.components.wrapper import DriverDevice
from mobilerun.core.state.options import android_state_provider


@pytest.fixture
def driver():
    return AndroidPortalHttpDriver(
        "http://portal.test",
        "test-token",
        transport=httpx.MockTransport(lambda _: httpx.Response(200)),
    )


@pytest.mark.anyio
async def test_cli_device_closes_owned_portal_client(driver, monkeypatch):
    monkeypatch.setattr(
        selection, "create_android_portal_http_driver", lambda *_: driver
    )
    device = await selection.connect_target(
        driver.url, kind="android", config=MobileConfig()
    )
    client = driver._client
    assert not client.is_closed
    await device.close()
    assert client.is_closed


@pytest.mark.anyio
async def test_driver_device_leaves_borrowed_portal_client_open(driver):
    device = DriverDevice(driver, state_provider_factory=android_state_provider)
    await device.connect()
    client = driver._client
    try:
        await device.close()
        assert not client.is_closed
    finally:
        await client.aclose()
