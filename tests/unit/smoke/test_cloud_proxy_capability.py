"""Read-only capability check for an existing cloud phone, including iOS."""

import asyncio

import pytest
from conftest import CLOUD_ID

from mobilerun.clients.cloud import devices
from mobilerun.core.devices import CloudDevice
from mobilerun.models.billing import Billing


@pytest.mark.smoke_cloud
def test_connected_phone_advertises_server_proxy_capability(cloud_ready):
    async def run():
        info = await devices.capabilities(CLOUD_ID, billing=Billing.NO_COMMERCE)
        async with CloudDevice(CLOUD_ID, billing=Billing.NO_COMMERCE) as phone:
            assert (
                "proxy_switch" in phone.supported_operations
            ) == info.capabilities.proxy
            assert await phone.screenshot()

    asyncio.run(run())
