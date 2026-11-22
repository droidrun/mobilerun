"""Read-only cloud checks, plus explicitly selected billable lifecycle checks."""

import asyncio

import pytest
from conftest import CLOUD_ID

from mobilerun.core.credentials import DictCredentials
from mobilerun.core.devices import CloudDevice
from mobilerun.errors import LocalCredentialsNotAllowed

pytestmark = pytest.mark.smoke_cloud


def test_cloud_info_and_wait_ready(cloud_ready):
    async def run():
        device = CloudDevice(CLOUD_ID)
        assert (await device.info()).id == CLOUD_ID
        await device.wait_ready(timeout=30)

    asyncio.run(run())


def test_cloud_rejects_local_secrets(cloud_ready):
    async def run():
        device = CloudDevice(CLOUD_ID)
        await DictCredentials({"test": "not-a-real-secret"}).bind(device)
        with pytest.raises(LocalCredentialsNotAllowed):
            await device.do("This must never be submitted")

    asyncio.run(run())


def test_paid_cloud_create_lifecycle(cloud_ready):
    async def run():
        device = await CloudDevice(name="SDK lifecycle smoke", billing="paid")
        try:
            info = await device.info()
            assert info.id == device._adb_device_id
            assert info.type == "android_cloud_phone"
            await device.stop()
            await device.resume()
            await device.wait_ready()
            await device.reboot()
            await device.wait_ready()
            await device.reset(mode="identity")
            await device.wait_ready()
        finally:
            await device.terminate()

    asyncio.run(run())
