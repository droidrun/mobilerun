"""Real device operations; leave devices on the home screen."""

import asyncio

import pytest
from conftest import CLOUD_ID, SERIAL


async def exercise(device):
    await device.connect()
    try:
        assert (await device.screenshot()).startswith(b"\x89PNG\r\n\x1a\n")
        await device.start_app("com.android.settings")
        state = await device.state.refresh()
        assert state.elements
        assert await device.list_apps()
        await device.swipe(200, 800, 200, 400, duration_ms=300)
        await device.tap(100, 100)
    finally:
        await device.press_button("home")


@pytest.mark.smoke_device
def test_local_device(local_ready):
    from mobilerun.core.devices import AndroidDevice

    asyncio.run(exercise(AndroidDevice(serial=SERIAL)))


@pytest.mark.smoke_device
def test_local_wrapper_and_pool_state(local_ready):
    from mobilerun import AndroidDevice, DevicePool, RecordingDevice

    async def run():
        async with AndroidDevice(SERIAL) as root:
            wrapped = RecordingDevice(root)
            await wrapped.connect()
            await wrapped.connect()
            try:
                await wrapped.start_app("com.android.settings")
                states = await DevicePool([wrapped]).state.refresh()
                assert states[0] is wrapped.state.current
                assert states[0].elements
                await wrapped.press_button("home")
                assert wrapped.recorded_actions[-1] == {
                    "action_type": "button_press",
                    "button": "home",
                }
            finally:
                await root.press_button("home")

    asyncio.run(run())


@pytest.mark.smoke_cloud
def test_cloud_device(cloud_ready):
    from mobilerun.core.devices.cloud import CloudDevice

    asyncio.run(exercise(CloudDevice(device_id=CLOUD_ID)))


@pytest.mark.smoke_cloud
def test_cloud_pool(cloud_ready):
    from mobilerun.core.devices.cloud import CloudDevice
    from mobilerun.core.devices.pool import DevicePool

    async def run():
        device = CloudDevice(device_id=CLOUD_ID)
        await device.connect()
        try:
            images = await DevicePool([device]).screenshot()
            assert len(images) == 1 and images[0].startswith(b"\x89PNG\r\n\x1a\n")
        finally:
            await device.press_button("home")

    asyncio.run(run())
