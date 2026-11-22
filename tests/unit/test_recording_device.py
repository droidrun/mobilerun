from mobilerun_core_local.driver.base import DeviceDriver

from mobilerun.core.devices import Device
from mobilerun.core.devices.recording import RecordingDevice
from mobilerun.core.devices.stealth import StealthDevice
from mobilerun.core.state.view import UIStateView


class FakeDevice(Device):
    platform = "Android"
    capabilities = set()
    supported_buttons = set()

    def __init__(self):
        super().__init__()
        self.taps = []

    async def tap(self, x, y):
        self.taps.append((x, y))

    async def _ensure_connected(self):
        pass


def test_recording_device_is_device_and_records_tap_and_reuses_provider():
    inner = FakeDevice()
    inner._ui_state_view = UIStateView(inner, inner._create_state_provider)
    provider = object()
    inner._set_state_provider(provider)
    device = RecordingDevice(inner)
    assert isinstance(device, Device)
    assert not isinstance(device, DeviceDriver)
    assert device.platform == "Android"
    assert device.state._resolved_provider is provider

    import asyncio

    asyncio.run(device.tap(12, 34))
    assert inner.taps == [(12, 34)]
    assert device.recorded_actions == [{"action_type": "tap", "x": 12, "y": 34}]


def test_stealth_device_is_device():
    device = StealthDevice(FakeDevice())
    assert isinstance(device, Device)
    assert not isinstance(device, DeviceDriver)
    assert device.platform == "Android"


def test_tap_element_on_recording_device_records_resolved_tap():
    import asyncio

    class State:
        def get_element_coords(self, index):
            assert index == 0
            return 40, 60

    class Provider:
        capabilities = {"element_index"}

        async def get_state(self):
            return State()

    inner = FakeDevice()
    device = RecordingDevice(inner, state_provider=Provider())
    asyncio.run(device.state.refresh())
    asyncio.run(device.tap_element(0))
    assert inner.taps == [(40, 60)]
    assert device.recorded_actions == [{"action_type": "tap", "x": 40, "y": 60}]


def test_wrappers_close_inner_directly_and_through_pool():
    import asyncio
    from unittest.mock import AsyncMock

    from mobilerun.core.devices.pool import DevicePool

    for wrapper in (RecordingDevice, StealthDevice):
        for mode in ("close", "context", "pool"):
            inner = FakeDevice()
            inner.close = AsyncMock()
            device = wrapper(inner)

            async def run(mode=mode, device=device):
                if mode == "close":
                    await device.close()
                elif mode == "context":
                    async with device:
                        pass
                else:
                    await DevicePool([device]).close()

            asyncio.run(run())
            inner.close.assert_awaited_once()
