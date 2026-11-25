"""Local-only replay through a recorder attached to a real device."""

import asyncio

import pytest
from conftest import SERIAL


@pytest.mark.smoke_device
def test_record_named_home_then_replay_wait(local_ready):
    from mobilerun import AndroidDevice, RecordingDevice
    from mobilerun.core.devices import Device
    from mobilerun.core.extensions import Extension, device_action
    from mobilerun.macro.player import MacroPlayer

    class Navigation(Extension):
        @device_action()
        async def home(self, device: Device) -> None:
            """Return the emulator to its home screen."""
            await device.press_button("home")

    async def run():
        async with RecordingDevice(AndroidDevice(SERIAL)) as recorder:
            navigation = await Navigation(name="navigation").bind(recorder)
            await navigation.home()
            wait = {"action_type": "wait", "duration": 0.01}
            await MacroPlayer(recorder).replay_action(wait)
            assert recorder.recorded_actions == [
                {
                    "action_type": "extension_call",
                    "name": "navigation.home",
                    "arguments": {},
                },
                wait,
            ]

    asyncio.run(run())
