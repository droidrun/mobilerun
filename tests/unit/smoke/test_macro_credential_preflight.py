"""Protected credential replay must not navigate the attached emulator."""

import asyncio

import pytest
from conftest import SERIAL

from mobilerun import AndroidDevice, RecordingDevice
from mobilerun.macro.errors import MacroError
from mobilerun.macro.player import MacroPlayer


@pytest.mark.smoke_device
def test_protected_step_prevents_navigation(local_ready):
    async def run():
        async with RecordingDevice(AndroidDevice(SERIAL)) as recorder:
            macro = {
                "macro_schema_version": "2.0",
                "actions": [
                    {"type": "button_press", "button": "home"},
                    {"type": "type_secret", "name": "example", "protected": True},
                ],
            }
            with pytest.raises(MacroError, match="approval"):
                await MacroPlayer(recorder).replay_macro(macro)
            assert not recorder.recorded_actions

    asyncio.run(run())
