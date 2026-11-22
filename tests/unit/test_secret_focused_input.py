import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.devices.base import Device
from mobilerun.errors import DeviceOperationNotSupported, NoFocusedInput


@pytest.mark.parametrize(
    "phone_state",
    [
        {"focusedElement": None, "isEditable": False},
        {"focusedElement": {"className": "Button"}, "isEditable": False},
    ],
)
def test_secret_refuses_unfocused_or_noneditable_input(phone_state):
    async def run():
        device = Device()
        device.state.refresh = AsyncMock(
            return_value=SimpleNamespace(phone_state=phone_state)
        )
        device._input_text_raw = AsyncMock(return_value=True)
        with pytest.raises(NoFocusedInput, match="No text field is focused") as error:
            await device._input_secret("secret-canary")
        assert "secret-canary" not in str(error.value)
        device._input_text_raw.assert_not_awaited()

    asyncio.run(run())


def test_secret_types_only_when_focused_and_editable():
    async def run():
        device = Device()
        device.state.refresh = AsyncMock(
            return_value=SimpleNamespace(
                phone_state={
                    "focusedElement": {"className": "EditText"},
                    "isEditable": True,
                }
            )
        )
        device._input_text_raw = AsyncMock(return_value=True)
        assert await device._input_secret("demo-user")
        device._input_text_raw.assert_awaited_once_with("demo-user")

    asyncio.run(run())


def test_secret_rejects_unknown_focus_capability():
    async def run():
        device = Device()
        device.state.refresh = AsyncMock(
            return_value=SimpleNamespace(
                phone_state={"observationMode": "screenshot_only"}
            )
        )
        device._input_text_raw = AsyncMock(return_value=True)
        with pytest.raises(DeviceOperationNotSupported):
            await device._input_secret("secret-canary")
        device._input_text_raw.assert_not_awaited()

    asyncio.run(run())
