import asyncio
from unittest.mock import AsyncMock

from mobilerun.core.devices.android import AndroidDevice


def test_close_disables_portal_keyboard():
    device = AndroidDevice(serial="test")
    device._adb_device = type("ADB", (), {"shell": AsyncMock()})()
    asyncio.run(device.close())
    assert device._adb_device.shell.await_count == 1
    assert device._adb_device.shell.await_args.args[0].startswith("ime disable ")


def test_close_ignores_keyboard_cleanup_failure():
    device = AndroidDevice(serial="test")
    device._adb_device = type(
        "ADB", (), {"shell": AsyncMock(side_effect=RuntimeError())}
    )()
    asyncio.run(device.close())
    assert device._adb_device.shell.await_count == 1
