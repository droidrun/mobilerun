"""Discover a single local device for interactive CLI use."""

from async_adbutils import adb
from mobilerun_core_local.driver.ios import discover_ios_device as discover_ios_device

from mobilerun.errors import DeviceNotFound


async def find_android_serial() -> str:
    devices = await adb.list()
    online = [device.serial for device in devices if device.state == "device"]
    if len(online) == 1:
        return online[0]
    if not online:
        message = "No connected Android device found."
        not_ready = [f"{device.serial} ({device.state})" for device in devices]
        if not_ready:
            message += f" Not ready: {', '.join(not_ready)}."
        raise DeviceNotFound(message)
    raise DeviceNotFound(
        f"Several devices connected: {', '.join(online)}. "
        "Pass DEVICE or run `mobilerun device connect <id>`."
    )
