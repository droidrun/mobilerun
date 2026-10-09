"""Android device selection."""

import os

from async_adbutils import adb


class AndroidDeviceSelectionError(ValueError):
    """No single Android device could be selected."""


async def resolve_android_serial(serial: str | None = None) -> str:
    """Return ``serial``, else ``ANDROID_SERIAL``, else the only online device."""
    if serial:
        return serial

    env_serial = os.environ.get("ANDROID_SERIAL", "").strip()
    if env_serial:
        return env_serial

    devices = await adb.list()
    online = [d.serial for d in devices if d.state == "device"]
    if len(online) == 1:
        return online[0]

    if not online:
        message = "No connected Android devices found."
        not_ready = [f"{d.serial} ({d.state})" for d in devices]
        if not_ready:
            message += f" Not ready: {', '.join(not_ready)}."
        raise AndroidDeviceSelectionError(message)

    raise AndroidDeviceSelectionError(
        f"Multiple Android devices connected: {', '.join(online)}. "
        "Pass -d/--device or set ANDROID_SERIAL."
    )
