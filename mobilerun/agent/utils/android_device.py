"""Android device selection."""

import os

from async_adbutils import adb
from mobilerun_core_local.driver.android import AndroidPortalHttpDriver


class AndroidDeviceSelectionError(ValueError):
    """No single Android device could be selected."""


def is_portal_http_url(device: str | None) -> bool:
    """True when ``device`` is a Portal HTTP base URL rather than an ADB serial."""
    return bool(device) and device.lower().startswith(("http://", "https://"))


def create_android_portal_http_driver(
    url: str, token: str | None
) -> AndroidPortalHttpDriver:
    """Return an unconnected driver that reaches Portal over HTTP, without ADB."""
    if not token:
        raise ValueError(
            "Android Portal HTTP needs the Portal auth token. Copy it from the "
            "Portal app and set MOBILERUN_DEVICE_TOKEN or device.auth_token."
        )
    return AndroidPortalHttpDriver(url, token)


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
