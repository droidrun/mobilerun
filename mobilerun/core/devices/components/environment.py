"""Environment controls for cloud-backed devices."""

import asyncio
import re
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mobilerun.clients.cloud import devices as cloud_devices
from mobilerun.core.devices.location import Location
from mobilerun.errors import DeviceOperationNotSupported

if TYPE_CHECKING:
    from mobilerun.core.devices.android import AndroidDevice
    from mobilerun.core.devices.cloud import CloudDevice


def validate_timezone(timezone: str) -> None:
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Invalid IANA timezone: {timezone}") from exc


def validate_locale(locale: str) -> None:
    if not re.fullmatch(
        r"[A-Za-z]{2,3}(?:-[A-Za-z]{4})?(?:-(?:[A-Za-z]{2}|[0-9]{3}))?", locale
    ):
        raise ValueError(f"Invalid locale: {locale}")


async def cloud_get_location(device: "CloudDevice") -> Location | None:
    await device._ensure_connected()
    result = await cloud_devices.get_location(
        device._require_device_id(), billing=device._billing, client=device._client
    )
    return (
        None
        if result is None
        else Location(latitude=result.latitude, longitude=result.longitude)
    )


async def cloud_set_location(
    device: "CloudDevice", latitude: float, longitude: float
) -> None:
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Invalid latitude or longitude")
    await device._ensure_connected()
    await device._call(
        device._write_client.devices.location.set(
            device._require_device_id(), latitude=latitude, longitude=longitude
        )
    )


async def cloud_reset_location(device: "CloudDevice") -> None:
    await device._ensure_connected()
    await device._call(
        device._write_client.devices.location.reset(device._require_device_id())
    )


async def cloud_set_language(device: "CloudDevice", locale: str) -> None:
    validate_locale(locale)
    await device._ensure_connected()
    await device._call(
        device._write_client.devices.language.set(
            device._require_device_id(), locale=locale
        )
    )


async def cloud_get_language(device: "CloudDevice") -> str:
    await device._ensure_connected()
    result = await cloud_devices.get_language(
        device._require_device_id(), billing=device._billing, client=device._client
    )
    return result.locale


async def cloud_get_timezone(device: "CloudDevice") -> str | None:
    await device._ensure_connected()
    result = await cloud_devices.get_timezone(
        device._require_device_id(), billing=device._billing, client=device._client
    )
    return result.timezone


async def cloud_set_timezone(device: "CloudDevice", timezone: str) -> None:
    validate_timezone(timezone)
    await device._ensure_connected()
    await device._call(
        device._write_client.devices.timezone.set(
            device._require_device_id(), timezone=timezone
        )
    )


async def cloud_set_clipboard(device: "CloudDevice", text: str) -> None:
    await device._ensure_connected()
    await device._call(
        device._write_client.devices.clipboard.set(
            device._require_device_id(), text=text
        )
    )


async def cloud_get_clipboard(device: "CloudDevice") -> str:
    await device._ensure_connected()
    response = await device._call(
        device._client.devices.clipboard.get(device._require_device_id())
    )
    return response.text


async def cloud_open_url(device: "CloudDevice", url: str) -> None:
    await device._ensure_connected()
    await device._call(
        device._write_client.devices.deep_link.execute_deep_link(
            device._require_device_id(), deep_link=url
        )
    )


async def android_shell(device: "AndroidDevice", command: str) -> None:
    await device._ensure_connected()
    adb = device._adb_device
    if adb is None:
        raise RuntimeError("Android device is not connected")
    result = await adb.shell2(command)
    if result.returncode:
        raise DeviceOperationNotSupported(f"Android command failed: {result.output}")


async def android_set_location(
    device: "AndroidDevice", latitude: float, longitude: float
) -> None:
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Invalid latitude or longitude")
    await device._ensure_connected()
    adb = device._adb_device
    if adb is None:
        raise RuntimeError("Android device is not connected")
    if not adb.serial.startswith("emulator-"):
        raise DeviceOperationNotSupported(
            "set_location on a physical Android device needs a mock-location app; supported on emulators and cloud phones"
        )
    result = await asyncio.to_thread(
        adb.adb_output, "emu", "geo", "fix", str(longitude), str(latitude)
    )
    if result is None or "KO" in result:
        raise DeviceOperationNotSupported(f"Emulator rejected location: {result}")
