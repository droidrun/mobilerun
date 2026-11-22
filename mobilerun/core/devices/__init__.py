"""Device actions and explicit behavioral wrappers."""

from typing import TYPE_CHECKING, Any

from mobilerun.core.devices.android import AndroidDevice
from mobilerun.core.devices.base import Device
from mobilerun.core.devices.ios import IOSDevice
from mobilerun.core.devices.pool import DevicePool
from mobilerun.core.devices.recording import RecordingDevice
from mobilerun.core.devices.remote import RemoteDevice
from mobilerun.core.devices.stealth import StealthDevice

if TYPE_CHECKING:
    from mobilerun.core.devices.cloud import CloudDevice

__all__ = [
    "Device",
    "AndroidDevice",
    "CloudDevice",
    "IOSDevice",
    "RemoteDevice",
    "DevicePool",
    "RecordingDevice",
    "StealthDevice",
]


def __getattr__(name: str) -> Any:
    if name == "CloudDevice":
        from mobilerun.core.devices.cloud import CloudDevice

        return CloudDevice
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
