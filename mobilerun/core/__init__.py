"""Common interfaces shared by the Mobilerun framework."""

from typing import TYPE_CHECKING, Any

from mobilerun.core.devices import (
    AndroidDevice,
    Device,
    DevicePool,
    IOSDevice,
    RecordingDevice,
    RemoteDevice,
    StealthDevice,
)
from mobilerun.core.proxy import (
    CloudFixedProxy,
    CloudRotatingProxy,
    CloudSocks5Proxy,
    LocalProxy,
)
from mobilerun.core.workspace import Workspace

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
    "Workspace",
    "CloudFixedProxy",
    "CloudRotatingProxy",
    "CloudSocks5Proxy",
    "LocalProxy",
]


def __getattr__(name: str) -> Any:
    if name == "CloudDevice":
        from mobilerun.core.devices.cloud import CloudDevice

        return CloudDevice
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
