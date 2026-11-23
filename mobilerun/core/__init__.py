"""Common interfaces shared by the Mobilerun framework."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mobilerun.core.devices import (
        AndroidDevice,
        CloudDevice,
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


_EXPORT_MODULES = {
    **{name: "mobilerun.core.devices" for name in __all__[:8]},
    "Workspace": "mobilerun.core.workspace",
    **{name: "mobilerun.core.proxy" for name in __all__[9:]},
}


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MODULES:
        return getattr(import_module(_EXPORT_MODULES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
