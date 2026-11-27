"""Device actions and explicit behavioral wrappers."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mobilerun.core.devices.android import AndroidDevice
    from mobilerun.core.devices.base import Device
    from mobilerun.core.devices.cloud import CloudDevice
    from mobilerun.core.devices.ios import IOSDevice
    from mobilerun.core.devices.pool import DevicePool
    from mobilerun.core.devices.recording import RecordingDevice
    from mobilerun.core.devices.remote import RemoteDevice
    from mobilerun.core.devices.stealth import StealthDevice

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


_EXPORT_MODULES = dict(
    zip(
        __all__,
        (
            "base",
            "android",
            "cloud",
            "ios",
            "remote",
            "pool",
            "recording",
            "stealth",
        ),
        strict=True,
    )
)


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MODULES:
        return getattr(import_module(f"{__name__}.{_EXPORT_MODULES[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
