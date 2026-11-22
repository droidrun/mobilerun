"""Shared Mobilerun device framework; LLM APIs live in the agents package."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from typing import TYPE_CHECKING

from mobilerun._legacy.aliases import install_legacy_finder
from mobilerun._legacy.aliases import resolve_export as __getattr__  # noqa: F401
from mobilerun.core import (
    AndroidDevice,
    CloudFixedProxy,
    CloudRotatingProxy,
    CloudSocks5Proxy,
    Device,
    DevicePool,
    IOSDevice,
    LocalProxy,
    RecordingDevice,
    RemoteDevice,
    StealthDevice,
    Workspace,
)

if TYPE_CHECKING:
    from mobilerun.core.devices.cloud import CloudDevice

from mobilerun.config.user_config import (
    DeviceConfig,
    LoggingConfig,
    MobileConfig,
    TelemetryConfig,
    TracingConfig,
)
from mobilerun.version import __version__

install_legacy_finder()

__all__ = [
    "__version__",
    "DeviceConfig",
    "LoggingConfig",
    "MobileConfig",
    "TelemetryConfig",
    "TracingConfig",
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
