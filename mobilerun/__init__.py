"""Shared Mobilerun device framework; LLM APIs live in the agents package."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mobilerun.core import (
        AndroidDevice as AndroidDevice,
    )
    from mobilerun.core import (
        CloudDevice as CloudDevice,
    )
    from mobilerun.core import (
        CloudFixedProxy as CloudFixedProxy,
    )
    from mobilerun.core import (
        CloudRotatingProxy as CloudRotatingProxy,
    )
    from mobilerun.core import (
        CloudSocks5Proxy as CloudSocks5Proxy,
    )
    from mobilerun.core import (
        Device as Device,
    )
    from mobilerun.core import (
        DevicePool as DevicePool,
    )
    from mobilerun.core import (
        IOSDevice as IOSDevice,
    )
    from mobilerun.core import (
        LocalProxy as LocalProxy,
    )
    from mobilerun.core import (
        RecordingDevice as RecordingDevice,
    )
    from mobilerun.core import (
        RemoteDevice as RemoteDevice,
    )
    from mobilerun.core import (
        StealthDevice as StealthDevice,
    )
    from mobilerun.core import (
        Workspace as Workspace,
    )

from mobilerun.config.user_config import (
    DeviceConfig as DeviceConfig,
)
from mobilerun.config.user_config import (
    LoggingConfig as LoggingConfig,
)
from mobilerun.config.user_config import (
    MobileConfig as MobileConfig,
)
from mobilerun.config.user_config import (
    TelemetryConfig as TelemetryConfig,
)
from mobilerun.config.user_config import (
    TracingConfig as TracingConfig,
)
from mobilerun.version import __version__ as __version__

_EAGER_EXPORTS = (
    "__version__",
    "DeviceConfig",
    "LoggingConfig",
    "MobileConfig",
    "TelemetryConfig",
    "TracingConfig",
)

_CORE_EXPORTS = (
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
)

__all__ = [*_EAGER_EXPORTS, *_CORE_EXPORTS]


def __getattr__(name: str) -> Any:
    if name in _CORE_EXPORTS:
        return getattr(import_module("mobilerun.core"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
