"""Device, Portal, and state errors."""

from mobilerun.errors.base import DeviceError


class DeviceAuthError(DeviceError):
    """An OAuth 2.0 error response from the token endpoint."""

    def __init__(self, code: str, description: str = "") -> None:
        self.code = code
        self.description = description
        super().__init__(f"{code}: {description}" if description else code)


class DeviceNotFound(DeviceError, RuntimeError):
    """The requested Android device could not be obtained."""


class DeviceNotReady(DeviceError, RuntimeError):
    pass


class DeviceOperationNotSupported(DeviceError, RuntimeError):
    """The selected device cannot perform the requested operation."""

    value_safe = True


class NoFocusedInput(DeviceError):
    """No editable text field is focused."""

    value_safe = True


class PortalApkNotFound(DeviceError, FileNotFoundError):
    """A requested Portal APK does not exist."""


class PortalInstallFailed(DeviceError, RuntimeError):
    """Installing the Portal APK failed."""

    def __init__(self, message: str, *, apk_path: str | None = None) -> None:
        self.apk_path = apk_path
        super().__init__(message)


class UnsupportedStateCapability(DeviceError, RuntimeError):
    def __init__(self, provider: object, capability: str) -> None:
        provider_name = type(provider).__name__
        super().__init__(
            f"State provider {provider_name} does not support capability '{capability}'"
        )


class StateFetchFailed(DeviceError):
    """Fetching device state failed after all retry attempts."""
