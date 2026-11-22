"""Operator, extension, and expectation errors."""

from mobilerun.errors.base import CredentialError, DeviceError, OperatorError
from mobilerun.models.results import (
    ExecutionResult,
    ExpectationResult,
    ExpectationVerdict,
)


class ExtensionConflict(DeviceError):
    """A name or exclusive binding is already occupied."""

    value_safe = True


class UnmanagedProxyConflict(ExtensionConflict):
    """An active unmanaged cloud proxy requires explicit replacement."""


class ExtensionNotSupported(DeviceError):
    """Declared device facts do not satisfy an extension."""

    value_safe = True


class ExtensionNotBound(DeviceError):
    """The extension is detached, closing, or uncertain."""

    value_safe = True


class ExtensionActionError(DeviceError):
    """An action failed without exposing arguments or provider responses."""

    value_safe = True

    def __init__(self, message: str, *, cause_type: str = "") -> None:
        super().__init__(message)
        self.cause_type = cause_type


class ExtensionClosed(DeviceError):
    """The extension is shutting down or has already released its resource."""

    value_safe = True


class OperatorNotAvailable(OperatorError, NotImplementedError):
    """No operator is available to carry out an instruction."""


class OperatorNotSupported(OperatorError):
    """An operator cannot run on the selected device."""


class ExpectationFailed(DeviceError):
    def __init__(self, result: ExpectationResult) -> None:
        self.result = result
        self.condition = result.condition
        self.reason = result.reason
        super().__init__(self.reason or f"Expectation failed: {self.condition}")


class ExpectationInconclusive(DeviceError):
    def __init__(
        self, condition: str, task: ExecutionResult[ExpectationVerdict]
    ) -> None:
        self.condition = condition
        self.task = task
        super().__init__(
            "Could not determine whether the expectation holds: no structured verdict"
        )


class PlaceholderError(CredentialError, ValueError):
    pass
