"""Errors for sandbox operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mobilerun.errors.base import MobilerunError
from mobilerun.errors.cloud import CloudAPIError

if TYPE_CHECKING:
    from .models import OwnershipReceipt


class SandboxError(MobilerunError):
    def __init__(self, message: str, *, missing_deployment: bool = False) -> None:
        super().__init__(message)
        self.missing_deployment = missing_deployment

    def __str__(self) -> str:
        message = super().__str__()
        if self.missing_deployment:
            return (
                f"{message}; use `mobilerun sandbox deploy FILE` to deploy the function"
            )
        return message


class FlowCapabilityError(CloudAPIError):
    """The workflow cannot be represented safely by the installed SDK."""

    safe_rendering: ClassVar[bool] = True


class FlowStagingError(CloudAPIError):
    """A staging failure with known resource IDs for recovery."""

    safe_rendering: ClassVar[bool] = True

    def __init__(
        self, trigger_id: str | None, action_ids: list[str], flow_id: str | None
    ) -> None:
        super().__init__(operation="Stage flow", code="request_failed")
        self.trigger_id = trigger_id
        self.action_ids = tuple(action_ids)
        self.flow_id = flow_id


class SandboxDeclarationError(SandboxError):
    """Invalid sandbox declaration."""


class SandboxPackagingError(SandboxError):
    """Sandbox source could not be packaged."""


class SandboxDeploymentError(SandboxError):
    """Deployment failed, possibly after creating owned resources."""

    def __init__(
        self,
        message: str,
        *,
        receipt: OwnershipReceipt | None = None,
        missing_deployment: bool = False,
    ) -> None:
        super().__init__(message, missing_deployment=missing_deployment)
        self.receipt = receipt


class SandboxDeploymentConflict(SandboxDeploymentError):
    """Observed resources conflict with recorded ownership."""


class SandboxRunError(SandboxError):
    """A sandbox invocation or execution failed."""

    def __init__(
        self,
        message: str,
        *,
        invocation_id: str | None = None,
        execution_id: str | None = None,
        missing_deployment: bool = False,
    ) -> None:
        super().__init__(message, missing_deployment=missing_deployment)
        self.invocation_id = invocation_id
        self.execution_id = execution_id


class SandboxRunTimeout(SandboxRunError, TimeoutError):
    """A sandbox run exceeded its deadline."""


class SandboxConfirmationRequired(SandboxError):
    """A mutation needs explicit confirmation."""
