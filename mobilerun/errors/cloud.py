"""Cloud API, authentication, billing, and proxy errors."""

from collections.abc import Awaitable, Callable
from typing import ClassVar, Generic, Literal, TypeVar

from mobilerun.errors.base import CloudError, CredentialError
from mobilerun.models.checkout import Checkout

T = TypeVar("T")


class CloudAPIError(CloudError):
    """A cloud API request failed."""


class CloudAuthenticationError(CloudAPIError):
    """The cloud rejected the supplied authentication."""

    _hint: ClassVar[str] = "Run `mobilerun account login` again."


class CloudPermissionDenied(CloudAPIError):
    """The caller lacks permission for the operation."""

    _hint: ClassVar[str] = "Use credentials with permission for this resource."


class CloudResourceNotFound(CloudAPIError):
    """The requested cloud resource does not exist."""

    _hint: ClassVar[str] = "Check the resource id and account."


class CloudConflict(CloudAPIError):
    """The resource state conflicts with the operation."""

    _hint: ClassVar[str] = "Wait for the current operation to finish and retry."


class CloudValidationError(CloudAPIError):
    """The cloud rejected invalid parameters."""

    _hint: ClassVar[str] = "Check the supplied parameters."


class CloudRateLimited(CloudAPIError):
    """The caller exceeded a cloud rate limit."""

    _hint: ClassVar[str] = "Wait before retrying; reduce request frequency."


class CloudServiceUnavailable(CloudAPIError):
    """The cloud service is temporarily unavailable."""

    _hint: ClassVar[str] = "Retry later or contact support with the request id."


class CloudConnectionError(CloudAPIError):
    """The cloud could not be reached."""

    _hint: ClassVar[str] = "Check your network and cloud base URL."


class CloudTimeout(CloudAPIError, TimeoutError):
    """The cloud request timed out."""

    # Audited multiple-inheritance layout also declares a weakref descriptor.
    safe_rendering: ClassVar[bool] = True
    _hint: ClassVar[str] = "Check connectivity or increase the request timeout."


class BillingError(CloudAPIError, ValueError):
    """The billing policy does not authorize this operation."""


class IncludedCapacityUnavailable(BillingError):
    """Included inventory is exhausted; arbitrary country text is never rendered."""

    safe_rendering: ClassVar[bool] = True

    def __init__(
        self, resource: Literal["number", "mailbox"], *, country: object = None
    ) -> None:
        if resource == "number":
            super().__init__(
                operation="Create number",
                status=409,
                code="included_number_capacity",
                country=country,
            )
        elif resource == "mailbox":
            super().__init__(
                operation="Create mailbox", status=409, code="included_mailbox_capacity"
            )
        else:
            raise ValueError("Expected a number or mailbox resource")


class CloudAuthRequired(CloudAuthenticationError):
    """Mobilerun Cloud credentials are required to construct a cloud device."""


class CloudDeviceNotFound(CloudResourceNotFound):
    """The requested cloud device does not exist."""


class CloudTaskTimeout(CloudTimeout):
    """A Mobilerun Cloud task exceeded its configured wait timeout."""


class InvalidAPIKeyError(CredentialError):
    """The cloud API rejected the supplied API key."""


class InvalidCloudDeviceId(CloudValidationError):
    """The supplied cloud device ID is invalid."""

    _hint: ClassVar[str] = "Pass a valid cloud device UUID."


class PaymentRequired(BillingError, Generic[T]):
    safe_rendering: ClassVar[bool] = True

    def __init__(
        self, checkout: Checkout, resume: Callable[[float], Awaitable[T]]
    ) -> None:
        self.checkout = checkout
        self._resume = resume
        self.resource = checkout.resource
        self.resource_id = checkout.resource_id
        self.payment_url = checkout.url
        self.checkout_expires_at = checkout.expires_at
        super().__init__(
            code="payment_required",
            resource_id=checkout.resource_id,
        )

    async def wait_until_paid(self, timeout: float = 600) -> T:
        """Wait for activation without cancelling the pending resource on timeout."""
        return await self._resume(timeout)


class PurchaseInProgress(CloudConflict):
    """A purchase already exists; its checkout URL may be sensitive."""

    safe_rendering: ClassVar[bool] = True

    def __init__(self, resource_id: str, checkout: Checkout | None) -> None:
        self.resource_id = resource_id
        self.checkout = checkout
        super().__init__(
            code="purchase_in_progress",
            resource_id=resource_id,
        )


class ProxyNotActive(CloudError):
    """The cloud phone has no active proxy after a switch."""
