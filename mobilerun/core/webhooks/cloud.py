"""Manage organization webhook endpoints using the configured cloud account."""

from collections.abc import Sequence
from uuid import UUID

from pydantic import HttpUrl

from mobilerun.clients.cloud import webhooks
from mobilerun.config.settings import get_settings
from mobilerun.errors.cloud import CloudValidationError
from mobilerun.models.billing import Billing
from mobilerun.models.webhooks import WebhookEndpoint

DEFAULT_WEBHOOK_EVENT = "workflow.step.emit"
_WEBHOOK_URL_MAX_LENGTH = 2048


def _validate_webhook_url_length(value: str) -> None:
    if len(value) > _WEBHOOK_URL_MAX_LENGTH:
        raise CloudValidationError(
            f"Webhook URL exceeds {_WEBHOOK_URL_MAX_LENGTH} characters."
        )


def validate_webhook_url(value: str) -> str:
    """Return the canonical HTTPS registration URL or raise a value-safe error."""
    if not isinstance(value, str):
        raise CloudValidationError("Webhook URL must be a string.")
    _validate_webhook_url_length(value)
    if any(
        c.isspace() or ord(c) < 32 or 127 <= ord(c) <= 159 or c == "\\" for c in value
    ):
        raise CloudValidationError(
            "Remove whitespace, control characters and backslashes from the URL."
        )
    if not value.lower().startswith("https://"):
        raise CloudValidationError("Webhook URL must start with https://.")
    parsed = None
    if value[8:9] not in ("", "/", "?", "#"):
        try:
            parsed = HttpUrl(value)
        except ValueError:
            pass  # Do not chain the parser error: it contains the caller's URL.
    if parsed is None:
        raise CloudValidationError(
            "Webhook URL needs a valid hostname and port (if specified)."
        )
    if parsed.username is not None or parsed.password is not None:
        raise CloudValidationError(
            "Remove the username or password from the webhook URL."
        )
    if parsed.fragment is not None:
        raise CloudValidationError("Remove the fragment (#…) from the webhook URL.")
    # webhooks-api's normalizeUrl removes trailing slashes only from the path.
    base, separator, query = str(parsed).partition("?")
    if parsed.path and parsed.path != "/":
        base = base.rstrip("/")
        if not parsed.path.rstrip("/"):
            base += "/"
    canonical = base + separator + query
    _validate_webhook_url_length(canonical)
    return canonical


def _id(value: str) -> str:
    if isinstance(value, str):
        try:
            return str(UUID(value))
        except ValueError:
            pass
    raise CloudValidationError("Expected a webhook endpoint UUID")


class CloudWebhooks:
    """Each operation closes its own client. Finding an endpoint does not transfer ownership."""

    def __init__(self, *, billing: Billing | str | None = None) -> None:
        self._billing = (
            Billing(billing)
            if billing is not None
            else get_settings().MOBILERUN_BILLING
        )

    async def register(
        self,
        url: str,
        *,
        event_types: Sequence[str],
        idempotency_key: str | None = None,
    ) -> WebhookEndpoint:
        """Register a new HTTPS destination. An explicit empty filter subscribes to all events."""
        if isinstance(event_types, (str, bytes)):
            raise CloudValidationError("Expected a sequence of webhook event types")
        return await webhooks.register_webhook(
            validate_webhook_url(url),
            event_types,
            idempotency_key,
            billing=self._billing,
        )

    async def get(self, endpoint_id: str) -> WebhookEndpoint:
        """Get endpoint metadata by UUID."""
        return await webhooks.get_webhook(_id(endpoint_id), billing=self._billing)

    async def list(self) -> list[WebhookEndpoint]:
        """List all endpoint kinds and states visible to the account."""
        return await webhooks.list_webhooks(
            get_settings().MOBILERUN_WEBHOOK_PAGE_SIZE, billing=self._billing
        )

    async def ping(self, endpoint_id: str) -> tuple[bool, int | None]:
        """Send one service-signed test. Return success and receiver HTTP status."""
        return await webhooks.ping_webhook(
            _id(endpoint_id),
            get_settings().MOBILERUN_WEBHOOK_PING_TIMEOUT,
            billing=self._billing,
        )

    async def find_by_url(self, url: str) -> WebhookEndpoint | None:
        """Find a non-deleted HTTP endpoint by canonical URL, without changing its state."""
        target = validate_webhook_url(url)
        return next(
            (
                e
                for e in await self.list()
                if e.kind == "http"
                and e.url
                and e.state != "DELETED"
                and e.url.get_secret_value() == target
            ),
            None,
        )

    async def delete(self, endpoint_id: str) -> None:
        """Delete by UUID. Callers must decide whether they own or may delete it."""
        await webhooks.delete_webhook(_id(endpoint_id), billing=self._billing)
