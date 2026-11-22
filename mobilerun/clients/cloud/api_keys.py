"""Cloud API key validation."""

from mobilerun_sdk import APIError

from mobilerun.clients.cloud.billing import requires
from mobilerun.clients.cloud.client import cloud_client
from mobilerun.config.settings import get_settings
from mobilerun.errors import InvalidAPIKeyError
from mobilerun.models.billing import Billing
from mobilerun.clients.cloud.errors import cloud_error


@requires(Billing.NO_COMMERCE)
async def validate_api_key(api_key: str, *, billing: Billing) -> None:
    """Validate access to the devices endpoint."""
    if not api_key:
        raise InvalidAPIKeyError("Invalid API key")
    settings = get_settings()
    base = settings.MOBILERUN_CLOUD_BASE_URL.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    try:
        async with cloud_client(
            billing=billing, api_key=api_key, base_url=base
        ) as client:
            await client.devices.list(
                page_size=1, timeout=settings.MOBILERUN_CLOUD_API_TIMEOUT
            )
    except APIError as exc:
        raise cloud_error(exc, "Validate cloud API key") from exc
