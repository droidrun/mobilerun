"""Shared SDK client factory."""

from collections.abc import Awaitable
from typing import TypeVar

import httpx
from mobilerun_sdk import APIError, AsyncMobilerun

from mobilerun.clients.cloud.billing import requires
from mobilerun.clients.local.login_store import resolve_cloud_api_key
from mobilerun.config.settings import get_settings
from mobilerun.errors.cloud import CloudAuthRequired
from mobilerun.models.billing import Billing
from mobilerun.clients.cloud.errors import cloud_error

T = TypeVar("T")


async def _device_request(request: Awaitable[T]) -> T:
    """Translate transport failures without leaking the SDK into core."""
    try:
        return await request
    except (APIError, httpx.HTTPError) as exc:
        raise cloud_error(exc, "Cloud phone request", device=True) from exc


@requires(Billing.NO_COMMERCE)
def cloud_client(
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    billing: Billing | str,
) -> AsyncMobilerun:
    """Construct a client; callers own its lifetime."""
    key = api_key if api_key is not None else resolve_cloud_api_key()
    if not key:
        raise CloudAuthRequired(code="credentials_required")
    return AsyncMobilerun(
        api_key=key, base_url=base_url or get_settings().MOBILERUN_CLOUD_BASE_URL
    )
