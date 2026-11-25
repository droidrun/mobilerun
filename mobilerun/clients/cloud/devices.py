"""Functional cloud device SDK boundaries."""

from contextlib import nullcontext
from datetime import datetime
from typing import Literal, TypedDict, Unpack, cast

import httpx
from mobilerun_sdk import APIError, AsyncMobilerun
from mobilerun_sdk._types import Body, Headers, NotGiven, Query, SequenceNotStr
from mobilerun_sdk.types import (
    DeviceCreateResponse,
    DeviceRetrieveCapabilitiesResponse,
    DeviceRetrieveResponse,
    DeviceSetNameResponse,
    DeviceWaitReadyResponse,
)
from mobilerun_sdk.types.device_create_params import DeviceCreateParams, Proxy
from mobilerun_sdk.types.device_list_response import Item as DeviceItem
from mobilerun_sdk.types.devices import (
    LanguageGetResponse,
    NumberBindResponse,
    NumberRetrieveResponse,
    NumberUnbindResponse,
    TimezoneGetResponse,
)
from mobilerun_sdk.types.shared.location import Location as DeviceLocation
from mobilerun_sdk.types.shared_params import DeviceCarrier, DeviceIdentifiers, Location

from mobilerun.clients.cloud.billing import (
    _billing_preference,
    _validate_create_params,
    requires,
)
from mobilerun.clients.cloud.client import cloud_client
from mobilerun.clients.cloud.errors import cloud_error
from mobilerun.errors.cloud import BillingError
from mobilerun.models.billing import Billing, Plan
from mobilerun.utils.redaction import secret_scope

DeviceCreateOptions = TypedDict(
    "DeviceCreateOptions",
    {
        "query_country": str,
        "device_type": Literal["android_cloud_phone", "ios_phone", "device_slot"],
        "profile_id": str,
        "template": str,
        "android_version": int,
        "apps": SequenceNotStr[str] | None,
        "files": SequenceNotStr[str] | None,
        "name": str,
        "locale": str,
        "timezone": str,
        "body_country": str,
        "carrier": DeviceCarrier,
        "identifiers": DeviceIdentifiers,
        "location": Location,
        "extra_headers": Headers | None,
        "extra_query": Query | None,
        "extra_body": Body | None,
        "timeout": float | httpx.Timeout | None | NotGiven,
        "idempotency_key": str | None,
    },
    total=False,
)

RequestOptions = TypedDict(
    "RequestOptions",
    {
        "extra_headers": Headers | None,
        "extra_query": Query | None,
        "extra_body": Body | None,
        "timeout": float | httpx.Timeout | None | NotGiven,
        "idempotency_key": str | None,
    },
    total=False,
)
ReadOptions = TypedDict(
    "ReadOptions",
    {
        "extra_headers": Headers | None,
        "extra_query": Query | None,
        "extra_body": Body | None,
        "timeout": float | httpx.Timeout | None | NotGiven,
    },
    total=False,
)
TerminateOptions = TypedDict(
    "TerminateOptions",
    {
        "previous_device_id": str,
        "terminate_at": str | datetime,
        "extra_headers": Headers | None,
        "extra_query": Query | None,
        "extra_body": Body | None,
        "timeout": float | httpx.Timeout | None | NotGiven,
        "idempotency_key": str | None,
    },
    total=False,
)
ResetOptions = TypedDict(
    "ResetOptions",
    {
        "mode": Literal["full", "identity"],
        "preserve": list[Literal["proxy", "location", "locale", "timezone"]] | None,
        "extra_headers": Headers | None,
        "extra_query": Query | None,
        "extra_body": Body | None,
        "timeout": float | httpx.Timeout | None | NotGiven,
        "idempotency_key": str | None,
    },
    total=False,
)


@requires(Billing.INCLUDED)
async def create(
    *,
    billing: Billing,
    plan: Plan | str | None = None,
    proxy: Proxy | None = None,
    client: AsyncMobilerun | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **params: Unpack[DeviceCreateOptions],
) -> DeviceCreateResponse:
    """Create a cloud phone, optionally with a Connect proxy."""
    await _validate_create_params(params, allow_proxy=True)
    if proxy is None:
        raise ValueError("A proxy is required to create a cloud phone")
    if proxy is not None:
        connect = proxy.get("connect")
        socks = proxy.get("socks5")
        valid_connect = isinstance(connect, dict) and bool(
            connect.get("id") or connect.get("country")
        )
        valid_socks = isinstance(socks, dict) and bool(
            socks.get("host") and socks.get("port")
        )
        if not (valid_connect or valid_socks):
            raise ValueError("Pass a valid Connect or SOCKS5 proxy, or omit proxy")
    _create_params = cast(DeviceCreateParams, params)
    if proxy is not None:
        _create_params["proxy"] = proxy
    _create_params["billing"] = cast(
        Literal["auto", "subscription", "minute"],
        await _billing_preference("cloud phones", billing, plan=plan),
    )
    async with (
        nullcontext(client)
        if client is not None
        else cloud_client(api_key, base_url, billing=billing)
    ) as sdk:
        socks = proxy.get("socks5") or {}
        with secret_scope(socks.get("password", "")):
            try:
                return await sdk.devices.create(**_create_params)
            except APIError as exc:
                raise cloud_error(exc, "Create cloud phone", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def list_devices(
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[DeviceItem]:
    """List every device page."""
    async with (
        nullcontext(client)
        if client is not None
        else cloud_client(api_key, base_url, billing=billing)
    ) as sdk:
        items, page = [], 1
        try:
            while True:
                response = await sdk.devices.list(page=page, page_size=100)
                items.extend(cast(list[DeviceItem], response.items))
                if not response.pagination.has_next:
                    return items
                page += 1
        except APIError as exc:
            raise cloud_error(exc, "List cloud phones") from exc


@requires(Billing.NO_COMMERCE)
async def capabilities(
    device_id: str,
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> DeviceRetrieveCapabilitiesResponse:
    """Retrieve a cloud phone's supported capabilities."""
    async with (
        nullcontext(client)
        if client is not None
        else cloud_client(api_key, base_url, billing=billing)
    ) as sdk:
        try:
            return await sdk.devices.retrieve_capabilities(device_id)
        except APIError as exc:
            raise cloud_error(
                exc, "Retrieve cloud phone capabilities", device=True
            ) from exc


@requires(Billing.NO_COMMERCE)
async def set_name(
    device_id: str, name: str, *, billing: Billing, client: AsyncMobilerun
) -> DeviceSetNameResponse:
    """Set a cloud phone's display name and return its updated info."""
    try:
        return await client.devices.set_name(device_id, name=name)
    except APIError as exc:
        raise cloud_error(exc, "Rename cloud phone", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def retrieve(
    device_id: str, *, billing: Billing, client: AsyncMobilerun
) -> DeviceRetrieveResponse:
    """Retrieve a cloud phone."""
    try:
        return await client.devices.retrieve(device_id)
    except APIError as exc:
        raise cloud_error(exc, "Retrieve cloud phone", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def wait_ready(
    device_id: str,
    *,
    billing: Billing,
    client: AsyncMobilerun,
    **params: Unpack[ReadOptions],
) -> DeviceWaitReadyResponse:
    """Wait for phone readiness using SDK polling."""
    try:
        return await client.devices.wait_ready(device_id, **params)
    except APIError as exc:
        raise cloud_error(exc, "Wait for cloud phone", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def terminate(
    device_id: str,
    *,
    billing: Billing,
    client: AsyncMobilerun,
    **params: Unpack[TerminateOptions],
) -> None:
    """Terminate a cloud phone."""
    try:
        return await client.devices.terminate(device_id, **params)
    except APIError as exc:
        raise cloud_error(exc, "Terminate cloud phone", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def resume(
    device_id: str,
    *,
    billing: Billing,
    client: AsyncMobilerun,
    **params: Unpack[RequestOptions],
) -> None:
    """Resume an existing phone without starting minute billing unless authorized."""
    if billing is not Billing.PAID:
        info = await retrieve(device_id, billing=billing, client=client)
        if info.billing_strategy != Plan.SUBSCRIPTION:
            raise BillingError(
                operation="resume", code="invalid_billing", required_billing="paid"
            )
    try:
        return await client.devices.resume(device_id, **params)
    except APIError as exc:
        raise cloud_error(exc, "Resume cloud phone", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def stop(
    device_id: str,
    *,
    billing: Billing,
    client: AsyncMobilerun,
    **params: Unpack[RequestOptions],
) -> None:
    """Stop a cloud phone."""
    try:
        return await client.devices.stop(device_id, **params)
    except APIError as exc:
        raise cloud_error(exc, "Stop cloud phone", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def reboot(
    device_id: str,
    *,
    billing: Billing,
    client: AsyncMobilerun,
    **params: Unpack[RequestOptions],
) -> None:
    """Reboot a cloud phone."""
    try:
        return await client.devices.reboot(device_id, **params)
    except APIError as exc:
        raise cloud_error(exc, "Reboot cloud phone", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def reset(
    device_id: str,
    *,
    billing: Billing,
    client: AsyncMobilerun,
    **params: Unpack[ResetOptions],
) -> None:
    """Reset a cloud phone."""
    try:
        return await client.devices.reset(device_id, **params)
    except APIError as exc:
        raise cloud_error(exc, "Reset cloud phone", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def get_location(
    device_id: str, *, billing: Billing, client: AsyncMobilerun
) -> DeviceLocation | None:
    """Retrieve the simulated GPS location of a cloud phone."""
    try:
        return await client.devices.location.get(device_id)
    except APIError as exc:
        raise cloud_error(exc, "Retrieve cloud phone location", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def number(
    device_id: str, *, billing: Billing, client: AsyncMobilerun
) -> NumberRetrieveResponse:
    """Retrieve the bound number."""
    try:
        return await client.devices.number.retrieve(device_id)
    except APIError as exc:
        raise cloud_error(exc, "Retrieve phone number binding", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def get_language(
    device_id: str, *, billing: Billing, client: AsyncMobilerun
) -> LanguageGetResponse:
    """Retrieve the phone's BCP-47 locale."""
    try:
        return await client.devices.language.get(device_id)
    except APIError as exc:
        raise cloud_error(exc, "Retrieve cloud phone language", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def get_timezone(
    device_id: str, *, billing: Billing, client: AsyncMobilerun
) -> TimezoneGetResponse:
    """Retrieve the phone's IANA timezone."""
    try:
        return await client.devices.timezone.get(device_id)
    except APIError as exc:
        raise cloud_error(exc, "Retrieve cloud phone timezone", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def bind_number(
    device_id: str, number_id: str, *, billing: Billing, client: AsyncMobilerun
) -> NumberBindResponse:
    """Bind an existing number."""
    try:
        return await client.devices.number.bind(device_id, number_id=number_id)
    except APIError as exc:
        raise cloud_error(exc, "Bind phone number", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def unbind_number(
    device_id: str, *, billing: Billing, client: AsyncMobilerun
) -> NumberUnbindResponse:
    """Unbind a number."""
    try:
        return await client.devices.number.unbind(device_id)
    except APIError as exc:
        raise cloud_error(exc, "Unbind phone number", device=True) from exc
