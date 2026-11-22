"""Functional number SDK boundaries."""

from contextlib import nullcontext
from datetime import datetime
from typing import Literal, TypedDict, Unpack, cast

import httpx
from mobilerun_sdk import APIError, APIStatusError, AsyncMobilerun
from mobilerun_sdk._types import Body, Headers, NotGiven, Query
from mobilerun_sdk.types import (
    NumberCapacityResponse,
    NumberCountriesResponse,
    NumberCreateResponse,
    NumberDeleteResponse,
    NumberRetrieveResponse,
)
from mobilerun_sdk.types.number_create_params import NumberCreateParams
from mobilerun_sdk.types.number_list_response import Item as NumberItem
from mobilerun_sdk.types.numbers import MessageSendResponse
from mobilerun_sdk.types.numbers.message_list_response import Item as MessageItem

from mobilerun.clients.cloud.billing import (
    _billing_preference,
    _validate_create_params,
    requires,
)
from mobilerun.clients.cloud.checkout import CheckoutMode, checkout_callback
from mobilerun.clients.cloud.client import cloud_client
from mobilerun.clients.cloud.errors import cloud_error
from mobilerun.clients.cloud.payments import (
    CheckoutCallback,
    PaymentAdapter,
    PaymentSnapshot,
    wait_for_payment,
    wait_until_active,
)
from mobilerun.errors import IncludedCapacityUnavailable, PurchaseInProgress
from mobilerun.models.billing import Billing, ResourcePlan
from mobilerun.models.checkout import Checkout
from mobilerun.utils.timestamps import to_api_timestamp

NumberCreateOptions = TypedDict(
    "NumberCreateOptions",
    {
        "country": str,
        "label": str | None,
        "extra_headers": Headers | None,
        "extra_query": Query | None,
        "extra_body": Body | None,
        "timeout": float | httpx.Timeout | None | NotGiven,
        "idempotency_key": str | None,
    },
    total=False,
)


MessageOptions = TypedDict(
    "MessageOptions",
    {
        "direction": Literal["all", "inbound", "outbound"],
        "since": str | datetime,
        "extra_headers": Headers | None,
        "extra_query": Query | None,
        "extra_body": Body | None,
        "timeout": float | httpx.Timeout | None | NotGiven,
    },
    total=False,
)


@requires(Billing.INCLUDED)
async def create(
    *,
    billing: Billing,
    plan: ResourcePlan | str | None = None,
    client: AsyncMobilerun | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    keep: bool = False,
    on_checkout: CheckoutCallback | None = None,
    checkout: CheckoutMode = "error",
    payment_timeout: float = 600,
    **params: Unpack[NumberCreateOptions],
) -> NumberCreateResponse:
    """Provision a phone number."""
    on_checkout = checkout_callback(checkout, on_checkout)
    await _validate_create_params(params)
    unknown = params.keys() - NumberCreateOptions.__annotations__.keys()
    if unknown:
        raise TypeError(f"Unexpected create options: {', '.join(sorted(unknown))}")
    _create_params = cast(NumberCreateParams, params)
    _create_params["billing_preference"] = cast(
        Literal["included", "included_only", "rent"],
        await _billing_preference("numbers", billing, plan=plan),
    )
    async with (
        nullcontext(client)
        if client is not None
        else cloud_client(api_key, base_url, billing=billing)
    ) as sdk:
        try:
            response = await sdk.numbers.create(**_create_params)

            async def payment_state() -> PaymentSnapshot:
                data = (await sdk.numbers.retrieve(response.data.number_id)).data
                return PaymentSnapshot(data.state, data.checkout_url)

            async def resume(timeout: float) -> NumberRetrieveResponse:
                async def fetch() -> PaymentSnapshot:
                    data = (
                        await retrieve(
                            response.data.number_id,
                            billing=Billing.NO_COMMERCE,
                            client=client,
                            api_key=api_key,
                            base_url=base_url,
                        )
                    ).data
                    return PaymentSnapshot(data.state, data.checkout_url)

                adapter = PaymentAdapter(
                    "phone number",
                    response.data.number_id,
                    fetch,
                    lambda: sdk.numbers.delete(response.data.number_id),
                )
                await wait_until_active(adapter, timeout=timeout)
                return await retrieve(
                    response.data.number_id,
                    billing=Billing.NO_COMMERCE,
                    client=client,
                    api_key=api_key,
                    base_url=base_url,
                )

            await wait_for_payment(
                PaymentAdapter(
                    "phone number",
                    response.data.number_id,
                    payment_state,
                    lambda: sdk.numbers.delete(response.data.number_id),
                ),
                PaymentSnapshot(response.data.state, response.data.checkout_url),
                resume=resume,
                on_checkout=on_checkout,
                timeout=payment_timeout,
            )
            return response
        except APIError as exc:
            body = getattr(exc, "body", None)
            if (
                isinstance(exc, APIStatusError)
                and exc.status_code == 409
                and isinstance(body, dict)
                and body.get("code") == "included_capacity_unavailable"
            ):
                raise IncludedCapacityUnavailable(
                    "number", country=params.get("country")
                ) from None
            if (
                isinstance(exc, APIStatusError)
                and exc.status_code == 409
                and isinstance(body, dict)
                and body.get("code") == "purchase_in_progress"
            ):
                pending = []
                page = 1
                try:
                    while True:
                        result = await sdk.numbers.list(page=page, page_size=100)
                        pending.extend(
                            item
                            for item in result.items
                            if item.state in ("awaiting_payment", "provisioning")
                        )
                        if not result.pagination.has_next:
                            break
                        page += 1
                except APIError:
                    raise cloud_error(exc, "Create number") from None
                if len(pending) == 1:
                    item = pending[0]
                    pending_checkout = (
                        Checkout(item.checkout_url, None, "phone number", item.id)
                        if item.checkout_url
                        else None
                    )
                    raise PurchaseInProgress(item.id, pending_checkout) from None
            raise cloud_error(exc, "Create number") from exc


@requires(Billing.NO_COMMERCE)
async def list_numbers(
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[NumberItem]:
    """List every number page."""
    async with (
        nullcontext(client)
        if client is not None
        else cloud_client(api_key, base_url, billing=billing)
    ) as sdk:
        items, page = [], 1
        try:
            while True:
                response = await sdk.numbers.list(page=page, page_size=100)
                items.extend(response.items)
                if not response.pagination.has_next:
                    return items
                page += 1
        except APIError as exc:
            raise cloud_error(exc, "List numbers") from exc


@requires(Billing.NO_COMMERCE)
async def retrieve(
    number_id: str,
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> NumberRetrieveResponse:
    """Retrieve a number."""
    async with (
        nullcontext(client)
        if client is not None
        else cloud_client(api_key, base_url, billing=billing)
    ) as sdk:
        try:
            return await sdk.numbers.retrieve(number_id)
        except APIError as exc:
            raise cloud_error(exc, "Retrieve number") from exc


@requires(Billing.NO_COMMERCE)
async def delete(
    number_id: str,
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> NumberDeleteResponse:
    """Release a number."""
    async with (
        nullcontext(client)
        if client is not None
        else cloud_client(api_key, base_url, billing=billing)
    ) as sdk:
        try:
            return await sdk.numbers.delete(number_id)
        except APIError as exc:
            raise cloud_error(exc, "Release number") from exc


@requires(Billing.NO_COMMERCE)
async def countries(
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> NumberCountriesResponse:
    """List number countries."""
    async with (
        nullcontext(client)
        if client is not None
        else cloud_client(api_key, base_url, billing=billing)
    ) as sdk:
        try:
            return await sdk.numbers.countries()
        except APIError as exc:
            raise cloud_error(exc, "List number countries") from exc


@requires(Billing.NO_COMMERCE)
async def capacity(
    country: str,
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> NumberCapacityResponse:
    """Retrieve number capacity."""
    async with (
        nullcontext(client)
        if client is not None
        else cloud_client(api_key, base_url, billing=billing)
    ) as sdk:
        try:
            return await sdk.numbers.capacity(country=country)
        except APIError as exc:
            raise cloud_error(exc, "Retrieve number capacity") from exc


@requires(Billing.NO_COMMERCE)
async def messages(
    number_id: str,
    *,
    billing: Billing,
    after: str | datetime | None = None,
    client: AsyncMobilerun | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **params: Unpack[MessageOptions],
) -> list[MessageItem]:
    """List every SMS message page."""
    if after is not None:
        params["since"] = (
            to_api_timestamp(after) if isinstance(after, datetime) else after
        )
    async with (
        nullcontext(client)
        if client is not None
        else cloud_client(api_key, base_url, billing=billing)
    ) as sdk:
        items, page = [], 1
        try:
            while True:
                response = await sdk.numbers.messages.list(
                    number_id, page=page, page_size=100, **params
                )
                items.extend(response.items)
                if not response.pagination.has_next:
                    return items
                page += 1
        except APIError as exc:
            raise cloud_error(exc, "List SMS messages") from exc


@requires(Billing.NO_COMMERCE)
async def send(
    number_id: str,
    to: str,
    text: str,
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> MessageSendResponse:
    """Send an SMS using an existing number."""
    async with (
        nullcontext(client)
        if client is not None
        else cloud_client(api_key, base_url, billing=billing)
    ) as sdk:
        try:
            return await sdk.numbers.messages.send(number_id, to=to, body=text)
        except APIError as exc:
            raise cloud_error(exc, "Send SMS") from exc
