"""Functional mailbox SDK boundaries."""

from contextlib import nullcontext
from typing import Literal, Unpack, cast

from mobilerun_sdk import APIError, APIStatusError, AsyncMobilerun, NotFoundError
from mobilerun_sdk.types.mailbox_capacity_response import MailboxCapacityResponse
from mobilerun_sdk.types.mailbox_create_params import MailboxCreateParams
from mobilerun_sdk.types.mailbox_create_response import MailboxCreateResponse
from mobilerun_sdk.types.mailbox_delete_response import MailboxDeleteResponse
from mobilerun_sdk.types.mailbox_list_response import Item
from mobilerun_sdk.types.mailbox_otp_params import MailboxOtpParams
from mobilerun_sdk.types.mailbox_otp_response import MailboxOtpResponse
from mobilerun_sdk.types.mailbox_retrieve_response import MailboxRetrieveResponse
from mobilerun_sdk.types.mailboxes.message_list_response import Item as MessageItem

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
    wait_for_payment,
    wait_until_active,
)
from mobilerun.errors import IncludedCapacityUnavailable
from mobilerun.models.billing import Billing, ResourcePlan
from mobilerun.models.checkout import PaymentSnapshot


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
    **params: Unpack[MailboxCreateParams],
) -> MailboxCreateResponse:
    """Provision a mailbox."""
    on_checkout = checkout_callback(checkout, on_checkout)
    await _validate_create_params(params)
    unknown = params.keys() - MailboxCreateParams.__annotations__.keys()
    if unknown:
        raise TypeError(f"Unexpected create options: {', '.join(sorted(unknown))}")
    params["billing_preference"] = cast(
        Literal["included", "included_only", "rent"],
        await _billing_preference("mailboxes", billing, plan=plan),
    )
    async with (
        nullcontext(client)
        if client is not None
        else cloud_client(api_key, base_url, billing=billing)
    ) as sdk:
        try:
            response = await sdk.mailboxes.create(**params)

            if response.data.status == "awaiting_payment":

                async def payment_state() -> PaymentSnapshot:
                    data = (await sdk.mailboxes.retrieve(response.data.id)).data
                    return PaymentSnapshot(
                        data.status, data.checkout_url, data.checkout_expires_at
                    )

                async def resume(timeout: float) -> MailboxRetrieveResponse:
                    async def fetch() -> PaymentSnapshot:
                        data = (
                            await retrieve(
                                response.data.id,
                                billing=Billing.NO_COMMERCE,
                                client=client,
                                api_key=api_key,
                                base_url=base_url,
                            )
                        ).data
                        return PaymentSnapshot(
                            data.status, data.checkout_url, data.checkout_expires_at
                        )

                    adapter = PaymentAdapter(
                        "mailbox",
                        response.data.id,
                        fetch,
                        lambda: sdk.mailboxes.delete(response.data.id),
                    )
                    await wait_until_active(adapter, timeout=timeout)
                    return await retrieve(
                        response.data.id,
                        billing=Billing.NO_COMMERCE,
                        client=client,
                        api_key=api_key,
                        base_url=base_url,
                    )

                await wait_for_payment(
                    PaymentAdapter(
                        "mailbox",
                        response.data.id,
                        payment_state,
                        lambda: sdk.mailboxes.delete(response.data.id),
                    ),
                    PaymentSnapshot(
                        response.data.status,
                        response.data.checkout_url,
                        response.data.checkout_expires_at,
                    ),
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
                raise IncludedCapacityUnavailable("mailbox") from None
            raise cloud_error(exc, "Create mailbox") from exc


@requires(Billing.NO_COMMERCE)
async def list_mailboxes(
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[Item]:
    """List every mailbox page."""
    async with (
        nullcontext(client)
        if client is not None
        else cloud_client(api_key, base_url, billing=billing)
    ) as sdk:
        items, page = [], 1
        try:
            while True:
                response = await sdk.mailboxes.list(page=page, page_size=100)
                items.extend(response.items)
                if not response.pagination.has_next:
                    return items
                page += 1
        except APIError as exc:
            raise cloud_error(exc, "List mailboxes") from exc


@requires(Billing.NO_COMMERCE)
async def retrieve(
    mailbox_id: str,
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> MailboxRetrieveResponse:
    """Retrieve a mailbox."""
    async with (
        nullcontext(client)
        if client is not None
        else cloud_client(api_key, base_url, billing=billing)
    ) as sdk:
        try:
            return await sdk.mailboxes.retrieve(mailbox_id)
        except APIError as exc:
            raise cloud_error(exc, "Retrieve mailbox") from exc


@requires(Billing.NO_COMMERCE)
async def delete(
    mailbox_id: str,
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> MailboxDeleteResponse:
    """Release a mailbox."""
    async with (
        nullcontext(client)
        if client is not None
        else cloud_client(api_key, base_url, billing=billing)
    ) as sdk:
        try:
            return await sdk.mailboxes.delete(mailbox_id)
        except APIError as exc:
            raise cloud_error(exc, "Release mailbox") from exc


@requires(Billing.NO_COMMERCE)
async def capacity(
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> MailboxCapacityResponse:
    """Retrieve mailbox capacity."""
    async with (
        nullcontext(client)
        if client is not None
        else cloud_client(api_key, base_url, billing=billing)
    ) as sdk:
        try:
            return await sdk.mailboxes.capacity()
        except APIError as exc:
            raise cloud_error(exc, "Retrieve mailbox capacity") from exc


@requires(Billing.NO_COMMERCE)
async def otp(
    mailbox_id: str,
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **params: Unpack[MailboxOtpParams],
) -> MailboxOtpResponse | None:
    """Retrieve an OTP; no matching message returns None."""
    async with (
        nullcontext(client)
        if client is not None
        else cloud_client(api_key, base_url, billing=billing)
    ) as sdk:
        try:
            return await sdk.mailboxes.otp(mailbox_id, **params)
        except NotFoundError:
            return None
        except APIError as exc:
            raise cloud_error(exc, "Retrieve mailbox OTP") from exc


@requires(Billing.NO_COMMERCE)
async def messages(
    mailbox_id: str,
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[MessageItem]:
    """List all mailbox message pages."""
    async with (
        nullcontext(client)
        if client is not None
        else cloud_client(api_key, base_url, billing=billing)
    ) as sdk:
        items: list[MessageItem] = []
        cursor = None
        try:
            while True:
                response = await sdk.mailboxes.messages.list(
                    mailbox_id, **({"cursor": cursor} if cursor else {})
                )
                items.extend(response.items)
                cursor = response.next_cursor
                if not cursor:
                    return items
        except APIError as exc:
            raise cloud_error(exc, "List mailbox messages") from exc
