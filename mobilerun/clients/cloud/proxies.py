"""Mobilerun Connect proxy management."""

from typing import Literal, TypedDict, Unpack

from mobilerun_sdk import APIError, APIStatusError
from mobilerun_sdk.types.connect.proxy_buy_response import ProxyBuyResponse
from mobilerun_sdk.types.connect.proxy_list_params import ProxyListParams
from mobilerun_sdk.types.connect.proxy_list_response import ProxyListResponse
from mobilerun_sdk.types.connect.proxy_retrieve_response import ProxyRetrieveResponse
from mobilerun_sdk.types.devices.proxy_connect_params import ProxyConnectParams
from mobilerun_sdk.types.devices.proxy_status_response import ProxyStatusResponse

from mobilerun.clients.cloud.billing import _validate_create_params, requires
from mobilerun.clients.cloud.client import cloud_client
from mobilerun.core.devices.components.identifiers import secret_scope
from mobilerun.errors.cloud import CloudConflict
from mobilerun.models.billing import Billing
from mobilerun.clients.cloud.checkout import CheckoutMode, checkout_callback
from mobilerun.clients.cloud.errors import cloud_error
from mobilerun.clients.cloud.payments import (
    CheckoutCallback,
    PaymentAdapter,
    PaymentSnapshot,
    wait_for_payment,
    wait_until_active,
)

BuyParams = TypedDict("BuyParams", {"name": str}, total=False)


@requires(Billing.NO_COMMERCE)
async def connect_device(
    device_id: str,
    params: ProxyConnectParams,
    *,
    billing: Billing,
    api_key: str | None = None,
    base_url: str | None = None,
) -> None:
    """Connect a proxy, purchasing capacity when requested."""
    await _validate_create_params(params, allow_connect=True)
    async with cloud_client(api_key, base_url, billing=billing) as client:
        socks = params.get("socks5") or {}
        with secret_scope(socks.get("password", ""), params.get("password", "")):
            try:
                return await client.devices.proxy.connect(device_id, **params)
            except APIError as exc:
                raise cloud_error(exc, "Connect phone proxy", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def disconnect_device(
    device_id: str,
    *,
    billing: Billing,
    api_key: str | None = None,
    base_url: str | None = None,
) -> None:
    """Disconnect a phone proxy."""
    async with cloud_client(api_key, base_url, billing=billing) as client:
        try:
            return await client.with_options(max_retries=0).devices.proxy.disconnect(
                device_id
            )
        except APIError as exc:
            if isinstance(exc, APIStatusError) and exc.status_code == 409:
                raise CloudConflict(
                    operation="Disconnect phone proxy",
                    status=409,
                    code="proxy_disconnect",
                ) from exc
            raise cloud_error(exc, "Disconnect phone proxy", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def device_status(
    device_id: str,
    *,
    billing: Billing,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ProxyStatusResponse:
    """Retrieve a phone proxy's status."""
    async with cloud_client(api_key, base_url, billing=billing) as client:
        try:
            return await client.devices.proxy.status(device_id)
        except APIError as exc:
            raise cloud_error(exc, "Retrieve phone proxy status", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def list_proxies(
    *,
    billing: Billing,
    api_key: str | None = None,
    base_url: str | None = None,
    **params: Unpack[ProxyListParams],
) -> ProxyListResponse:
    """List a proxy page."""
    async with cloud_client(api_key, base_url, billing=billing) as client:
        try:
            return await client.connect.proxies.list(**params)
        except APIError as exc:
            raise cloud_error(exc, "List proxies") from exc


@requires(Billing.NO_COMMERCE)
async def get_proxy(
    id: str,
    *,
    billing: Billing,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ProxyRetrieveResponse:
    """Retrieve a proxy."""
    async with cloud_client(api_key, base_url, billing=billing) as client:
        try:
            return await client.connect.proxies.retrieve(id)
        except APIError as exc:
            raise cloud_error(exc, "Retrieve proxy") from exc


@requires(Billing.NO_COMMERCE)
async def cancel_proxy(
    id: str,
    *,
    billing: Billing,
    api_key: str | None = None,
    base_url: str | None = None,
) -> None:
    """Cancel a Connect proxy and release its provisioning."""
    async with cloud_client(api_key, base_url, billing=billing) as client:
        try:
            await client.connect.proxies.cancel(id)
        except APIError as exc:
            raise cloud_error(exc, "Release proxy") from exc


@requires(Billing.PAID)
async def buy_proxy(
    country: str,
    *,
    type: Literal["dedicated_residential", "residential", "mobile"] = "residential",
    billing: Billing,
    api_key: str | None = None,
    base_url: str | None = None,
    keep: bool = False,
    on_checkout: CheckoutCallback | None = None,
    checkout: CheckoutMode = "error",
    payment_timeout: float = 600,
    **params: Unpack[BuyParams],
) -> ProxyBuyResponse:
    """Purchase Connect proxy capacity."""
    on_checkout = checkout_callback(checkout, on_checkout)
    await _validate_create_params(params)
    unknown = params.keys() - BuyParams.__annotations__.keys()
    if unknown:
        raise TypeError(f"Unexpected purchase options: {', '.join(sorted(unknown))}")
    async with cloud_client(api_key, base_url, billing=billing) as client:
        try:
            response = await client.connect.proxies.buy(
                country=country, type=type, **params
            )

            async def payment_state() -> PaymentSnapshot:
                data = await client.connect.proxies.retrieve(response.id)
                return PaymentSnapshot(data.status, data.payment_url)

            async def resume(timeout: float) -> ProxyRetrieveResponse:
                async def fetch() -> PaymentSnapshot:
                    data = await get_proxy(
                        response.id,
                        billing=Billing.NO_COMMERCE,
                        api_key=api_key,
                        base_url=base_url,
                    )
                    return PaymentSnapshot(data.status, data.payment_url)

                adapter = PaymentAdapter(
                    "Connect proxy",
                    response.id,
                    fetch,
                    lambda: client.connect.proxies.cancel(response.id),
                    preparing=frozenset({"checking"}),
                )
                await wait_until_active(adapter, timeout=timeout)
                return await get_proxy(
                    response.id,
                    billing=Billing.NO_COMMERCE,
                    api_key=api_key,
                    base_url=base_url,
                )

            await wait_for_payment(
                PaymentAdapter(
                    "Connect proxy",
                    response.id,
                    payment_state,
                    lambda: client.connect.proxies.cancel(response.id),
                    preparing=frozenset({"checking"}),
                ),
                PaymentSnapshot(response.status, response.payment_url),
                resume=resume,
                on_checkout=on_checkout,
                timeout=payment_timeout,
            )
            return response
        except APIError as exc:
            raise cloud_error(exc, "Buy Connect proxy") from exc
