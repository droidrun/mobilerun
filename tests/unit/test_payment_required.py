"""Checkout preserves pending resources in non-interactive mode."""

import asyncio
from types import SimpleNamespace as NS
from typing import cast
from unittest.mock import AsyncMock

import pytest

from mobilerun.clients.cloud import mailboxes, numbers, proxies
from mobilerun.clients.cloud.client import AsyncMobilerun
from mobilerun.errors import PaymentRequired
from mobilerun.models.billing import Billing


@pytest.mark.parametrize("kind", ["number", "mailbox", "proxy"])
@pytest.mark.parametrize("keep", [False, True])
def test_payment_required_preserves_resource(monkeypatch, kind, keep):
    url = "https://checkout.example.test/pay"
    data = NS(
        number_id="id",
        id="id",
        state="awaiting_payment",
        status="pending_payment" if kind == "proxy" else "awaiting_payment",
        checkout_url=url,
        checkout_expires_at=None,
        payment_url=url,
    )
    create = AsyncMock(return_value=data if kind == "proxy" else NS(data=data))
    cancel = AsyncMock()
    sdk = NS(
        numbers=NS(create=create, delete=cancel),
        mailboxes=NS(create=create, delete=cancel),
        connect=NS(proxies=NS(buy=create, cancel=cancel)),
    )
    context = AsyncMock()
    context.__aenter__.return_value = sdk
    monkeypatch.setattr(proxies, "cloud_client", lambda *args, **kwargs: context)

    async def run():
        with pytest.raises(PaymentRequired) as caught:
            if kind == "proxy":
                await proxies.buy_proxy("US", billing=Billing.PAID, keep=keep)
            else:
                await (numbers if kind == "number" else mailboxes).create(
                    client=cast(AsyncMobilerun, sdk), billing=Billing.PAID, keep=keep
                )
        assert caught.value.payment_url == url
        assert url not in str(caught.value)

    asyncio.run(run())
    cancel.assert_not_awaited()
    assert create.await_args is not None
    assert "keep" not in create.await_args.kwargs


def test_nonpayment_proxy_response_is_unchanged(monkeypatch):
    response = NS(id="id", status="provisioning", payment_url=None)
    endpoint = NS(buy=AsyncMock(return_value=response), cancel=AsyncMock())
    context = AsyncMock()
    context.__aenter__.return_value = NS(connect=NS(proxies=endpoint))
    monkeypatch.setattr(proxies, "cloud_client", lambda *args, **kwargs: context)
    assert asyncio.run(proxies.buy_proxy("US", billing=Billing.PAID)) is response
    endpoint.cancel.assert_not_awaited()
