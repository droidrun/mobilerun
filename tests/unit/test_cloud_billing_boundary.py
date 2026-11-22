"""Every billable adapter uses the same fail-closed commerce policy."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from mobilerun.clients.cloud import devices, mailboxes, numbers, proxies
from mobilerun.errors import BillingError
from mobilerun.models.billing import Billing


@pytest.mark.parametrize(
    "resource",
    ["cloud phones", "numbers", "mailboxes", "Connect proxies"],
)
@pytest.mark.parametrize("setting", list(Billing))
@pytest.mark.parametrize(
    "explicit", [None, *Billing, "paid", "included", "no_commerce"]
)
def test_policy_matrix(monkeypatch, resource, setting, explicit):
    monkeypatch.setenv("MOBILERUN_BILLING", setting)
    effective = None if explicit is None else Billing(explicit)
    blocked = effective in (None, Billing.NO_COMMERCE) or (
        effective is Billing.INCLUDED and resource == "Connect proxies"
    )
    if resource == "Connect proxies":
        # A fake factory must never be reached for refused commerce.
        def client(*args, **kwargs):
            raise AssertionError("HTTP reached")

        monkeypatch.setattr(proxies, "cloud_client", client)
        if not blocked:
            return

        def call():
            return proxies.buy_proxy("US", billing=explicit)

    else:
        adapter = {"cloud phones": devices, "numbers": numbers, "mailboxes": mailboxes}[
            resource
        ]
        endpoint = AsyncMock(
            return_value=NS(
                id="created",
                data=NS(
                    number_id="created",
                    id="created",
                    state="active",
                    status="active",
                    checkout_url=None,
                ),
            )
        )
        client = NS(
            **{
                "devices" if resource == "cloud phones" else resource: NS(
                    create=endpoint
                )
            }
        )
        params = (
            {"proxy": {"connect": {"id": "existing"}}}
            if resource == "cloud phones"
            else {}
        )

        def call():
            return adapter.create(billing=explicit, client=client, **params)

    if blocked:
        with pytest.raises(BillingError, match="given.*requires.*MOBILERUN_BILLING"):
            asyncio.run(call())
    else:
        expected = (
            ("included_only" if effective is Billing.INCLUDED else "included")
            if resource in {"numbers", "mailboxes"}
            else ("subscription" if effective is Billing.INCLUDED else "auto")
        )
        asyncio.run(call())
        key = "billing" if resource == "cloud phones" else "billing_preference"
        assert endpoint.call_args.kwargs[key] == expected


@pytest.mark.parametrize(
    "adapter,resource",
    [(numbers, "numbers"), (mailboxes, "mailboxes"), (devices, "devices")],
)
@pytest.mark.parametrize("mode", list(Billing))
def test_resource_calls_cannot_skip_policy(adapter, resource, mode):
    create = AsyncMock(
        return_value=NS(
            id="created",
            data=NS(
                number_id="created",
                id="created",
                state="active",
                status="active",
                checkout_url=None,
            ),
        )
    )
    client = NS(**{resource: NS(create=create)})
    params = {"proxy": {"connect": {"id": "existing"}}} if resource == "devices" else {}
    if mode is Billing.NO_COMMERCE:
        with pytest.raises(ValueError):
            asyncio.run(adapter.create(client=client, billing=mode, **params))
        create.assert_not_awaited()
    else:
        asyncio.run(adapter.create(client=client, billing=mode, **params))
        expected = (
            ("subscription" if mode is Billing.INCLUDED else "auto")
            if resource == "devices"
            else ("included_only" if mode is Billing.INCLUDED else "included")
        )
        create.assert_awaited_once_with(
            **{"billing" if resource == "devices" else "billing_preference": expected},
            **params,
        )


@pytest.mark.parametrize("mode", ["no_commerce", "included", "paid"])
@pytest.mark.parametrize("plan", [None, "auto", "minute", "subscription"])
def test_cloud_phone_plan_policy(mode, plan):
    blocked = mode == "no_commerce" or (
        mode == "included" and plan in {"auto", "minute"}
    )
    if blocked:
        with pytest.raises(ValueError):
            asyncio.run(devices.create(billing=mode, plan=plan))
    else:
        expected = plan or ("subscription" if mode == "included" else "auto")
        create = AsyncMock()
        asyncio.run(
            devices.create(
                billing=mode,
                plan=plan,
                proxy={"connect": {"id": "existing"}},
                client=NS(devices=NS(create=create)),
            )
        )
        assert create.call_args.kwargs["billing"] == expected


@pytest.mark.parametrize(
    "adapter,resource",
    [(numbers, "numbers"), (mailboxes, "mailboxes"), (devices, "devices")],
)
@pytest.mark.parametrize(
    "field", ["billing", "billingPreference", "billing_preference", "proxy"]
)
def test_extra_body_cannot_override_commerce_policy(adapter, resource, field):
    create = AsyncMock()
    client = NS(**{resource: NS(create=create)})
    with pytest.raises(ValueError):
        asyncio.run(
            adapter.create(
                client=client, billing="included", extra_body={field: "rent"}
            )
        )
    create.assert_not_awaited()
