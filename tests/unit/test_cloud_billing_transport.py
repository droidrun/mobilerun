"""Commerce decisions are enforced before real SDK serialization."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import cast

import httpx
import pytest
from mobilerun_sdk import AsyncMobilerun

from mobilerun.clients.cloud import devices, mailboxes, numbers
from mobilerun.errors import CloudAPIError
from mobilerun.models.billing import Billing


@pytest.mark.parametrize("adapter", [devices, numbers, mailboxes])
@pytest.mark.parametrize(
    "field", ["billing", "billingPreference", "billing_preference", "proxy"]
)
def test_extra_body_cannot_reach_transport(adapter, field):
    seen = []

    async def run():
        async with AsyncMobilerun(
            api_key="test",
            base_url="https://sdk.test/v1",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: seen.append(request) or httpx.Response(400)
                )
            ),
        ) as client:
            with pytest.raises(ValueError):
                await adapter.create(
                    client=client, billing="included", extra_body={field: "rent"}
                )

    asyncio.run(run())
    assert not seen


@pytest.mark.parametrize(
    "mode,plan,expected",
    [
        ("paid", None, "auto"),
        ("paid", "auto", "auto"),
        ("paid", "minute", "minute"),
        ("paid", "subscription", "subscription"),
        ("included", None, "subscription"),
        ("included", "subscription", "subscription"),
    ],
)
def test_plan_reaches_real_sdk_body(mode, plan, expected):
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(400, json={"message": "test rejection"})

    async def run():
        async with AsyncMobilerun(
            api_key="test",
            base_url="https://sdk.test/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            with pytest.raises(CloudAPIError):
                await devices.create(
                    client=client,
                    billing=mode,
                    plan=plan,
                    device_type="android_cloud_phone",
                    proxy={"connect": {"id": "existing"}},
                )

    asyncio.run(run())
    assert len(seen) == 1
    assert seen[0]["billing"] == expected


@pytest.mark.parametrize(
    "mode,plan",
    [
        ("no_commerce", None),
        ("no_commerce", "auto"),
        ("no_commerce", "minute"),
        ("no_commerce", "subscription"),
        ("included", "auto"),
        ("included", "minute"),
    ],
)
def test_blocked_plan_never_reaches_transport(mode, plan):
    seen = []

    async def run():
        async with AsyncMobilerun(
            api_key="test",
            base_url="https://sdk.test/v1",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: seen.append(request) or httpx.Response(400)
                )
            ),
        ) as client:
            with pytest.raises(ValueError):
                await devices.create(
                    client=client,
                    billing=mode,
                    plan=plan,
                    proxy={"connect": {"id": "existing"}},
                )

    asyncio.run(run())
    assert not seen


def test_extra_query_cannot_override_device_billing():
    seen = []

    async def run():
        async with AsyncMobilerun(
            api_key="test",
            base_url="https://sdk.test/v1",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: seen.append(request) or httpx.Response(400)
                )
            ),
        ) as client:
            with pytest.raises(ValueError):
                await devices.create(
                    client=client,
                    billing=Billing.INCLUDED,
                    extra_query={"billing": "minute"},
                )

    asyncio.run(run())
    assert not seen


@pytest.mark.parametrize(
    "proxy",
    [
        {},
        {"connect": {}},
        {"connect": {"country": None}},
        {"socks5": {"host": "example.test"}},
    ],
)
def test_invalid_explicit_cloud_phone_proxy_is_rejected_before_http(proxy):
    seen = []

    async def run():
        async with AsyncMobilerun(
            api_key="test",
            base_url="https://sdk.test/v1",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: seen.append(request) or httpx.Response(400)
                )
            ),
        ) as client:
            with pytest.raises(ValueError, match="proxy"):
                await devices.create(client=client, billing=Billing.PAID, proxy=proxy)

    asyncio.run(run())
    assert not seen


@pytest.mark.parametrize(
    "billing,proxy,allowed",
    [
        ("included", {"connect": {"country": "us"}}, False),
        ("included", {"connect": {"id": "existing"}}, True),
        ("included", {"socks5": {"host": "example.test", "port": 1080}}, True),
        ("paid", {"connect": {"country": "us"}}, True),
        ("no_commerce", {"connect": {"id": "existing"}}, False),
    ],
)
def test_proxy_billing_is_checked_before_transport(billing, proxy, allowed):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(400, json={"message": "test rejection"})

    async def run():
        async with AsyncMobilerun(
            api_key="test",
            base_url="https://sdk.test/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            with pytest.raises(CloudAPIError):
                await devices.create(client=client, billing=billing, proxy=proxy)

    asyncio.run(run())
    assert bool(seen) is allowed


@pytest.mark.parametrize("adapter", [numbers, mailboxes])
@pytest.mark.parametrize(
    "mode,expected", [("included", "included_only"), ("paid", "included")]
)
def test_number_and_mailbox_billing_serialized(adapter, mode, expected):
    import json

    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(400, json={"message": "test rejection"})

    async def run():
        async with AsyncMobilerun(
            api_key="test",
            base_url="https://sdk.test/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            with pytest.raises(CloudAPIError):
                await adapter.create(client=client, billing=mode)

    asyncio.run(run())
    assert bodies[0]["billingPreference"] == expected


@pytest.mark.parametrize("options", [{}, {"proxy": None}])
def test_missing_proxy_is_rejected_before_transport(options):
    seen = []

    def handler(request):
        import json

        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "phone", "name": None})

    async def run():
        async with AsyncMobilerun(
            api_key="test",
            base_url="https://sdk.test/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            with pytest.raises(ValueError, match="proxy"):
                await devices.create(client=client, billing=Billing.PAID, **options)

    asyncio.run(run())
    assert not seen


@pytest.mark.parametrize("resource", ["device", "number", "mailbox", "connect", "buy"])
@pytest.mark.parametrize("extra", ["extra_body", "extra_query"])
@pytest.mark.parametrize(
    "field", ["proxy", "connect", "billing", "billingPreference", "billingMode"]
)
def test_commerce_fields_in_extras_never_reach_real_transport(
    monkeypatch, resource, extra, field
):
    from contextlib import nullcontext

    from mobilerun.clients.cloud import proxies

    seen = []

    async def run():
        async with AsyncMobilerun(
            api_key="test",
            base_url="https://sdk.test/v1",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: seen.append(request) or httpx.Response(400)
                )
            ),
        ) as client:
            monkeypatch.setattr(
                proxies, "cloud_client", lambda *args: nullcontext(client)
            )
            extras = {extra: {field: {"country": "DE"}}}
            # Deliberately send invalid fields to exercise runtime commerce guards.
            with pytest.raises(ValueError):
                if resource == "device":
                    await cast(Callable[..., Awaitable[object]], devices.create)(
                        client=client,
                        billing="paid",
                        proxy={"connect": {"id": "existing"}},
                        **extras,
                    )
                elif resource == "number":
                    await cast(Callable[..., Awaitable[object]], numbers.create)(
                        client=client, billing=Billing.PAID, **extras
                    )
                elif resource == "mailbox":
                    await cast(Callable[..., Awaitable[object]], mailboxes.create)(
                        client=client, billing=Billing.PAID, **extras
                    )
                elif resource == "connect":
                    await cast(
                        Callable[..., Awaitable[object]], proxies.connect_device
                    )("phone", extras, billing=Billing.NO_COMMERCE)
                else:
                    await cast(Callable[..., Awaitable[object]], proxies.buy_proxy)(
                        "DE", billing=Billing.PAID, **extras
                    )

    asyncio.run(run())
    assert not seen
