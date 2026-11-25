import ast
import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest
from mobilerun_sdk import APIStatusError

from mobilerun.clients.cloud import devices, numbers, payments
from mobilerun.core.devices import AndroidDevice, CloudDevice
from mobilerun.core.devices.components import environment
from mobilerun.core.mailbox import CloudMailbox
from mobilerun.core.phone_number import CloudPhoneNumber
from mobilerun.errors import BillingError, CloudConflict, PurchaseInProgress
from mobilerun.models import Billing
from mobilerun.models.checkout import PaymentSnapshot


@pytest.mark.parametrize("resource", [CloudPhoneNumber, CloudMailbox])
@pytest.mark.parametrize("failure", [RuntimeError, TimeoutError])
def test_existing_checkout_failure_keeps_resource(resource, failure):
    data = NS(
        status="awaiting_payment",
        state="awaiting_payment",
        address=None,
        phone_number=None,
        device_binding=None,
        checkout_expires_at=None,
        checkout_url="https://checkout.test",
    )
    endpoint = NS(retrieve=AsyncMock(return_value=NS(data=data)), delete=AsyncMock())
    sdk = NS(numbers=endpoint, mailboxes=endpoint)

    async def run():
        with pytest.raises(failure):
            await resource(
                "existing", client=sdk, on_checkout=AsyncMock(side_effect=failure)
            )
        endpoint.delete.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"label": "work"},
        {"label": "personal", "idempotency_key": "new-request"},
        {"label": "personal", "extra_body": {"label": "work"}},
        {"label": "personal", "extra_query": {"country": "GB"}},
        {"label": "personal", "extra_headers": {"Idempotency-Key": "new-request"}},
    ],
)
def test_pending_purchase_requires_verifiable_identity(options):
    conflict = APIStatusError(
        "conflict",
        response=httpx.Response(
            409, request=httpx.Request("POST", "https://example.test")
        ),
        body={"code": "purchase_in_progress"},
    )
    pending = NS(
        id="pending",
        state="awaiting_payment",
        country_code="US",
        label="personal",
        checkout_url="https://checkout.test",
    )
    sdk = NS(
        numbers=NS(
            create=AsyncMock(side_effect=conflict),
            list=AsyncMock(
                return_value=NS(items=[pending], pagination=NS(has_next=False))
            ),
        )
    )
    with pytest.raises(CloudConflict) as caught:
        asyncio.run(
            numbers.create(client=sdk, billing=Billing.PAID, country="US", **options)
        )
    assert not isinstance(caught.value, PurchaseInProgress)


@pytest.mark.parametrize("intermediate", [[], ["provisioning"]])
def test_number_wait_ready_uses_activation_seen_during_checkout_poll(
    monkeypatch, intermediate
):
    retrieve = AsyncMock(
        side_effect=[
            NS(data=NS(state="awaiting_payment", checkout_url=None, phone_number=None)),
            *(
                NS(data=NS(state=state, checkout_url=None, phone_number=None))
                for state in intermediate
            ),
            NS(data=NS(state="active", checkout_url=None, phone_number="+123")),
        ]
    )
    number = CloudPhoneNumber("existing", client=NS(numbers=NS(retrieve=retrieve)))
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    assert asyncio.run(number.wait_ready()) is number
    assert number.phone_number == "+123"


def test_paid_binding_smoke_calls_match_device_client_signatures():
    source = Path(__file__).parent / "smoke/test_phone_number_binding.py"
    for call in ast.walk(ast.parse(source.read_text())):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "devices"
        ):
            inspect.signature(getattr(devices, call.func.attr)).bind(
                *[None for _ in call.args], **{kw.arg: None for kw in call.keywords}
            )


@pytest.mark.parametrize(
    "billing,plan,country",
    [("included", "auto", "US"), ("paid", "included", "US"), ("paid", "rent", "GB")],
)
def test_incompatible_pending_purchase_does_not_open_checkout(billing, plan, country):
    response = httpx.Response(
        409, request=httpx.Request("POST", "https://example.test")
    )
    conflict = APIStatusError(
        "conflict", response=response, body={"code": "purchase_in_progress"}
    )
    pending = NS(
        id="pending",
        state="awaiting_payment",
        country_code="US",
        checkout_url="https://checkout.test",
    )
    sdk = NS(
        numbers=NS(
            create=AsyncMock(side_effect=conflict),
            list=AsyncMock(
                return_value=NS(items=[pending], pagination=NS(has_next=False))
            ),
        )
    )
    callback = AsyncMock(side_effect=AssertionError("unexpected checkout"))

    async def run():
        with pytest.raises(CloudConflict) as caught:
            await CloudPhoneNumber(
                client=sdk,
                country=country,
                billing=billing,
                plan=plan,
                on_checkout=callback,
            )
        assert (
            not isinstance(caught.value, PurchaseInProgress)
            or caught.value.checkout is None
        )
        callback.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("billing", [Billing.NO_COMMERCE, Billing.INCLUDED])
def test_sms_requires_paid_before_transport(billing):
    send = AsyncMock()
    sdk = NS(numbers=NS(messages=NS(send=send)))
    with pytest.raises(BillingError):
        asyncio.run(
            numbers.send("existing", "+123", "test", billing=billing, client=sdk)
        )
    send.assert_not_awaited()


def test_checking_proxy_returns_at_provisioning_without_checkout_or_cancel(monkeypatch):
    provisioning = PaymentSnapshot("provisioning")
    retrieve = AsyncMock(side_effect=[provisioning, TimeoutError()])
    callback, cancel = AsyncMock(), AsyncMock()
    adapter = payments.PaymentAdapter(
        "Connect proxy", "proxy", retrieve, cancel, preparing=frozenset({"checking"})
    )
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    result = asyncio.run(
        payments.wait_for_payment(
            adapter,
            PaymentSnapshot("checking"),
            resume=AsyncMock(),
            on_checkout=callback,
        )
    )
    assert result == provisioning
    retrieve.assert_awaited_once()
    callback.assert_not_awaited()
    cancel.assert_not_awaited()


@pytest.mark.parametrize("states", [["active"], ["provisioning", "active"]])
def test_activation_without_checkout_url(monkeypatch, states):
    retrieve = AsyncMock(side_effect=[PaymentSnapshot(state) for state in states])
    adapter = payments.PaymentAdapter("number", "existing", retrieve, AsyncMock())
    callback = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    result = asyncio.run(
        payments.wait_for_payment(
            adapter,
            PaymentSnapshot("awaiting_payment"),
            resume=AsyncMock(),
            on_checkout=callback,
        )
    )
    assert result.status == "active"
    callback.assert_not_awaited()


@pytest.mark.parametrize("resource", [CloudPhoneNumber, CloudMailbox])
def test_create_without_checkout_url_returns_usable_resource(monkeypatch, resource):
    def response(state):
        if resource is CloudPhoneNumber:
            return NS(
                data=NS(
                    number_id="n",
                    state=state,
                    checkout_url=None,
                    phone_number="+123" if state == "active" else None,
                )
            )
        return NS(
            data=NS(
                id="m",
                status=state,
                checkout_url=None,
                checkout_expires_at=None,
                address="a@example.test" if state == "active" else None,
            )
        )

    states = iter(["provisioning", "active"])
    endpoint = NS(
        create=AsyncMock(return_value=response("awaiting_payment")),
        retrieve=AsyncMock(side_effect=lambda _: response(next(states, "active"))),
        delete=AsyncMock(),
    )
    sdk = NS(numbers=endpoint, mailboxes=endpoint)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    async def run():
        async with resource(client=sdk, billing=Billing.PAID) as handle:
            if resource is CloudPhoneNumber:
                assert (await handle.wait_ready()).phone_number == "+123"
            else:
                assert handle.address == "a@example.test"
        endpoint.delete.assert_awaited_once_with(handle.id)

    asyncio.run(run())


@pytest.mark.parametrize("cloud", [False, True])
def test_device_catalog_dispatches_open_url(monkeypatch, cloud):
    device = (
        CloudDevice("existing", api_key="test") if cloud else AndroidDevice("local")
    )
    backend = AsyncMock()
    monkeypatch.setattr(
        environment, "cloud_open_url" if cloud else "android_shell", backend
    )
    url = "https://example.test"
    asyncio.run(device.actions.call("device.open_url", {"url": url}))
    backend.assert_awaited_once_with(
        device, url if cloud else f"am start -a android.intent.action.VIEW -d {url}"
    )
