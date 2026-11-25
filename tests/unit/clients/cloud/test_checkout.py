"""Resource plans, payment handoff, and checkout ownership contracts."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from typing import cast
from unittest.mock import AsyncMock

import pytest

from mobilerun.clients.cloud import mailboxes, numbers, proxies
from mobilerun.clients.cloud.client import AsyncMobilerun
from mobilerun.clients.cloud.payments import (
    PaymentAdapter,
    wait_for_payment,
    wait_until_active,
)
from mobilerun.core.mailbox.cloud import CloudMailbox
from mobilerun.core.phone_number.cloud import CloudPhoneNumber
from mobilerun.errors import PaymentRequired
from mobilerun.errors.cloud import BillingError
from mobilerun.models.billing import Billing
from mobilerun.models.checkout import PaymentSnapshot


def _plan_client(create):
    endpoint = NS(create=create)
    return NS(numbers=endpoint, mailboxes=endpoint)


@pytest.mark.parametrize("resource", [numbers, mailboxes])
@pytest.mark.parametrize(
    "billing,plan,expected",
    [
        ("included", None, "included_only"),
        ("paid", None, "included"),
        ("included", "auto", "included_only"),
        ("paid", "auto", "included"),
        ("included", "included", "included_only"),
        ("paid", "included", "included_only"),
        ("paid", "rent", "rent"),
    ],
)
def test_create_preference(resource, billing, plan, expected):
    data = NS(
        number_id="num",
        state="active",
        checkout_url=None,
        id="box",
        status="active",
        address="x@example.com",
    )
    create = AsyncMock(return_value=NS(data=data))
    client = _plan_client(create)
    kwargs = {} if plan is None else {"plan": plan}
    asyncio.run(resource.create(billing=billing, client=client, **kwargs))
    assert create.await_args.kwargs["billing_preference"] == expected


@pytest.mark.parametrize("resource", [numbers, mailboxes])
@pytest.mark.parametrize("billing", ["included", "no_commerce"])
def test_rent_requires_paid(resource, billing):
    create = AsyncMock()
    client = _plan_client(create)
    with pytest.raises(BillingError) as exc:
        asyncio.run(resource.create(billing=billing, client=client, plan="rent"))
    assert exc.value.code == "invalid_billing"
    assert exc.value.required_billing == "paid"
    create.assert_not_awaited()


@pytest.mark.parametrize("resource", [numbers, mailboxes])
def test_invalid_plan_rejected_before_sdk(resource):
    create = AsyncMock()
    client = _plan_client(create)
    with pytest.raises(ValueError):
        asyncio.run(resource.create(billing="paid", client=client, plan="invalid"))
    create.assert_not_awaited()


@pytest.mark.parametrize("resource", [numbers, mailboxes])
@pytest.mark.parametrize("field", ["extra_body", "extra_query"])
def test_billing_preference_cannot_be_smuggled(resource, field):
    create = AsyncMock()
    client = _plan_client(create)
    with pytest.raises(BillingError) as exc:
        asyncio.run(
            resource.create(
                billing="paid",
                client=client,
                plan="rent",
                **{field: {"billing_preference": "included"}},
            )
        )
    assert exc.value.code == "billing_override"
    create.assert_not_awaited()


@pytest.mark.parametrize(
    "cls,resource", [(CloudPhoneNumber, numbers), (CloudMailbox, mailboxes)]
)
@pytest.mark.parametrize("plan", ["auto", "included", "rent"])
def test_core_forwards_plan(monkeypatch, cls, resource, plan):
    data = NS(
        number_id="num",
        state="active",
        checkout_url=None,
        id="box",
        status="active",
        address="x@example.com",
    )
    create = AsyncMock(return_value=NS(data=data))
    monkeypatch.setattr(resource, "create", create)

    async def run():
        return await cls(billing="paid", plan=plan)

    asyncio.run(run())
    assert create.await_args.kwargs["plan"] == plan


@pytest.mark.parametrize(
    "cls,resource", [(CloudPhoneNumber, numbers), (CloudMailbox, mailboxes)]
)
@pytest.mark.parametrize(
    "status,expected", [("available", 3), ("exhausted", 0), ("not_included", 0)]
)
def test_included_available(monkeypatch, cls, resource, status, expected):
    remaining = 3 if status != "exhausted" else 0
    capacity = AsyncMock(
        return_value=NS(
            data=NS(status=status, remaining=remaining, included_remaining=99)
        )
    )
    monkeypatch.setattr(resource, "capacity", capacity)
    args = ("US",) if cls is CloudPhoneNumber else ()
    client = object()
    assert (
        asyncio.run(cls.included_available(*args, billing="no_commerce", client=client))
        == expected
    )
    capacity.assert_awaited_once_with(*args, billing="no_commerce", client=client)


def test_resource_plan_public_export():
    from mobilerun.models import ResourcePlan

    assert [p.value for p in ResourcePlan] == ["auto", "included", "rent"]


@pytest.mark.parametrize("cls", [CloudPhoneNumber, CloudMailbox])
def test_constructor_rejects_invalid_plan_eagerly(cls):
    with pytest.raises(ValueError):
        cls(billing="paid", plan="invalid")


@pytest.mark.parametrize(
    "cls,resource", [(CloudPhoneNumber, numbers), (CloudMailbox, mailboxes)]
)
def test_existing_resource_does_not_create(monkeypatch, cls, resource):
    data = NS(
        phone_number="+10000000000",
        device_binding=None,
        state="active",
        checkout_url=None,
        address="a@example.com",
        status="active",
    )
    retrieve = AsyncMock(return_value=NS(data=data))
    create = AsyncMock()
    monkeypatch.setattr(resource, "retrieve", retrieve)
    monkeypatch.setattr(resource, "create", create)

    async def run():
        return await cls("existing", billing="included", plan="rent")

    asyncio.run(run())
    create.assert_not_awaited()
    retrieve.assert_awaited_once()


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


def test_delayed_checkout_is_reported_without_cancel(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    expiry = datetime(2030, 1, 1, tzinfo=timezone.utc)
    endpoint = NS(
        create=AsyncMock(
            return_value=NS(
                data=NS(
                    id="box",
                    status="awaiting_payment",
                    checkout_url=None,
                    checkout_expires_at=None,
                )
            )
        ),
        retrieve=AsyncMock(
            side_effect=[
                NS(
                    data=NS(
                        status="awaiting_payment",
                        checkout_url=None,
                        checkout_expires_at=None,
                    )
                ),
                NS(
                    data=NS(
                        status="awaiting_payment",
                        checkout_url="https://example.test/pay",
                        checkout_expires_at=expiry,
                    )
                ),
            ]
        ),
        delete=AsyncMock(),
    )
    with pytest.raises(PaymentRequired) as caught:
        asyncio.run(
            mailboxes.create(
                client=cast(AsyncMobilerun, NS(mailboxes=endpoint)),
                billing=Billing.PAID,
            )
        )
    assert caught.value.payment_url == "https://example.test/pay"
    assert caught.value.checkout_expires_at == expiry
    endpoint.delete.assert_not_awaited()


def test_interactive_checkout_waits_until_active(monkeypatch, capsys):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    monkeypatch.setattr("webbrowser.open", lambda url: True)
    endpoint = NS(
        create=AsyncMock(
            return_value=NS(
                data=NS(
                    id="box",
                    status="awaiting_payment",
                    checkout_url="https://example.test/pay",
                    checkout_expires_at=None,
                )
            )
        ),
        retrieve=AsyncMock(
            side_effect=[
                NS(
                    data=NS(
                        status="provisioning",
                        checkout_url=None,
                        checkout_expires_at=None,
                    )
                ),
                NS(
                    data=NS(
                        status="active", checkout_url=None, checkout_expires_at=None
                    )
                ),
            ]
        ),
        delete=AsyncMock(),
    )
    callback = AsyncMock()
    asyncio.run(
        mailboxes.create(
            client=cast(AsyncMobilerun, NS(mailboxes=endpoint)),
            billing=Billing.PAID,
            on_checkout=callback,
        )
    )
    callback.assert_awaited_once()
    assert not capsys.readouterr().out
    endpoint.delete.assert_not_awaited()


def test_left_payment_without_url_is_error(monkeypatch):
    from mobilerun.errors import CloudError

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    cancel = AsyncMock()
    adapter = PaymentAdapter(
        "mailbox", "box", AsyncMock(return_value=PaymentSnapshot("failed")), cancel
    )
    with pytest.raises(CloudError, match="without a checkout URL"):
        asyncio.run(
            wait_for_payment(
                adapter, PaymentSnapshot("awaiting_payment"), resume=AsyncMock()
            )
        )
    cancel.assert_not_awaited()


def test_checkout_budget_and_interactive_timeout_cancel(monkeypatch):
    monkeypatch.setattr("webbrowser.open", lambda url: True)
    cancel = AsyncMock()
    adapter = PaymentAdapter("mailbox", "box", AsyncMock(), cancel)
    with pytest.raises(TimeoutError, match="box.*check the dashboard"):
        asyncio.run(
            wait_for_payment(
                adapter,
                PaymentSnapshot("awaiting_payment"),
                resume=AsyncMock(),
                checkout_timeout=0,
            )
        )
    cancel.assert_not_awaited()
    with pytest.raises(TimeoutError, match="box.*check the dashboard"):
        asyncio.run(
            wait_for_payment(
                adapter,
                PaymentSnapshot("awaiting_payment", "https://example.test"),
                on_checkout=lambda checkout: None,
                resume=AsyncMock(),
                timeout=0,
            )
        )
    cancel.assert_awaited_once()


def test_interactive_cancellation_cancels_pending(monkeypatch):
    monkeypatch.setattr("webbrowser.open", lambda url: True)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError))
    cancel = AsyncMock()
    adapter = PaymentAdapter(
        "mailbox",
        "box",
        AsyncMock(return_value=PaymentSnapshot("provisioning")),
        cancel,
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            wait_for_payment(
                adapter,
                PaymentSnapshot("awaiting_payment", "https://example.test"),
                on_checkout=lambda checkout: None,
                resume=AsyncMock(),
            )
        )
    cancel.assert_awaited_once()


def test_proxy_checking_then_checkout(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    endpoint = NS(
        buy=AsyncMock(return_value=NS(id="p", status="checking", payment_url=None)),
        retrieve=AsyncMock(
            return_value=NS(
                status="pending_payment", payment_url="https://example.test"
            )
        ),
        cancel=AsyncMock(),
    )
    context = AsyncMock()
    context.__aenter__.return_value = NS(connect=NS(proxies=endpoint))
    monkeypatch.setattr(proxies, "cloud_client", lambda *args, **kwargs: context)
    with pytest.raises(PaymentRequired) as caught:
        asyncio.run(proxies.buy_proxy("US", billing=Billing.PAID))
    assert caught.value.checkout.url == "https://example.test"
    assert "https://example.test" not in str(caught.value)
    endpoint.cancel.assert_not_awaited()


@pytest.mark.parametrize("resource", ["number", "proxy"])
@pytest.mark.parametrize("timeout", [False, True])
def test_cli_interactive_paid_checkout(monkeypatch, resource, timeout):
    from click.testing import CliRunner

    from mobilerun.cli.app import cli

    url = "https://example.test/pay"
    opened = []
    monkeypatch.setattr("mobilerun.cli.base.menu.is_interactive", lambda: True)

    async def confirm(message, **kwargs):
        print(message)
        return True

    monkeypatch.setattr("mobilerun.cli.base.safety.confirm_prompt_async", confirm)
    monkeypatch.setattr("webbrowser.open", lambda value: opened.append(value))
    monkeypatch.setattr(
        asyncio, "sleep", AsyncMock(side_effect=TimeoutError if timeout else None)
    )
    if resource == "number":
        data = NS(number_id="n1", state="awaiting_payment", checkout_url=url)
        active = NS(state="active", checkout_url=None, phone_number="+123")
        endpoint = NS(
            create=AsyncMock(return_value=NS(data=data)),
            retrieve=AsyncMock(return_value=NS(data=data if timeout else active)),
            delete=AsyncMock(),
        )
        client = NS(numbers=endpoint)
        module = numbers
        args = ["number", "create", "--country", "US", "--billing", "paid"]
        cancel = endpoint.delete
        success = "✓ Created number +123 (n1; active)"
    else:
        endpoint = NS(
            buy=AsyncMock(
                return_value=NS(id="p1", status="pending_payment", payment_url=url)
            ),
            retrieve=AsyncMock(
                return_value=NS(
                    status="pending_payment" if timeout else "active", payment_url=None
                )
            ),
            cancel=AsyncMock(),
        )
        client = NS(connect=NS(proxies=endpoint))
        module = proxies
        args = ["proxy", "buy", "--country", "US", "--billing", "paid"]
        cancel = endpoint.cancel
        success = "✓ Bought proxy p1 (US)"
    context = AsyncMock()
    context.__aenter__.return_value = client
    monkeypatch.setattr(module, "cloud_client", lambda *args, **kwargs: context)
    result = CliRunner().invoke(cli, args, input="y\n")
    assert "Continue?" in result.output
    assert f"Complete payment: {url}" in result.output
    assert opened == [url]
    if timeout:
        assert result.exit_code != 0
        assert "Cancelled pending" in result.output
        cancel.assert_awaited_once()
        assert success not in result.output
    else:
        assert result.exit_code == 0, result.output
        assert success in result.output
        endpoint.retrieve.assert_awaited()
        cancel.assert_not_awaited()


def test_checkout_callback_is_typed_and_library_is_silent(monkeypatch, capsys):
    from unittest.mock import Mock

    from mobilerun.models import Checkout

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    callback = Mock()
    endpoint = NS(
        create=AsyncMock(
            return_value=NS(
                data=NS(
                    id="box",
                    status="awaiting_payment",
                    checkout_url="https://example.test/pay",
                    checkout_expires_at=None,
                )
            )
        ),
        retrieve=AsyncMock(
            return_value=NS(
                data=NS(status="active", checkout_url=None, checkout_expires_at=None)
            )
        ),
        delete=AsyncMock(),
    )
    asyncio.run(
        mailboxes.create(
            client=cast(AsyncMobilerun, NS(mailboxes=endpoint)),
            billing=Billing.PAID,
            on_checkout=callback,
        )
    )
    callback.assert_called_once_with(
        Checkout("https://example.test/pay", None, "mailbox", "box")
    )
    assert not capsys.readouterr().out


@pytest.mark.parametrize("kind", ["mailbox", "number"])
def test_core_payment_error_resumes_to_resource(monkeypatch, kind):
    from mobilerun.core.mailbox import CloudMailbox
    from mobilerun.core.phone_number import CloudPhoneNumber
    from mobilerun.models import Checkout

    module = mailboxes if kind == "mailbox" else numbers
    resource_type = CloudMailbox if kind == "mailbox" else CloudPhoneNumber
    checkout = Checkout("https://example.test/pay", None, kind, "resource")
    monkeypatch.setattr(
        module, "create", AsyncMock(side_effect=PaymentRequired(checkout, AsyncMock()))
    )
    monkeypatch.setattr(
        module,
        "retrieve",
        AsyncMock(
            return_value=NS(
                data=NS(
                    status="active",
                    state="active",
                    address="test@example.test",
                    phone_number="+123",
                    checkout_url=None,
                    checkout_expires_at=None,
                )
            )
        ),
    )
    cancel = AsyncMock()
    monkeypatch.setattr(module, "delete", cancel)

    async def run():
        with pytest.raises(PaymentRequired) as caught:
            await resource_type(billing="paid")
        assert caught.value.checkout == checkout
        resource = await caught.value.wait_until_paid()
        assert isinstance(resource, resource_type)
        assert resource.id == "resource"
        return resource

    asyncio.run(run())
    cancel.assert_not_awaited()


def test_resumed_timeout_does_not_cancel():
    from mobilerun.models import Checkout

    cancel = AsyncMock()
    adapter = PaymentAdapter(
        "mailbox",
        "box",
        AsyncMock(return_value=PaymentSnapshot("awaiting_payment")),
        cancel,
    )

    async def resume(timeout):
        await wait_until_active(adapter, timeout=timeout)

    error = PaymentRequired(
        Checkout("https://example.test", None, "mailbox", "box"), resume
    )
    with pytest.raises(TimeoutError, match="box"):
        asyncio.run(error.wait_until_paid(timeout=0))
    cancel.assert_not_awaited()


def test_slow_async_callback_is_timed_out_and_cancelled():
    cancel = AsyncMock()
    adapter = PaymentAdapter("mailbox", "box", AsyncMock(), cancel)

    async def callback(checkout):
        await asyncio.Event().wait()

    with pytest.raises(TimeoutError, match="box"):
        asyncio.run(
            wait_for_payment(
                adapter,
                PaymentSnapshot("awaiting_payment", "https://example.test"),
                resume=AsyncMock(),
                on_checkout=callback,
                timeout=0.01,
            )
        )
    cancel.assert_awaited_once()


@pytest.mark.parametrize("mode", ["print", "browser"])
def test_checkout_modes_print_and_wait(mode, monkeypatch, capsys):
    from mobilerun.clients.cloud.checkout import checkout_callback

    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    monkeypatch.setattr("sys.platform", "darwin")
    retrieve = AsyncMock(return_value=PaymentSnapshot("active"))
    adapter = PaymentAdapter("mailbox", "box", retrieve, AsyncMock())
    asyncio.run(
        wait_for_payment(
            adapter,
            PaymentSnapshot("awaiting_payment", "https://example.test"),
            resume=AsyncMock(),
            on_checkout=checkout_callback(mode, None),
        )
    )
    assert "Complete payment: https://example.test" in capsys.readouterr().out
    assert opened == (["https://example.test"] if mode == "browser" else [])
    retrieve.assert_awaited_once()


def test_checkout_mode_and_callback_conflict():
    with pytest.raises(ValueError, match="on_checkout"):
        CloudMailbox(checkout="print", on_checkout=lambda _: None)


def test_browser_checkout_headless_still_prints_expiry(monkeypatch, capsys):
    from mobilerun.clients.cloud.checkout import browser_checkout
    from mobilerun.models.checkout import Checkout

    opened = []
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    browser_checkout(
        Checkout("https://example.test", datetime(2030, 1, 1), "mailbox", "box")
    )
    assert "expires 2030-01-01" in capsys.readouterr().out
    assert not opened


@pytest.mark.parametrize("resource", ["mailbox", "number", "proxy"])
def test_cloud_creators_reject_checkout_and_callback(resource):
    creator = {
        "mailbox": mailboxes.create,
        "number": numbers.create,
        "proxy": proxies.buy_proxy,
    }[resource]
    params = {"country": "US"} if resource == "proxy" else {}
    with pytest.raises(ValueError, match="on_checkout"):
        asyncio.run(
            creator(
                billing=Billing.PAID,
                checkout="print",
                on_checkout=lambda _: None,
                **params,
            )
        )


@pytest.mark.parametrize("async_callback", [False, True])
@pytest.mark.parametrize("kind", ["mailbox", "number"])
def test_callback_failure_releases_created_resource(kind, async_callback):
    url = "https://example.test/pay"
    if kind == "mailbox":
        data = NS(
            id="box",
            status="awaiting_payment",
            checkout_url=url,
            checkout_expires_at=None,
        )
        creator, endpoint_name, resource_id = mailboxes.create, "mailboxes", "box"
    else:
        data = NS(number_id="number", state="awaiting_payment", checkout_url=url)
        creator, endpoint_name, resource_id = numbers.create, "numbers", "number"
    endpoint = NS(create=AsyncMock(return_value=NS(data=data)), delete=AsyncMock())
    client = cast(AsyncMobilerun, NS(**{endpoint_name: endpoint}))

    def callback(_):
        raise RuntimeError("callback failed")

    async def async_cb(_):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        asyncio.run(
            creator(
                client=client,
                billing=Billing.PAID,
                on_checkout=async_cb if async_callback else callback,
            )
        )
    endpoint.delete.assert_awaited_once_with(resource_id)
