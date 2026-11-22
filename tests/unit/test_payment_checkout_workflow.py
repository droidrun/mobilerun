import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from typing import cast
from unittest.mock import AsyncMock

import pytest

from mobilerun.clients.cloud import mailboxes
from mobilerun.clients.cloud.client import AsyncMobilerun
from mobilerun.errors import PaymentRequired
from mobilerun.models.billing import Billing


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
    from mobilerun.clients.cloud.payments import (
        PaymentAdapter,
        PaymentSnapshot,
        wait_for_payment,
    )
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
    from mobilerun.clients.cloud.payments import (
        PaymentAdapter,
        PaymentSnapshot,
        wait_for_payment,
    )

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
    from mobilerun.clients.cloud.payments import (
        PaymentAdapter,
        PaymentSnapshot,
        wait_for_payment,
    )

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
    from mobilerun.clients.cloud import proxies

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
    from mobilerun.clients.cloud import numbers, proxies

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
    from mobilerun.clients.cloud import numbers
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
    from mobilerun.clients.cloud.payments import (
        PaymentAdapter,
        PaymentSnapshot,
        wait_until_active,
    )
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
    from mobilerun.clients.cloud.payments import (
        PaymentAdapter,
        PaymentSnapshot,
        wait_for_payment,
    )

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
    from mobilerun.clients.cloud.payments import (
        PaymentAdapter,
        PaymentSnapshot,
        wait_for_payment,
    )

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
    from mobilerun.core.mailbox.cloud import CloudMailbox

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
    from mobilerun.clients.cloud import numbers, proxies

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
def test_mailbox_callback_failure_releases_created_resource(async_callback):
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
        delete=AsyncMock(),
    )

    def callback(_):
        raise RuntimeError("callback failed")

    async def async_cb(_):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        asyncio.run(
            mailboxes.create(
                client=cast(AsyncMobilerun, NS(mailboxes=endpoint)),
                billing=Billing.PAID,
                on_checkout=async_cb if async_callback else callback,
            )
        )
    endpoint.delete.assert_awaited_once_with("box")


@pytest.mark.parametrize("async_callback", [False, True])
def test_number_callback_failure_releases_created_resource(async_callback):
    from mobilerun.clients.cloud import numbers

    endpoint = NS(
        create=AsyncMock(
            return_value=NS(
                data=NS(
                    number_id="number",
                    state="awaiting_payment",
                    checkout_url="https://example.test/pay",
                )
            )
        ),
        delete=AsyncMock(),
    )

    def callback(_):
        raise RuntimeError("callback failed")

    async def async_cb(_):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        asyncio.run(
            numbers.create(
                client=cast(AsyncMobilerun, NS(numbers=endpoint)),
                billing=Billing.PAID,
                on_checkout=async_cb if async_callback else callback,
            )
        )
    endpoint.delete.assert_awaited_once_with("number")
