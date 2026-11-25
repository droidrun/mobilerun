"""Offline resource demos: billing consent, ownership and cleanup."""

import asyncio
from argparse import Namespace
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from tests.demos import mailbox_demo, phone_demo, proxy_demo, task_demo


def test_payment_checkout_is_printed(monkeypatch, capsys):
    from mobilerun.errors import PaymentRequired
    from mobilerun.models.checkout import Checkout

    error = PaymentRequired(
        Checkout("https://checkout.test", None, "mailbox", "id"), AsyncMock()
    )
    monkeypatch.setattr(mailbox_demo, "CloudMailbox", AsyncMock(side_effect=error))
    asyncio.run(
        mailbox_demo.main(
            Namespace(payment_timeout=600, yes=True, keep=False, timeout=1)
        )
    )
    assert "https://checkout.test" in capsys.readouterr().out


def test_declines_before_creating(monkeypatch):
    create = AsyncMock()
    monkeypatch.setattr(mailbox_demo, "CloudMailbox", create)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    asyncio.run(
        mailbox_demo.main(
            Namespace(payment_timeout=600, yes=False, keep=False, timeout=180)
        )
    )
    create.assert_not_called()


def test_confirmation_accepts_y(monkeypatch, capsys):
    mailbox = SimpleNamespace(
        id="box-1",
        address="test@example.com",
        wait_for_code=AsyncMock(return_value="1234"),
        release=AsyncMock(),
    )
    monkeypatch.setattr(mailbox_demo, "CloudMailbox", AsyncMock(return_value=mailbox))

    def confirm(prompt):
        print(prompt)
        return "y"

    monkeypatch.setattr("builtins.input", confirm)
    asyncio.run(
        mailbox_demo.main(
            Namespace(payment_timeout=600, yes=False, keep=False, timeout=1)
        )
    )
    assert (
        "This may use paid cloud services. Continue? [y/N]" in capsys.readouterr().out
    )
    mailbox.release.assert_awaited_once()


def test_yes_receives_code_and_releases(monkeypatch, capsys):
    mailbox = SimpleNamespace(
        id="box-1",
        address="test@example.com",
        wait_for_code=AsyncMock(return_value="123456"),
        release=AsyncMock(),
    )
    create = AsyncMock(return_value=mailbox)
    monkeypatch.setattr(mailbox_demo, "CloudMailbox", create)
    asyncio.run(
        mailbox_demo.main(
            Namespace(payment_timeout=600, yes=True, keep=False, timeout=25)
        )
    )
    create.assert_called_once_with(
        billing="paid", keep=False, checkout="print", payment_timeout=600
    )
    assert mailbox.wait_for_code.await_args.kwargs["timeout"] == 25
    mailbox.release.assert_awaited_once()
    output = capsys.readouterr().out
    assert "test@example.com" in output and "123456" in output
    assert "Released mailbox box-1." in output


def test_timeout_still_releases(monkeypatch):
    mailbox = SimpleNamespace(
        id="box-1",
        address="test@example.com",
        wait_for_code=AsyncMock(side_effect=TimeoutError),
        release=AsyncMock(),
    )
    monkeypatch.setattr(mailbox_demo, "CloudMailbox", AsyncMock(return_value=mailbox))
    with pytest.raises(TimeoutError):
        asyncio.run(
            mailbox_demo.main(
                Namespace(payment_timeout=600, yes=True, keep=False, timeout=1)
            )
        )
    mailbox.release.assert_awaited_once()


def test_keep_does_not_release_mailbox(monkeypatch):
    mailbox = SimpleNamespace(
        id="box-1",
        address="test@example.com",
        wait_for_code=AsyncMock(return_value="1234"),
        release=AsyncMock(),
    )
    create = AsyncMock(return_value=mailbox)
    monkeypatch.setattr(mailbox_demo, "CloudMailbox", create)
    asyncio.run(
        mailbox_demo.main(
            Namespace(payment_timeout=600, yes=True, keep=True, timeout=1)
        )
    )
    create.assert_called_once_with(
        billing="paid", keep=True, checkout="print", payment_timeout=600
    )
    mailbox.release.assert_not_awaited()


def test_real_unbound_mailbox_waits_for_code(monkeypatch):
    from mobilerun.core.mailbox import cloud

    mailbox = cloud.CloudMailbox("box-1", billing="paid")
    mailbox._address = "test@example.com"
    fetch = AsyncMock(return_value="1234")
    monkeypatch.setattr(cloud, "wait_for_code", fetch)
    monkeypatch.setattr(mailbox_demo, "CloudMailbox", AsyncMock(return_value=mailbox))
    release = AsyncMock()
    monkeypatch.setattr(mailbox, "release", release)
    asyncio.run(
        mailbox_demo.main(
            Namespace(payment_timeout=600, yes=True, keep=False, timeout=1)
        )
    )
    fetch.assert_awaited_once()
    release.assert_awaited_once()


def test_included_uses_quota_and_passes_billing(monkeypatch):
    mailbox = SimpleNamespace(
        id="box",
        address="example",
        wait_for_code=AsyncMock(return_value="1234"),
        release=AsyncMock(),
    )
    create = AsyncMock(return_value=mailbox)
    monkeypatch.setattr(mailbox_demo, "CloudMailbox", create)
    prompts = []
    monkeypatch.setattr("builtins.input", lambda text: prompts.append(text) or "y")
    asyncio.run(
        mailbox_demo.main(
            Namespace(
                yes=False,
                billing="included",
                payment_timeout=600,
                keep=False,
                timeout=1,
            )
        )
    )
    assert create.await_args.kwargs["billing"] == "included"
    assert "included quota only (no charges)" in prompts[0]
    assert "paid" not in prompts[0].lower()


def phone_args(**kwargs):
    return Namespace(
        yes=kwargs.get("yes", True),
        country=kwargs.get("country", None),
        keep=kwargs.get("keep", False),
        bind=kwargs.get("bind", False),
        timeout=30,
    )


@pytest.fixture
def phone_setup(monkeypatch):
    number = SimpleNamespace(
        id="number-1",
        phone_number="+15555550100",
        wait_ready=AsyncMock(),
        wait_for_code=AsyncMock(return_value="123456"),
        bind=AsyncMock(),
        unbind=AsyncMock(),
        release=AsyncMock(),
    )
    create = AsyncMock(return_value=number)
    monkeypatch.setattr(phone_demo, "CloudPhoneNumber", create)
    return number, create


def test_happy_path(phone_setup, capsys):
    number, create = phone_setup
    asyncio.run(phone_demo.main(phone_args()))
    create.assert_awaited_once_with(
        country="US", billing="paid", checkout="print", keep=False
    )
    number.wait_ready.assert_awaited_once()
    number.wait_for_code.assert_awaited_once_with(timeout=30)
    number.release.assert_awaited_once()
    assert "Send an SMS with a code to +15555550100" in capsys.readouterr().out


def test_pending_checkout_is_printed(phone_setup, capsys):
    from mobilerun.errors import PurchaseInProgress
    from mobilerun.models.checkout import Checkout

    _, create = phone_setup
    create.side_effect = PurchaseInProgress(
        "id", Checkout("https://checkout.test", None, "phone number", "id")
    )
    asyncio.run(phone_demo.main(phone_args()))
    assert "https://checkout.test" in capsys.readouterr().out


def test_decline_avoids_paid_creation(phone_setup, monkeypatch):
    number, create = phone_setup
    monkeypatch.setattr("builtins.input", lambda _: "n")
    asyncio.run(phone_demo.main(phone_args(yes=False)))
    create.assert_not_called()


def test_timeout_releases_number(phone_setup):
    number, _ = phone_setup
    number.wait_for_code.side_effect = TimeoutError
    with pytest.raises(TimeoutError):
        asyncio.run(phone_demo.main(phone_args()))
    number.release.assert_awaited_once()


def test_keep_does_not_release_phone(phone_setup, capsys):
    number, create = phone_setup
    asyncio.run(phone_demo.main(phone_args(keep=True, country="DE")))
    create.assert_awaited_once_with(
        country="DE", billing="paid", checkout="print", keep=True
    )
    number.release.assert_not_awaited()
    assert "Kept number" in capsys.readouterr().out


def test_bind_unbinds_before_phone_closes(phone_setup, monkeypatch, capsys):
    number, _ = phone_setup
    events = []
    phone = SimpleNamespace(number=number, sms=number, device_id="phone")

    async def unbind():
        events.append("unbind")

    number.unbind.side_effect = unbind

    class Phone:
        def __init__(self, **kwargs):
            assert kwargs == {"billing": "paid", "plan": None}

        async def __aenter__(self):
            return phone

        async def __aexit__(self, *args):
            events.append("close")

    monkeypatch.setattr(phone_demo, "CloudDevice", Phone)
    asyncio.run(phone_demo.main(phone_args(bind=True)))
    number.bind.assert_awaited_once_with(phone)
    assert events == ["unbind", "close"]
    number.release.assert_awaited_once()
    assert "phone.number" in capsys.readouterr().out


def test_bind_timeout_unbinds_before_phone_closes(phone_setup, monkeypatch):
    number, _ = phone_setup
    events = []
    phone = SimpleNamespace(number=number, sms=number, device_id="phone")
    number.wait_for_code.side_effect = TimeoutError

    async def unbind():
        events.append("unbind")

    number.unbind.side_effect = unbind

    class Phone:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return phone

        async def __aexit__(self, *args):
            events.append("close")

    monkeypatch.setattr(phone_demo, "CloudDevice", Phone)
    with pytest.raises(TimeoutError):
        asyncio.run(phone_demo.main(phone_args(bind=True)))
    assert events == ["unbind", "close"]
    number.release.assert_awaited_once()


def test_readiness_failure_releases_number(phone_setup):
    number, _ = phone_setup
    number.wait_ready.side_effect = TimeoutError
    with pytest.raises(TimeoutError):
        asyncio.run(phone_demo.main(phone_args()))
    number.wait_for_code.assert_not_awaited()
    number.release.assert_awaited_once()


def test_included_billing_is_used_for_number(phone_setup, capsys):
    number, create = phone_setup
    options = phone_args()
    options.billing = "included"
    options.plan = None
    asyncio.run(phone_demo.main(options))
    assert create.await_args.kwargs["billing"] == "included"
    assert "Billing: included" in capsys.readouterr().out


def test_included_binding_uses_subscription_and_quota_warning(
    phone_setup, monkeypatch, capsys
):
    number, _ = phone_setup
    options = phone_args(bind=True, yes=False)
    options.billing = "included"
    options.plan = "subscription"
    prompts = []
    monkeypatch.setattr("builtins.input", lambda text: prompts.append(text) or "y")
    phone = SimpleNamespace(number=number, sms=number, device_id="phone")

    class Phone:
        def __init__(self, **kwargs):
            assert kwargs == {"billing": "included", "plan": "subscription"}

        async def __aenter__(self):
            return phone

        async def __aexit__(self, *unused):
            pass

    monkeypatch.setattr(phone_demo, "CloudDevice", Phone)
    asyncio.run(phone_demo.main(options))
    assert "included quota only (no charges)" in prompts[0]
    assert "paid" not in prompts[0].lower()
    assert "phone plan: subscription" in capsys.readouterr().out


def test_included_minute_rejected_before_purchase(phone_setup):
    _, create = phone_setup
    options = phone_args(bind=True)
    options.billing = "included"
    options.plan = "minute"
    with pytest.raises(ValueError, match="subscription"):
        asyncio.run(phone_demo.main(options))
    create.assert_not_called()


@pytest.mark.parametrize("number_plan", ["auto", "included", "rent"])
def test_number_plan_cli_reaches_constructor(
    phone_setup, monkeypatch, capsys, number_plan
):
    _, create = phone_setup
    monkeypatch.setattr(
        "sys.argv", ["phone_demo", "--yes", "--number-plan", number_plan]
    )
    options = phone_demo.parse_args()
    asyncio.run(phone_demo.main(options))
    assert create.await_args.kwargs["plan"] == number_plan
    assert f"number plan: {number_plan}" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--phone-plan", "--plan"])
def test_phone_plan_cli_alias(monkeypatch, flag):
    monkeypatch.setattr("sys.argv", ["phone_demo", "--bind", flag, "minute"])
    assert phone_demo.parse_args().phone_plan == "minute"


@pytest.fixture
def proxy_setup(monkeypatch):
    phone = SimpleNamespace(
        proxy_status=AsyncMock(
            return_value=SimpleNamespace(connected=True, name="old")
        ),
        set_proxy=AsyncMock(return_value=SimpleNamespace(connected=True, name="new")),
        close=AsyncMock(),
        stop=AsyncMock(),
        terminate=AsyncMock(),
    )
    phone.proxy = SimpleNamespace(status=phone.proxy_status)
    monkeypatch.setattr(
        proxy_demo.proxies,
        "device_status",
        AsyncMock(side_effect=AssertionError("bypass")),
    )

    async def bind(proxy, device, **kwargs):
        assert kwargs == {"force": True}
        await device.set_proxy(proxy)
        device.proxy = proxy
        return proxy

    monkeypatch.setattr(proxy_demo.Proxy, "bind", bind)

    async def status(proxy):
        return SimpleNamespace(connected=True, name="new")

    monkeypatch.setattr(proxy_demo.Proxy, "status", status)
    device = AsyncMock(return_value=phone)
    monkeypatch.setattr(proxy_demo, "CloudDevice", device)
    monkeypatch.setattr(
        proxy_demo.devices,
        "list_devices",
        AsyncMock(return_value=[SimpleNamespace(name="tester1", id="phone")]),
    )
    monkeypatch.setattr(
        proxy_demo.proxies,
        "list_proxies",
        AsyncMock(
            return_value=SimpleNamespace(
                items=[
                    SimpleNamespace(
                        id="old-id", name="old", country="US", status="active"
                    ),
                    SimpleNamespace(
                        id="new-id", name="new", country="DE", status="active"
                    ),
                ],
                pagination=SimpleNamespace(has_next=False),
            )
        ),
    )
    return phone, device


def run_proxy_demo(*args):
    return asyncio.run(proxy_demo.main(list(args)))


def test_selects_different_existing_proxy_and_closes_without_stopping(
    proxy_setup, capsys
):
    phone, device = proxy_setup
    assert run_proxy_demo() == 0
    device.assert_awaited_once_with(device_id="phone", billing=proxy_demo.Billing.PAID)
    assert phone.set_proxy.await_args.args[0].id == "new-id"
    phone.close.assert_awaited_once()
    phone.stop.assert_not_awaited()
    phone.terminate.assert_not_awaited()
    assert "old" in capsys.readouterr().out


def test_missing_phone_lists_names(proxy_setup, capsys):
    assert run_proxy_demo("--name", "missing") == 1
    assert "tester1" in capsys.readouterr().err
    proxy_setup[1].assert_not_awaited()


def test_explicit_proxy_id(proxy_setup):
    assert run_proxy_demo("--proxy-id", "new-id") == 0
    assert proxy_setup[0].set_proxy.await_args.args[0].id == "new-id"


def test_no_alternative(proxy_setup, capsys):
    cast(AsyncMock, proxy_demo.proxies.list_proxies).return_value.items.pop()
    assert run_proxy_demo() == 1
    assert "alternative" in capsys.readouterr().err.lower()
    proxy_setup[0].set_proxy.assert_not_awaited()
    proxy_setup[0].close.assert_awaited_once()


def test_country_decline_never_switches(proxy_setup, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert run_proxy_demo("--country", "DE") == 1
    proxy_setup[1].assert_not_awaited()


def test_country_yes_uses_paid(proxy_setup):
    assert run_proxy_demo("--country", "DE", "--yes") == 0
    proxy_setup[1].assert_awaited_once_with(
        device_id="phone", billing=proxy_demo.Billing.PAID
    )
    assert proxy_setup[0].set_proxy.await_args.args[0].country == "DE"


def test_unknown_current_proxy_fails_safely(proxy_setup, capsys):
    proxy_setup[0].proxy_status.return_value.name = None
    assert run_proxy_demo() == 1
    assert "identify" in capsys.readouterr().err.lower()
    proxy_setup[0].set_proxy.assert_not_awaited()


def test_proxy_not_active_reports_error_and_closes(proxy_setup, capsys):
    proxy_setup[0].set_proxy.side_effect = proxy_demo.ProxyNotActive("disconnected")
    assert run_proxy_demo() == 1
    assert "not active" in capsys.readouterr().err
    proxy_setup[0].close.assert_awaited_once()


def test_demo_creates_minute_phone_and_closes(monkeypatch, capsys):
    proxy_status = AsyncMock(return_value=SimpleNamespace(connected=False))
    proxy = SimpleNamespace(country="US", status=proxy_status)
    monkeypatch.setattr(task_demo, "CloudRotatingProxy", lambda **kwargs: proxy)
    phone = SimpleNamespace(
        device_id="phone",
        proxy=SimpleNamespace(
            status=AsyncMock(side_effect=AssertionError("Use proxy handle"))
        ),
        info=AsyncMock(return_value=SimpleNamespace()),
        do=AsyncMock(
            side_effect=[
                SimpleNamespace(task_id="open"),
                SimpleNamespace(
                    task_id="version", output=task_demo.AndroidVersion(version="15")
                ),
                SimpleNamespace(task_id="timed"),
            ]
        ),
        expect=AsyncMock(
            return_value=SimpleNamespace(task=SimpleNamespace(task_id="expect"))
        ),
        press_button=AsyncMock(),
    )

    class Device:
        def __init__(self, **kwargs):
            proxy = kwargs.pop("proxy")
            assert proxy is not None and proxy.country == "US"
            assert kwargs == {
                "name": "cloud-phone-demo",
                "billing": "paid",
                "plan": None,
            }

        async def __aenter__(self):
            return phone

        async def __aexit__(self, *args):
            phone.closed = True

    monkeypatch.setattr(task_demo, "CloudDevice", Device)
    asyncio.run(task_demo.main(yes=True))
    assert phone.closed
    phone.info.assert_not_awaited()
    proxy_status.assert_awaited_once()
    phone.press_button.assert_awaited_once_with("home")
    assert phone.do.await_count == 3
    assert "15" in capsys.readouterr().out


def test_task_demo_decline_does_not_create(monkeypatch):
    from unittest.mock import Mock

    create = Mock()
    monkeypatch.setattr(task_demo, "CloudDevice", create)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    asyncio.run(task_demo.main(yes=False))
    create.assert_not_called()


def test_included_plan_passed_to_phone_and_warns(monkeypatch):
    from mobilerun.models import Billing

    calls = []
    prompts = []
    monkeypatch.setattr("builtins.input", lambda text: prompts.append(text) or "y")

    def phone(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("stop before creation")

    monkeypatch.setattr(task_demo, "CloudDevice", phone)
    with pytest.raises(RuntimeError, match="stop before creation"):
        asyncio.run(task_demo.main(billing=Billing.INCLUDED, plan="subscription"))
    assert calls[0]["billing"] == "included"
    assert calls[0]["plan"] == "subscription"
    assert "included quota only (no charges)" in prompts[0]
    assert "paid" not in prompts[0].lower()


def test_included_minute_is_rejected_before_creation(monkeypatch):
    from mobilerun.models import Billing

    create = AsyncMock()
    monkeypatch.setattr(task_demo, "CloudDevice", create)
    with pytest.raises(ValueError, match="subscription"):
        asyncio.run(task_demo.main(yes=True, billing=Billing.INCLUDED, plan="minute"))
    create.assert_not_called()
