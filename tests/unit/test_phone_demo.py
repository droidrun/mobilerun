import asyncio
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def args(**kwargs):
    return Namespace(
        yes=kwargs.get("yes", True),
        country=kwargs.get("country", None),
        keep=kwargs.get("keep", False),
        bind=kwargs.get("bind", False),
        timeout=30,
    )


@pytest.fixture
def setup_demo(monkeypatch):
    from tests.demos import phone_demo as demo

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
    monkeypatch.setattr(demo, "CloudPhoneNumber", create)
    return demo, number, create


def test_happy_path(setup_demo, capsys):
    demo, number, create = setup_demo
    asyncio.run(demo.main(args()))
    create.assert_awaited_once_with(
        country="US", billing="paid", checkout="print", keep=False
    )
    number.wait_ready.assert_awaited_once()
    number.wait_for_code.assert_awaited_once_with(timeout=30)
    number.release.assert_awaited_once()
    assert "Send an SMS with a code to +15555550100" in capsys.readouterr().out


def test_pending_checkout_is_printed(setup_demo, capsys):
    from mobilerun.errors import PurchaseInProgress
    from mobilerun.models.checkout import Checkout

    demo, _, create = setup_demo
    create.side_effect = PurchaseInProgress(
        "id", Checkout("https://checkout.test", None, "phone number", "id")
    )
    asyncio.run(demo.main(args()))
    assert "https://checkout.test" in capsys.readouterr().out


def test_decline_avoids_paid_creation(setup_demo, monkeypatch):
    demo, number, create = setup_demo
    monkeypatch.setattr("builtins.input", lambda _: "n")
    asyncio.run(demo.main(args(yes=False)))
    create.assert_not_called()


def test_timeout_releases_number(setup_demo):
    demo, number, _ = setup_demo
    number.wait_for_code.side_effect = TimeoutError
    with pytest.raises(TimeoutError):
        asyncio.run(demo.main(args()))
    number.release.assert_awaited_once()


def test_keep_does_not_release(setup_demo, capsys):
    demo, number, create = setup_demo
    asyncio.run(demo.main(args(keep=True, country="DE")))
    create.assert_awaited_once_with(
        country="DE", billing="paid", checkout="print", keep=True
    )
    number.release.assert_not_awaited()
    assert "Kept number" in capsys.readouterr().out


def test_bind_unbinds_before_phone_closes(setup_demo, monkeypatch, capsys):
    demo, number, _ = setup_demo
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

    monkeypatch.setattr(demo, "CloudDevice", Phone)
    asyncio.run(demo.main(args(bind=True)))
    number.bind.assert_awaited_once_with(phone)
    assert events == ["unbind", "close"]
    number.release.assert_awaited_once()
    assert "phone.number" in capsys.readouterr().out


def test_bind_timeout_unbinds_before_phone_closes(setup_demo, monkeypatch):
    demo, number, _ = setup_demo
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

    monkeypatch.setattr(demo, "CloudDevice", Phone)
    with pytest.raises(TimeoutError):
        asyncio.run(demo.main(args(bind=True)))
    assert events == ["unbind", "close"]
    number.release.assert_awaited_once()


def test_readiness_failure_releases_number(setup_demo):
    demo, number, _ = setup_demo
    number.wait_ready.side_effect = TimeoutError
    with pytest.raises(TimeoutError):
        asyncio.run(demo.main(args()))
    number.wait_for_code.assert_not_awaited()
    number.release.assert_awaited_once()


def test_included_billing_is_used_for_number(setup_demo, capsys):
    demo, number, create = setup_demo
    options = args()
    options.billing = "included"
    options.plan = None
    asyncio.run(demo.main(options))
    assert create.await_args.kwargs["billing"] == "included"
    assert "Billing: included" in capsys.readouterr().out


def test_included_binding_uses_subscription_and_quota_warning(
    setup_demo, monkeypatch, capsys
):
    demo, number, _ = setup_demo
    options = args(bind=True, yes=False)
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

    monkeypatch.setattr(demo, "CloudDevice", Phone)
    asyncio.run(demo.main(options))
    assert "included quota only (no charges)" in prompts[0]
    assert "paid" not in prompts[0].lower()
    assert "phone plan: subscription" in capsys.readouterr().out


def test_included_minute_rejected_before_purchase(setup_demo):
    demo, _, create = setup_demo
    options = args(bind=True)
    options.billing = "included"
    options.plan = "minute"
    with pytest.raises(ValueError, match="subscription"):
        asyncio.run(demo.main(options))
    create.assert_not_called()


@pytest.mark.parametrize("number_plan", ["auto", "included", "rent"])
def test_number_plan_cli_reaches_constructor(
    setup_demo, monkeypatch, capsys, number_plan
):
    demo, _, create = setup_demo
    monkeypatch.setattr(
        "sys.argv", ["phone_demo", "--yes", "--number-plan", number_plan]
    )
    options = demo.parse_args()
    asyncio.run(demo.main(options))
    assert create.await_args.kwargs["plan"] == number_plan
    assert f"number plan: {number_plan}" in capsys.readouterr().out


def test_phone_plan_cli_alias(setup_demo, monkeypatch):
    demo, _, _ = setup_demo
    for flag in ("--phone-plan", "--plan"):
        monkeypatch.setattr("sys.argv", ["phone_demo", "--bind", flag, "minute"])
        assert demo.parse_args().phone_plan == "minute"
