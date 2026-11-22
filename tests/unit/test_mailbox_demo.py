import asyncio
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def test_payment_checkout_is_printed(monkeypatch, capsys):
    from mobilerun.errors import PaymentRequired
    from mobilerun.models.checkout import Checkout
    from tests.demos import mailbox_demo as demo

    error = PaymentRequired(
        Checkout("https://checkout.test", None, "mailbox", "id"), AsyncMock()
    )
    monkeypatch.setattr(demo, "CloudMailbox", AsyncMock(side_effect=error))
    asyncio.run(
        demo.main(Namespace(payment_timeout=600, yes=True, keep=False, timeout=1))
    )
    assert "https://checkout.test" in capsys.readouterr().out


def test_declines_before_creating(monkeypatch):
    from tests.demos import mailbox_demo as demo

    create = AsyncMock()
    monkeypatch.setattr(demo, "CloudMailbox", create)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    asyncio.run(
        demo.main(Namespace(payment_timeout=600, yes=False, keep=False, timeout=180))
    )
    create.assert_not_called()


def test_confirmation_accepts_y(monkeypatch, capsys):
    from tests.demos import mailbox_demo as demo

    mailbox = SimpleNamespace(
        id="box-1",
        address="test@example.com",
        wait_for_code=AsyncMock(return_value="1234"),
        release=AsyncMock(),
    )
    monkeypatch.setattr(demo, "CloudMailbox", AsyncMock(return_value=mailbox))

    def confirm(prompt):
        print(prompt)
        return "y"

    monkeypatch.setattr("builtins.input", confirm)
    asyncio.run(
        demo.main(Namespace(payment_timeout=600, yes=False, keep=False, timeout=1))
    )
    assert (
        "This may use paid cloud services. Continue? [y/N]" in capsys.readouterr().out
    )
    mailbox.release.assert_awaited_once()


def test_yes_receives_code_and_releases(monkeypatch, capsys):
    from tests.demos import mailbox_demo as demo

    mailbox = SimpleNamespace(
        id="box-1",
        address="test@example.com",
        wait_for_code=AsyncMock(return_value="123456"),
        release=AsyncMock(),
    )
    create = AsyncMock(return_value=mailbox)
    monkeypatch.setattr(demo, "CloudMailbox", create)
    asyncio.run(
        demo.main(Namespace(payment_timeout=600, yes=True, keep=False, timeout=25))
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
    from tests.demos import mailbox_demo as demo

    mailbox = SimpleNamespace(
        id="box-1",
        address="test@example.com",
        wait_for_code=AsyncMock(side_effect=TimeoutError),
        release=AsyncMock(),
    )
    monkeypatch.setattr(demo, "CloudMailbox", AsyncMock(return_value=mailbox))
    with pytest.raises(TimeoutError):
        asyncio.run(
            demo.main(Namespace(payment_timeout=600, yes=True, keep=False, timeout=1))
        )
    mailbox.release.assert_awaited_once()


def test_keep_does_not_release(monkeypatch):
    from tests.demos import mailbox_demo as demo

    mailbox = SimpleNamespace(
        id="box-1",
        address="test@example.com",
        wait_for_code=AsyncMock(return_value="1234"),
        release=AsyncMock(),
    )
    create = AsyncMock(return_value=mailbox)
    monkeypatch.setattr(demo, "CloudMailbox", create)
    asyncio.run(
        demo.main(Namespace(payment_timeout=600, yes=True, keep=True, timeout=1))
    )
    create.assert_called_once_with(
        billing="paid", keep=True, checkout="print", payment_timeout=600
    )
    mailbox.release.assert_not_awaited()


def test_real_unbound_mailbox_waits_for_code(monkeypatch):
    from mobilerun.core.mailbox import cloud
    from tests.demos import mailbox_demo as demo

    mailbox = cloud.CloudMailbox("box-1", billing="paid")
    mailbox._address = "test@example.com"
    fetch = AsyncMock(return_value="1234")
    monkeypatch.setattr(cloud, "wait_for_code", fetch)
    monkeypatch.setattr(demo, "CloudMailbox", AsyncMock(return_value=mailbox))
    release = AsyncMock()
    monkeypatch.setattr(mailbox, "release", release)
    asyncio.run(
        demo.main(Namespace(payment_timeout=600, yes=True, keep=False, timeout=1))
    )
    fetch.assert_awaited_once()
    release.assert_awaited_once()


def test_included_uses_quota_and_passes_billing(monkeypatch):
    from tests.demos import mailbox_demo as demo

    mailbox = SimpleNamespace(
        id="box",
        address="example",
        wait_for_code=AsyncMock(return_value="1234"),
        release=AsyncMock(),
    )
    create = AsyncMock(return_value=mailbox)
    monkeypatch.setattr(demo, "CloudMailbox", create)
    prompts = []
    monkeypatch.setattr("builtins.input", lambda text: prompts.append(text) or "y")
    asyncio.run(
        demo.main(
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
