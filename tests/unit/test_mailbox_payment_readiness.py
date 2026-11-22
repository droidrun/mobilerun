import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.mailbox import CloudMailbox, cloud
from mobilerun.errors import PaymentRequired
from mobilerun.models.checkout import Checkout


def _data(status, address=None):
    return SimpleNamespace(
        data=SimpleNamespace(
            status=status, address=address, checkout_url=None, checkout_expires_at=None
        )
    )


async def _payment_error(monkeypatch):
    checkout = Checkout("https://checkout.test", None, "mailbox", "mailbox-id")
    monkeypatch.setattr(
        cloud.mailboxes,
        "create",
        AsyncMock(side_effect=PaymentRequired(checkout, AsyncMock())),
    )
    with pytest.raises(PaymentRequired) as caught:
        await CloudMailbox(billing="paid")
    return caught.value


def test_resume_waits_for_active_address(monkeypatch):
    async def run():
        error = await _payment_error(monkeypatch)
        retrieve = AsyncMock(
            side_effect=[_data("active"), _data("active", "ready@example.test")]
        )
        monkeypatch.setattr(cloud.mailboxes, "retrieve", retrieve)
        result = await error.wait_until_paid(timeout=2)
        assert result.address == "ready@example.test"
        assert retrieve.await_count == 2

    asyncio.run(run())


def test_resume_timeout_keeps_mailbox(monkeypatch):
    async def run():
        error = await _payment_error(monkeypatch)
        monkeypatch.setattr(
            cloud.mailboxes, "retrieve", AsyncMock(return_value=_data("active"))
        )
        delete = AsyncMock()
        monkeypatch.setattr(cloud.mailboxes, "delete", delete)
        with pytest.raises(TimeoutError, match="mailbox-id"):
            await error.wait_until_paid(timeout=0.01)
        delete.assert_not_awaited()

    asyncio.run(run())


def test_resume_terminal_state_is_clear(monkeypatch):
    async def run():
        error = await _payment_error(monkeypatch)
        monkeypatch.setattr(
            cloud.mailboxes, "retrieve", AsyncMock(return_value=_data("archived"))
        )
        with pytest.raises(RuntimeError, match="archived"):
            await error.wait_until_paid(timeout=1)

    asyncio.run(run())


def test_resume_payment_and_address_share_one_timeout(monkeypatch):
    async def run():
        error = await _payment_error(monkeypatch)
        retrieve = AsyncMock(side_effect=[_data("awaiting_payment"), _data("active")])
        monkeypatch.setattr(cloud.mailboxes, "retrieve", retrieve)
        started = asyncio.get_running_loop().time()
        with pytest.raises(TimeoutError, match="mailbox-id"):
            await error.wait_until_paid(timeout=1.1)
        assert retrieve.await_count == 2
        assert asyncio.get_running_loop().time() - started < 1.8

    asyncio.run(run())
