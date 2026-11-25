import asyncio
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from mobilerun.clients.cloud import devices
from mobilerun.core.devices import CloudDevice
from mobilerun.models import Billing, Plan
from tests.support import smoke_phone

NAME = "smoke run-123"


def phone(id, name=NAME, billing="minute"):
    return NS(id=id, name=name, billing_strategy=billing)


@pytest.fixture
def inventory(monkeypatch):
    listed, terminate = AsyncMock(), AsyncMock()
    monkeypatch.setattr(devices, "list_devices", listed)
    monkeypatch.setattr(devices, "terminate", terminate)
    monkeypatch.setattr(smoke_phone, "cloud_client", lambda **kw: nullcontext(None))
    monkeypatch.setattr(smoke_phone.asyncio, "sleep", AsyncMock())
    return listed, terminate


def lost_phone():
    return NS(cloud_id=None, name=None, close=AsyncMock())


def test_single_exact_minute_match_terminated_and_verified(inventory):
    listed, terminate = inventory
    other = phone("other", NAME + "-other")
    listed.side_effect = [[other], [other, phone("owned")], [other]]
    asyncio.run(smoke_phone.close_smoke_phone(lost_phone(), NAME))
    terminate.assert_awaited_once_with("owned", billing=Billing.INCLUDED, client=None)
    assert listed.await_count == 3


@pytest.mark.parametrize(
    "matches",
    [
        [phone("one"), phone("two")],
        [phone("one"), phone("two", billing="subscription")],
        [phone("subscription-id", billing="subscription")],
    ],
)
def test_unsafe_matches_never_deleted(inventory, matches):
    listed, terminate = inventory
    listed.return_value = matches
    with pytest.raises(AssertionError) as caught:
        asyncio.run(smoke_phone.close_smoke_phone(lost_phone(), NAME))
    assert NAME in str(caught.value)
    assert all(item.id in str(caught.value) for item in matches)
    terminate.assert_not_awaited()


def test_readiness_failure_rolled_back_without_scanning(monkeypatch, inventory):
    listed, terminate = inventory
    failure = TimeoutError("readiness failed")
    monkeypatch.setattr(devices, "create", AsyncMock(return_value=phone("owned")))
    monkeypatch.setattr(devices, "wait_ready", AsyncMock(side_effect=failure))

    async def exercise():
        device = CloudDevice(
            api_key="fake", billing=Billing.PAID, plan=Plan.MINUTE, name=NAME
        )
        try:
            await device
        finally:
            await smoke_phone.close_smoke_phone(device, NAME)

    with pytest.raises(TimeoutError) as caught:
        asyncio.run(exercise())
    assert caught.value is failure and not getattr(failure, "__notes__", [])
    listed.assert_not_awaited()
    terminate.assert_awaited_once()


def test_missing_phone_preserves_create_failure_with_unresolved_note(inventory):
    listed, terminate = inventory
    listed.return_value = []
    failure = TimeoutError("create response lost")

    async def exercise():
        try:
            raise failure
        finally:
            await smoke_phone.close_smoke_phone(lost_phone(), NAME)

    with pytest.raises(TimeoutError) as caught:
        asyncio.run(exercise())
    assert caught.value is failure
    assert any(
        f"unresolved: phone named {NAME!r} may exist" in note
        for note in failure.__notes__
    )
    assert 1 < listed.await_count <= 30
    terminate.assert_not_awaited()
