"""Awaitable resource ownership never spills onto passed IDs."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.mailbox import CloudMailbox as Mailbox
from mobilerun.core.phone_number import CloudPhoneNumber as PhoneNumber


@pytest.mark.parametrize("resource", [PhoneNumber, Mailbox])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("keep", [False, True])
def test_context_ownership_and_idempotent_close(resource, existing, keep):
    adapter = NS(
        create=AsyncMock(
            return_value=NS(
                data=NS(
                    number_id="new",
                    id="new",
                    address="demo@example.test",
                    state="provisioning",
                    status="active",
                    checkout_url=None,
                )
            )
        ),
        retrieve=AsyncMock(
            return_value=NS(
                data=NS(
                    device_binding=None,
                    phone_number=None,
                    address="demo@example.test",
                    state="active",
                    status="active",
                    checkout_url=None,
                )
            )
        ),
        delete=AsyncMock(),
    )

    async def exercise():
        instance = resource(
            "existing" if existing else None,
            client=NS(
                **{"numbers" if resource is PhoneNumber else "mailboxes": adapter}
            ),
            billing="paid",
            keep=keep,
        )
        async with instance as ready:
            assert ready is instance
            assert instance.id == ("existing" if existing else "new")
        await instance.close()

    asyncio.run(exercise())
    assert adapter.create.await_count == (0 if existing else 1)
    assert adapter.delete.await_count == (1 if not existing and not keep else 0)


@pytest.mark.parametrize("resource", [PhoneNumber, Mailbox])
def test_unawaited_creation_cannot_be_used(resource):
    instance = resource()
    with pytest.raises(RuntimeError, match="Await"):
        _ = instance.id
    with pytest.raises(RuntimeError, match="Await"):
        asyncio.run(instance.release())
    assert not hasattr(resource, "create")


@pytest.mark.parametrize("resource", [PhoneNumber, Mailbox])
def test_creation_failure_does_not_claim_ownership(resource):
    adapter = NS(
        create=AsyncMock(side_effect=ValueError("blocked")), delete=AsyncMock()
    )

    async def exercise():
        instance = resource(
            client=NS(
                **{"numbers" if resource is PhoneNumber else "mailboxes": adapter}
            ),
            billing="paid",
        )
        with pytest.raises(ValueError, match="blocked"):
            await instance
        await instance.close()

    asyncio.run(exercise())
    adapter.delete.assert_not_awaited()
