"""Read existing mailbox history without creating or modifying resources."""

import asyncio

import pytest

from mobilerun.clients.cloud import mailboxes

pytestmark = pytest.mark.smoke


def test_existing_mailbox_messages(real_env):
    async def exercise():
        existing = await mailboxes.list_mailboxes(billing="no_commerce")
        if not existing:
            pytest.skip("No existing mailbox available")
        result = await mailboxes.messages(existing[0].id, billing="no_commerce")
        assert isinstance(result, list)

    asyncio.run(exercise())
