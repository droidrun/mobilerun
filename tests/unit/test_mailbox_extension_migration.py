import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mobilerun.core.mailbox import CloudMailbox
from tests.support.devices import MemoryDevice
from tests.support.extensions import check_extension_contract


def test_mailbox_extension_contract():
    async def device():
        return MemoryDevice()

    asyncio.run(
        check_extension_contract(lambda name: CloudMailbox("id", name=name), device)
    )


def test_mailbox_code_and_context():
    async def run():
        mailbox = CloudMailbox("id")
        mailbox._address = "test@example.com"
        await mailbox.bind(MemoryDevice())
        with patch(
            "mobilerun.core.mailbox.cloud.mailboxes.otp",
            AsyncMock(
                return_value=SimpleNamespace(data=SimpleNamespace(code="123456"))
            ),
        ):
            assert await mailbox.wait_for_code() == "123456"
        assert "test@example.com" in repr(await mailbox.context())

    asyncio.run(run())
