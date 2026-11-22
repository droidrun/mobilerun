import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from mobilerun.utils.timestamps import to_api_timestamp


def test_utc_timestamp_uses_z_suffix():
    assert to_api_timestamp(datetime(2025, 1, 2, tzinfo=timezone.utc)).endswith("Z")


def test_offset_timestamp_converts_to_utc_z():
    value = datetime(2025, 1, 2, 3, 0, tzinfo=timezone(timedelta(hours=2)))
    assert to_api_timestamp(value) == "2025-01-02T01:00:00+00:00".replace("+00:00", "Z")


def test_naive_timestamp_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        to_api_timestamp(datetime(2025, 1, 2))


def test_mailbox_wait_for_code_passes_api_timestamp_string():
    from mobilerun.core.mailbox.cloud import CloudMailbox

    with patch(
        "mobilerun.core.mailbox.cloud.mailboxes.otp", new_callable=AsyncMock
    ) as otp:
        otp.return_value = None
        mailbox = CloudMailbox("mailbox-id")
        with patch(
            "mobilerun.core.mailbox.cloud.wait_for_code", new_callable=AsyncMock
        ) as wait:

            async def poll(fetch, **kwargs):
                await fetch(None)
                return "1234"

            wait.side_effect = poll
            wait.return_value = "1234"

            async def exercise() -> str:
                return await mailbox.wait_for_code(timeout=1)

            asyncio.run(exercise())
        assert otp.await_args is not None
        after = otp.await_args.kwargs["after"]
        assert isinstance(after, str)
        assert after.endswith("Z")
