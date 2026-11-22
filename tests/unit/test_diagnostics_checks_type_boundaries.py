import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun.diagnostics import checks
from mobilerun.diagnostics.results import Status


def test_device_without_serial_uses_adb_default(monkeypatch):
    device = SimpleNamespace(
        serial="device-1", get_state=AsyncMock(return_value="device")
    )
    lookup = AsyncMock(return_value=device)
    monkeypatch.setattr(checks.adb, "device", lookup)
    result, selected = asyncio.run(checks.check_device(None, False))
    assert result.status == Status.PASS
    assert selected is device
    lookup.assert_awaited_once_with()


@pytest.mark.parametrize(
    "check, status",
    [
        (checks.check_content_provider, Status.FAIL),
        (checks.check_keyboard, Status.WARN),
    ],
)
def test_non_text_adb_response_is_reported_as_failure(check, status):
    result = asyncio.run(
        check(SimpleNamespace(shell=AsyncMock(return_value=b"unexpected")), True)
    )
    assert result.status == status
    assert "non-text output" in result.message
