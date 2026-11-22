import asyncio
from types import SimpleNamespace

import pytest

from mobilerun.clients.portal.state_fetch import fetch_state_with_retry
from mobilerun.errors import ExpectationInconclusive, StateFetchFailed
from mobilerun.errors.base import DeviceError


def test_inconclusive_verdict_keeps_sensitive_details_out_of_message():
    task = SimpleNamespace(answer="secret-model-output")
    error = ExpectationInconclusive("secret-instruction", task)
    assert "secret" not in str(error)
    assert error.condition == "secret-instruction"
    assert error.task is task
    assert isinstance(error, DeviceError)


def test_portal_error_does_not_expose_device_state(caplog):
    async def fetch():
        return {"error": "secret-error", "phone_state": "secret-state"}

    with pytest.raises(StateFetchFailed) as caught:
        asyncio.run(fetch_state_with_retry(fetch, max_retries=1, retry_delays=[0]))
    assert "secret" not in str(caught.value)
    assert "secret" not in caplog.text
