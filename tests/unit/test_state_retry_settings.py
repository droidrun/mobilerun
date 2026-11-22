import asyncio

from mobilerun.clients.portal.state_fetch import fetch_state_with_retry
from mobilerun.errors import StateFetchFailed


def test_state_retry_settings_are_read_from_environment(monkeypatch):
    monkeypatch.setenv("MOBILERUN_STATE_RETRY_DELAYS", "[0,0]")
    monkeypatch.setenv("MOBILERUN_STATE_MAX_RETRIES", "2")
    calls = 0

    async def fetch():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary")
        return {"a11y_tree": {}, "phone_state": {}, "device_context": {}}

    result = asyncio.run(fetch_state_with_retry(fetch))

    assert calls == 2
    assert result["a11y_tree"] == {}


def test_retry_exhaustion_raises_state_fetch_failed(monkeypatch):
    monkeypatch.setenv("MOBILERUN_STATE_MAX_RETRIES", "1")
    monkeypatch.setenv("MOBILERUN_STATE_RETRY_DELAYS", "[0]")

    async def fetch():
        raise RuntimeError("offline")

    try:
        asyncio.run(fetch_state_with_retry(fetch))
    except StateFetchFailed as exc:
        assert str(exc) == "Failed to get state after 1 attempts: offline"
    else:
        raise AssertionError("Expected StateFetchFailed")
