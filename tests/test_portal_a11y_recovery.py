import asyncio

import pytest
from mobilerun_core_local.driver.android import AndroidDriver
from mobilerun_core_local.driver.android.portal import (
    PORTAL_PACKAGE_NAME,
    portal_a11y_service,
    portal_ime_id,
)

import mobilerun.tools.ui.provider as provider_module
from mobilerun.tools.ui.provider import AndroidStateProvider, fetch_state_with_retry

SERVICE_GONE = {"status": "error", "error": "Accessibility service not available"}
STATE = {"a11y_tree": {}, "phone_state": {}, "device_context": {}}


def _fetch(*responses, events):
    queue = list(responses)

    async def fetch():
        events.append("fetch")
        return queue.pop(0)

    return fetch


def _recovery(events):
    async def recovery():
        events.append("recover")

    return recovery


def test_service_not_available_recovers_after_first_failure():
    events = []
    result = asyncio.run(
        fetch_state_with_retry(
            fetch=_fetch(SERVICE_GONE, STATE, events=events),
            recovery=_recovery(events),
            retry_delays=[0],
        )
    )
    assert result is STATE
    assert events == ["fetch", "recover", "fetch"]


def test_service_not_available_recovers_only_once():
    events = []
    result = asyncio.run(
        fetch_state_with_retry(
            fetch=_fetch(SERVICE_GONE, SERVICE_GONE, STATE, events=events),
            recovery=_recovery(events),
            retry_delays=[0],
        )
    )
    assert result is STATE
    assert events == ["fetch", "recover", "fetch", "fetch"]


def test_other_errors_wait_for_recovery_after():
    events = []
    other = {"error": "Parse Error", "message": "bad"}
    result = asyncio.run(
        fetch_state_with_retry(
            fetch=_fetch(other, other, STATE, events=events),
            recovery=_recovery(events),
            retry_delays=[0],
            recovery_after=5,
        )
    )
    assert result is STATE
    assert events == ["fetch", "fetch", "fetch"]


class _Device:
    def __init__(self):
        self.commands = []

    async def shell(self, cmd, **kwargs):
        self.commands.append(cmd)
        return ""


def _provider(keyboard: bool):
    driver = object.__new__(AndroidDriver)
    driver.device = _Device()
    driver.portal = None
    driver._portal_keyboard_available = keyboard
    provider = object.__new__(AndroidStateProvider)
    provider.driver = driver
    return provider, driver.device


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def sleep(_):
        return None

    monkeypatch.setattr(provider_module.asyncio, "sleep", sleep)


def test_recover_portal_restores_portal_keyboard():
    provider, device = _provider(keyboard=True)
    asyncio.run(provider._recover_portal())

    a11y = portal_a11y_service(PORTAL_PACKAGE_NAME)
    ime = portal_ime_id(PORTAL_PACKAGE_NAME)
    assert f"settings put secure enabled_accessibility_services {a11y}" in (
        device.commands
    )
    assert device.commands[-2:] == [f"ime enable {ime}", f"ime set {ime}"]


def test_recover_portal_leaves_keyboard_alone_without_portal_keyboard():
    provider, device = _provider(keyboard=False)
    asyncio.run(provider._recover_portal())

    assert device.commands
    assert not any(cmd.startswith("ime ") for cmd in device.commands)


def test_recovery_needs_an_adb_device():
    provider, _ = _provider(keyboard=False)
    assert provider._portal_recovery() == provider._recover_portal

    provider.driver.device = None
    assert provider._portal_recovery() is None

    provider.driver = object()  # e.g. Portal HTTP driver
    assert provider._portal_recovery() is None


def test_state_read_without_recovery_does_not_report_one(caplog, monkeypatch):
    events = []
    other = {"error": "Parse Error", "message": "bad"}
    monkeypatch.setattr(provider_module.logger, "propagate", True)
    with caplog.at_level("INFO", logger="mobilerun"):
        asyncio.run(
            fetch_state_with_retry(
                fetch=_fetch(other, SERVICE_GONE, STATE, events=events),
                recovery=None,
                retry_delays=[0],
                recovery_after=1,
            )
        )
    assert events == ["fetch", "fetch", "fetch"]
    assert "get_state attempt 1 failed" in caplog.text
    assert "recover" not in caplog.text.lower()
    assert "re-enabling" not in caplog.text
