import asyncio
import functools
import json
from contextlib import contextmanager

import pytest

from mobilerun.cli import main

A11Y_SERVICE = (
    "com.mobilerun.portal/com.mobilerun.portal.service.MobilerunAccessibilityService"
)


def provider_row(**result):
    return f"Row: 0 result={json.dumps(result, separators=(',', ':'))}"


class FakePortalDevice:
    """Tracks the accessibility settings Portal setup reads and writes."""

    serial = "emulator-5554"

    def __init__(
        self, *, drop_first_enable=False, responds=True, ready_after=1, screen_on=True
    ):
        self.drop_first_enable = drop_first_enable
        self.responds = responds
        self.ready_after = ready_after
        self.screen_on = screen_on
        self.queries = 0
        self.installs = []
        self.shell_calls = []
        self.enable_writes = 0
        self.services = "null"
        self.a11y_enabled = "0"

    @property
    def service_enabled(self):
        return A11Y_SERVICE in self.services and self.a11y_enabled == "1"

    async def install(self, path, **kwargs):
        self.installs.append((path, kwargs))

    async def shell(self, cmd):
        self.shell_calls.append(cmd)
        if cmd.startswith("settings put secure enabled_accessibility_services "):
            self.enable_writes += 1
            self.services = cmd.rsplit(" ", 1)[1]
        elif cmd == "settings put secure accessibility_enabled 1":
            self.a11y_enabled = "1"
            if self.drop_first_enable and self.enable_writes == 1:
                # Service enabled before the install was registered.
                self.services, self.a11y_enabled = "", "0"
        elif cmd == "settings get secure enabled_accessibility_services":
            return self.services
        elif cmd == "settings get secure accessibility_enabled":
            return self.a11y_enabled
        elif cmd.startswith("content query --uri content://com.mobilerun.portal/"):
            self.queries += 1
            bound = (
                self.responds
                and self.service_enabled
                and self.queries >= self.ready_after
            )
            if not bound:
                return provider_row(
                    status="error", error="Accessibility service not available"
                )
            if cmd.endswith("/ping"):
                return provider_row(status="success", result="pong")
            if cmd.endswith("/state") and self.screen_on:
                return provider_row(status="success", result="{}")
            return provider_row(status="error", error="Accessibility tree is empty")
        return ""

    def provider_queries(self):
        return [c for c in self.shell_calls if c.startswith("content query")]


@pytest.fixture
def apk(tmp_path):
    path = tmp_path / "portal.apk"
    path.write_bytes(b"apk")
    return str(path)


@pytest.fixture
def run_setup(monkeypatch, apk):
    @contextmanager
    def fake_download(*args, **kwargs):
        yield apk

    async def fail_setup_portal(*args, **kwargs):
        raise AssertionError("explicit APK setup must not use setup_portal")

    monkeypatch.setattr(main, "download_versioned_portal_apk", fake_download)
    monkeypatch.setattr(main, "download_portal_apk", fake_download)
    monkeypatch.setattr(main, "setup_portal", fail_setup_portal)
    monkeypatch.setattr(
        main,
        "_wait_for_portal_service",
        functools.partial(main._wait_for_portal_service, timeout=0.2, interval=0),
    )

    def run(device, **kwargs):
        async def fake_device(serial):
            return device

        monkeypatch.setattr(main.adb, "device", fake_device)
        asyncio.run(main._setup_portal(device=device.serial, debug=False, **kwargs))

    return run


@pytest.mark.parametrize("option", ["path", "portal-version", "latest"])
def test_explicit_apk_install_does_not_launch_portal(run_setup, apk, option):
    device = FakePortalDevice()
    kwargs = {
        "path": {"path": apk},
        "portal-version": {"path": None, "specific_version": "0.7.25"},
        "latest": {"path": None, "latest": True},
    }[option]

    run_setup(device, **kwargs)

    assert len(device.installs) == 1
    path, install_kwargs = device.installs[0]
    assert path == apk
    assert install_kwargs["nolaunch"] is True
    assert install_kwargs["uninstall"] is True


def test_dropped_service_is_enabled_again(run_setup, apk, capsys):
    device = FakePortalDevice(drop_first_enable=True)

    run_setup(device, path=apk)

    assert device.enable_writes == 2
    assert device.service_enabled
    out = capsys.readouterr().out
    assert "Accessibility service enabled successfully!" in out
    assert "Setup complete!" in out
    assert "ACCESSIBILITY_SETTINGS" not in " ".join(device.shell_calls)


def test_ready_service_is_not_enabled_again(run_setup, apk, capsys):
    device = FakePortalDevice()

    run_setup(device, path=apk)

    assert device.enable_writes == 1
    assert len(device.provider_queries()) == 1
    assert "Setup complete!" in capsys.readouterr().out


def test_starting_service_is_not_enabled_again(run_setup, apk, capsys):
    device = FakePortalDevice(ready_after=3)

    run_setup(device, path=apk)

    assert device.enable_writes == 1
    assert len(device.provider_queries()) == 3
    assert "Setup complete!" in capsys.readouterr().out


def test_service_without_active_window_is_ready(run_setup, apk, capsys):
    device = FakePortalDevice(screen_on=False)

    run_setup(device, path=apk)

    out = capsys.readouterr().out
    assert "Setup complete!" in out
    assert "ACCESSIBILITY_SETTINGS" not in " ".join(device.shell_calls)


def test_unresponsive_service_falls_back_to_manual_setup(run_setup, apk, capsys):
    device = FakePortalDevice(responds=False)

    run_setup(device, path=apk)

    out = capsys.readouterr().out
    assert device.provider_queries()
    assert "Setup complete!" not in out
    assert "Accessibility service enabled successfully!" not in out
    assert "Please complete the following steps on your device" in out
    assert "am start -a android.settings.ACCESSIBILITY_SETTINGS" in device.shell_calls
