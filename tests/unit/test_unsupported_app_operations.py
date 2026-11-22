import asyncio

import pytest
from mobilerun_core_local.driver.android import AndroidPortalHttpDriver

from mobilerun.core.devices.components.wrapper import DriverDevice
from mobilerun.core.devices.ios import IOSDevice


def test_android_portal_app_operations_raise_domain_error():
    from mobilerun.core.state.options import android_state_provider
    from mobilerun.errors import DeviceOperationNotSupported

    device = DriverDevice(
        AndroidPortalHttpDriver("http://localhost:8080", "test"),
        state_provider_factory=android_state_provider,
    )
    with pytest.raises(DeviceOperationNotSupported, match="cannot install"):
        asyncio.run(device.install_apk("app.apk"))
    with pytest.raises(DeviceOperationNotSupported, match="cannot uninstall"):
        asyncio.run(device.uninstall_app("org.example"))


def test_ios_uninstall_raises_domain_error():
    from mobilerun.errors import DeviceOperationNotSupported

    device = IOSDevice("http://localhost")
    from mobilerun_core_local.driver.ios import IOSPortalDriver

    device._driver = IOSPortalDriver("http://localhost")
    with pytest.raises(DeviceOperationNotSupported, match="uninstall"):
        asyncio.run(device.uninstall_app("org.example"))


def test_android_local_install_delegates(monkeypatch):
    from unittest.mock import AsyncMock

    from mobilerun_core_local.driver.android import AndroidDriver

    from mobilerun.core.devices.android import AndroidDevice

    install = AsyncMock(return_value="installed")
    monkeypatch.setattr(AndroidDriver, "install_app", install)
    monkeypatch.setattr(
        "mobilerun.core.devices.android.inspect_apk",
        lambda path: ("org.example", 1, "1", "Example"),
    )
    from types import SimpleNamespace

    device = AndroidDevice("fake")
    device._driver.ensure_connected = AsyncMock()
    device._adb_device = SimpleNamespace(
        shell=AsyncMock(return_value="package:/data/app/base.apk")
    )
    assert (
        asyncio.run(device.install_apk("app.apk", reinstall=True)).package
        == "org.example"
    )
    install.assert_awaited_once_with("app.apk", reinstall=True)
