import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.devices import (
    DevicePool,
    IOSDevice,
    RecordingDevice,
    RemoteDevice,
    StealthDevice,
)
from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.models.apps import App

URL = "http://alice:secret-password@localhost:5555/path?token=hidden"


def test_cloud_install_verifies_system_apps():
    device = CloudDevice(device_id="device-id", api_key="test")
    device._client = SimpleNamespace(
        devices=SimpleNamespace(
            apps=SimpleNamespace(
                install=AsyncMock(),
                list_installs=AsyncMock(
                    side_effect=[
                        SimpleNamespace(installs=[]),
                        SimpleNamespace(
                            installs=[
                                SimpleNamespace(
                                    app_id="pkg", status="succeeded", started_at=1
                                )
                            ]
                        ),
                    ]
                ),
            )
        )
    )

    async def inventory(include_system=False):
        return [{"package_name": "pkg", "version_code": 4}] if include_system else []

    device.list_apps = inventory
    app = App("pkg", "1", "Example", id="app-id", version_code=4)
    assert asyncio.run(device.install_app(app, timeout=0.05)) is app


@pytest.mark.parametrize("mode", ["direct", "wrapper", "pool", "implicit"])
def test_remote_connection_is_serialized(mode):
    async def run():
        device = RemoteDevice("http://localhost:5555")
        calls = 0

        async def connect():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            device._driver._connected = True

        device._driver.connect = connect
        if mode == "direct":
            await asyncio.gather(device.connect(), device.connect())
        elif mode == "wrapper":
            wrapped = RecordingDevice(device)
            await asyncio.gather(wrapped.connect(), wrapped.connect())
        elif mode == "pool":
            await DevicePool([RecordingDevice(device), StealthDevice(device)]).connect()
        else:
            await asyncio.gather(device.connect(), device._ensure_connected())
        assert calls == 1

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["remote", "legacy", "portal-http"])
def test_backend_log_redacts_url(kind, monkeypatch, caplog):
    from mobilerun.core.devices import ios

    caplog.set_level(logging.INFO)

    async def emit(*args, **kwargs):
        logging.getLogger("mobilerun_core_local").info("Connected to %s", URL)
        logging.getLogger("mobilerun_core_local").info("Connected to " + URL)

    async def run():
        if kind == "remote":
            device = RemoteDevice(URL)
            monkeypatch.setattr(device._driver, "connect", emit)
        else:
            device = IOSDevice(URL)
            monkeypatch.setattr(
                ios, "detect_ios_portal_kind", AsyncMock(return_value=kind)
            )
            monkeypatch.setattr(
                ios.IOSPortalDriver if kind == "legacy" else ios.IOSPortalHttpDriver,
                "ensure_connected",
                emit,
            )
        await device.connect()
        logging.getLogger("mobilerun.core.devices.test").info(
            "Connected to %s", device.identifier
        )

    asyncio.run(run())
    assert caplog.records
    for record in caplog.records:
        assert "alice" not in record.getMessage()
        assert "secret-password" not in record.getMessage()
        assert "hidden" not in record.getMessage()


@pytest.mark.parametrize(
    "mode", ["connect-button", "connect-screenshot", "first-ops", "wrappers", "pool"]
)
def test_public_operations_share_connection_lock(mode, monkeypatch):
    async def run():
        device = RemoteDevice("http://localhost:5555")
        calls = 0

        async def connect():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            device._driver._connected = True

        async def button(_):
            await device._driver.ensure_connected()

        async def screenshot(_=True):
            await device._driver.ensure_connected()
            return b"image"

        async def tap(*_):
            await device._driver.ensure_connected()

        monkeypatch.setattr(device._driver, "connect", connect)
        monkeypatch.setattr(device._driver, "press_button", button)
        monkeypatch.setattr(device._driver, "screenshot", screenshot)
        monkeypatch.setattr(device._driver, "tap", tap)
        if mode == "connect-button":
            work = asyncio.gather(device.connect(), device.press_button("home"))
        elif mode == "connect-screenshot":
            work = asyncio.gather(device.connect(), device.screenshot())
        elif mode == "first-ops":
            work = asyncio.gather(device.tap(1, 2), device.screenshot())
        elif mode == "wrappers":
            nested = RecordingDevice(StealthDevice(device))
            other = RecordingDevice(device)
            work = asyncio.gather(nested.press_button("home"), other.screenshot())
        else:
            pool = DevicePool([RecordingDevice(device), StealthDevice(device)])
            work = asyncio.gather(pool.press_button("home"), pool.screenshot())
        await asyncio.wait_for(work, timeout=1)
        assert calls == 1

    asyncio.run(run())
