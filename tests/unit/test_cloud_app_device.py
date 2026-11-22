import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.models.apps import App


def test_cloud_install_by_path_and_id(monkeypatch, tmp_path):
    import mobilerun.core.devices.components.cloud_apps as cloud

    apk = tmp_path / "test.apk"
    apk.write_bytes(b"apk")
    app = App("org.example", "1.0", "Example", id="app-id", version_code=4)
    upload = AsyncMock(return_value=app)
    monkeypatch.setattr(cloud, "upload_app", upload)
    device = CloudDevice(device_id="device-id", api_key="test")
    device._client = SimpleNamespace(
        api_key="test",
        base_url="https://example.test",
        apps=SimpleNamespace(
            retrieve=AsyncMock(
                return_value=SimpleNamespace(
                    data=SimpleNamespace(
                        bundle_id="org.example", display_name="Example"
                    )
                )
            )
        ),
        devices=SimpleNamespace(
            apps=SimpleNamespace(
                install=AsyncMock(),
                delete=AsyncMock(),
                list_installs=AsyncMock(
                    side_effect=[
                        SimpleNamespace(installs=[]),
                        SimpleNamespace(
                            installs=[
                                SimpleNamespace(
                                    app_id="org.example",
                                    status="succeeded",
                                    started_at=1,
                                )
                            ]
                        ),
                        SimpleNamespace(installs=[]),
                        SimpleNamespace(
                            installs=[
                                SimpleNamespace(
                                    app_id="org.example",
                                    status="succeeded",
                                    started_at=2,
                                )
                            ]
                        ),
                    ]
                ),
            )
        ),
    )
    device.list_apps = AsyncMock(
        return_value=[{"package_name": "org.example", "version_code": 4}]
    )
    asyncio.run(device.install_apk(apk))
    upload.assert_awaited_once()
    device._client.devices.apps.install.assert_awaited_with(
        "device-id", package_name="org.example", version_code=4
    )
    asyncio.run(device.install_app(app))
    device._client.apps.retrieve.assert_not_awaited()
    device._client.devices.apps.install.assert_awaited_with(
        "device-id", package_name="org.example", version_code=4
    )
    asyncio.run(device.uninstall_app("org.example"))
    device._client.devices.apps.delete.assert_awaited_once_with(
        "org.example", device_id="device-id", extra_body={}
    )


def test_wrappers_and_pool_forward_cloud_app():
    from mobilerun.core.devices.base import Device
    from mobilerun.core.devices.pool import DevicePool
    from mobilerun.core.devices.recording import RecordingDevice
    from mobilerun.core.devices.stealth import StealthDevice

    class Fake(Device):
        platform = "android"
        capabilities = set()
        supported_buttons = set()
        _ui_state_view = SimpleNamespace(_provider=None)
        operator = None
        _is_cloud = True
        install_app = AsyncMock(return_value="ok")
        uninstall_app = AsyncMock(return_value="removed")

    async def run():
        fake = Fake()
        app = App("org.example", "1", "Example", id="app-id", version_code=4)
        pool = DevicePool([RecordingDevice(fake), StealthDevice(fake)])
        assert await pool.install_app(app) == ["ok", "ok"]
        assert await pool.uninstall_app("org.example") == ["removed", "removed"]
        assert fake.install_app.await_count == 2
        assert fake.uninstall_app.await_count == 2

    asyncio.run(run())


def test_cloud_install_rejects_unsupported_install_options():
    import pytest

    from mobilerun.errors import DeviceOperationNotSupported

    device = CloudDevice(device_id="device-id", api_key="test")
    with pytest.raises(DeviceOperationNotSupported, match="grant"):
        asyncio.run(
            device.install_app(App("p", "", "P", id="app-id"), grant_permissions=False)
        )
    with pytest.raises(DeviceOperationNotSupported, match="reinstall"):
        asyncio.run(device.install_app(App("p", "", "P", id="app-id"), reinstall=True))
