"""Cloud app installation, SDK requests, and upload contracts."""

import asyncio
import traceback
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from mobilerun_sdk import AsyncMobilerun

import mobilerun.core.devices.components.cloud_apps as cloud
from mobilerun.clients.cloud import apps
from mobilerun.clients.local import apk as local_apk
from mobilerun.core.devices.base import Device
from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.core.devices.pool import DevicePool
from mobilerun.core.devices.recording import RecordingDevice
from mobilerun.core.devices.stealth import StealthDevice
from mobilerun.errors import CloudConnectionError, DeviceOperationNotSupported
from mobilerun.models.apps import App
from mobilerun.models.billing import Billing


def test_cloud_install_by_path_and_id(monkeypatch, tmp_path):

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
        "device-id", package_name="org.example", version_code=4, background=True
    )
    asyncio.run(device.install_app(app))
    device._client.apps.retrieve.assert_not_awaited()
    device._client.devices.apps.install.assert_awaited_with(
        "device-id", package_name="org.example", version_code=4, background=True
    )
    asyncio.run(device.uninstall_app("org.example"))
    device._client.devices.apps.delete.assert_awaited_once_with(
        "org.example", device_id="device-id", extra_body={}
    )


def test_wrappers_and_pool_forward_cloud_app():

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


@pytest.mark.parametrize(
    "options, message",
    [({"grant_permissions": False}, "grant"), ({"reinstall": True}, "reinstall")],
    ids=["grant-permissions", "reinstall"],
)
def test_cloud_install_rejects_unsupported_install_options(options, message):

    device = CloudDevice(device_id="device-id", api_key="test")
    with pytest.raises(DeviceOperationNotSupported, match=message):
        asyncio.run(device.install_app(App("p", "", "P", id="app-id"), **options))


def test_uninstall_sends_required_empty_json_body():
    requests = []

    async def respond(request):
        requests.append(request)
        return httpx.Response(204, request=request)

    async def run():
        http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        async with AsyncMobilerun(
            api_key="test", base_url="https://example.test/v1", http_client=http
        ) as client:
            await apps.uninstall(
                client, "ai.example.app", billing=Billing.NO_COMMERCE, device_id="phone"
            )

    asyncio.run(run())
    assert len(requests) == 1
    assert requests[0].method == "DELETE"
    assert requests[0].url.path == "/v1/devices/phone/apps/ai.example.app"
    assert requests[0].content == b"{}"


def test_cloud_stop_uses_sdk_and_display_without_clearing_data():
    device = CloudDevice(device_id="phone", api_key="test", display_id=7)
    device._ensure_connected = AsyncMock()
    stop = AsyncMock()
    device._write_client = SimpleNamespace(
        devices=SimpleNamespace(apps=SimpleNamespace(stop=stop))
    )

    assert asyncio.run(device.stop_app("ai.example.app")) == "Stopped ai.example.app"
    stop.assert_awaited_once_with(
        "ai.example.app", device_id="phone", x_device_display_id=7
    )
    assert "stop_app" in device.supported_operations


def test_upload_confirms_and_marks_failure(monkeypatch, tmp_path):

    apk = tmp_path / "sample.apk"
    apk.write_bytes(b"apk")
    monkeypatch.setattr(
        apps, "inspect_apk", lambda path: ("org.example", 3, "1.2", "Example")
    )
    signed = SimpleNamespace(
        app_id="app-id",
        version_id="version-id",
        r2_upload_urls=[
            SimpleNamespace(file_name="sample.apk", r2_upload_url="https://upload")
        ],
    )
    sdk = SimpleNamespace(
        apps=SimpleNamespace(
            list=AsyncMock(
                return_value=SimpleNamespace(
                    items=[], pagination=SimpleNamespace(has_next=False)
                )
            ),
            create_signed_upload_url=AsyncMock(return_value=signed),
            confirm_upload=AsyncMock(),
            mark_failed=AsyncMock(),
        )
    )
    client = AsyncMock()
    client.__aenter__.return_value = sdk
    monkeypatch.setattr(apps, "cloud_client", lambda *args, **kwargs: client)
    http = AsyncMock()
    http.__aenter__.return_value = http
    http.put.return_value = SimpleNamespace(raise_for_status=lambda: None)
    monkeypatch.setattr(apps.httpx, "AsyncClient", lambda **kwargs: http)
    result = asyncio.run(apps.upload_app(apk, billing="included", api_key="test"))
    assert result.id == "app-id"
    sdk.apps.confirm_upload.assert_awaited_once_with("version-id")
    http.put.side_effect = RuntimeError("upload failed")
    with pytest.raises(RuntimeError, match="upload failed"):
        asyncio.run(apps.upload_app(apk, billing="included", api_key="test"))
    sdk.apps.mark_failed.assert_awaited_once_with("version-id")
    sdk.apps.mark_failed.reset_mock()
    token = "upload-signature-canary"
    http.put.side_effect = httpx.ConnectError(f"https://signed.test?token={token}")
    with pytest.raises(CloudConnectionError) as raised:
        asyncio.run(apps.upload_app(apk, billing="included", api_key="test"))
    error = raised.value
    for rendered in (
        str(error),
        repr(error),
        "".join(traceback.format_exception(error)),
    ):
        assert "signed.test" not in rendered
        assert token not in rendered
    sdk.apps.mark_failed.assert_awaited_once_with("version-id")


def test_app_items_paginates():

    item = SimpleNamespace(
        id="id",
        bundle_id="org.example",
        display_name="Example",
        version=SimpleNamespace(version_name="1", version_code=3),
    )
    client = SimpleNamespace(
        apps=SimpleNamespace(
            list=AsyncMock(
                side_effect=[
                    SimpleNamespace(
                        items=[], pagination=SimpleNamespace(has_next=True)
                    ),
                    SimpleNamespace(
                        items=[item], pagination=SimpleNamespace(has_next=False)
                    ),
                ]
            )
        )
    )

    assert asyncio.run(apps.list_apps(client, billing=Billing.NO_COMMERCE)) == [
        App("org.example", "1", "Example", id="id", version_code=3)
    ]
    assert [call.kwargs["page"] for call in client.apps.list.await_args_list] == [1, 2]


def test_apk_metadata_from_build_tools(monkeypatch, tmp_path):

    monkeypatch.setattr(local_apk.shutil, "which", lambda name: "/sdk/aapt")
    monkeypatch.setattr(
        local_apk.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="package: name='org.example' versionCode='12' versionName='1.2'\napplication-label:'Example'\n"
        ),
    )
    assert local_apk.inspect_apk(tmp_path / "app.apk") == (
        "org.example",
        12,
        "1.2",
        "Example",
    )
