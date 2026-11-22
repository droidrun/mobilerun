import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.devices.android import AndroidDevice
from mobilerun.core.devices.components.cloud_lifecycle import _CloudLifecycle
from mobilerun.core.devices.ios import IOSDevice
from mobilerun.core.devices.remote import RemoteDevice
from mobilerun.core.mailbox.cloud import CloudMailbox


def test_android_cleanup_after_extension_failure():
    device = AndroidDevice(serial="test")
    device._close_extensions = AsyncMock(side_effect=RuntimeError("detach"))
    device._started_emulator = True
    device._keep = False
    device.terminate = AsyncMock()
    with pytest.raises(RuntimeError, match="detach"):
        asyncio.run(device.close())
    device.terminate.assert_awaited_once()


@pytest.mark.parametrize("kind", ["ios", "remote"])
def test_client_cleanup_after_extension_failure(kind):
    if kind == "ios":
        device = IOSDevice("http://localhost")
    else:
        device = RemoteDevice("http://localhost")
    client = SimpleNamespace(aclose=AsyncMock())
    device._driver = SimpleNamespace(_client=client)
    device._close_extensions = AsyncMock(side_effect=RuntimeError("detach"))
    with pytest.raises(RuntimeError, match="detach"):
        asyncio.run(device.close())
    client.aclose.assert_awaited_once()


def test_cloud_cleanup_after_extension_failure(monkeypatch):
    from mobilerun.core.devices.components import cloud_lifecycle

    device = object.__new__(_CloudLifecycle)
    device._close_extensions = AsyncMock(side_effect=RuntimeError("detach"))
    device._close_resources = AsyncMock()
    device._connect_lock = asyncio.Lock()
    device._owned = True
    device._keep = False
    device.device_id = "device"
    device._billing = "included"
    device._client = object()
    device._owned_client = SimpleNamespace(close=AsyncMock())
    client = device._owned_client
    terminate = AsyncMock()
    monkeypatch.setattr(cloud_lifecycle.devices, "terminate", terminate)
    with pytest.raises(RuntimeError, match="detach"):
        asyncio.run(device.close())
    device._close_resources.assert_awaited_once()
    terminate.assert_awaited_once()
    client.close.assert_awaited_once()


def test_list_mailboxes_preserves_address(monkeypatch):
    from mobilerun.core.mailbox import cloud

    monkeypatch.setattr(
        cloud.mailboxes,
        "list_mailboxes",
        AsyncMock(return_value=[SimpleNamespace(id="id", address="a@example.com")]),
    )
    result = asyncio.run(CloudMailbox.list(billing="included"))
    assert result[0].address == "a@example.com"


def test_apk_missing_optional_version_name(monkeypatch, tmp_path):
    from mobilerun.clients.local import apk

    monkeypatch.setattr(apk.shutil, "which", lambda _: "/aapt")
    monkeypatch.setattr(
        apk.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            stdout="package: name='example' versionCode='1'\n"
        ),
    )
    assert apk.inspect_apk(tmp_path / "app.apk") == ("example", 1, "", "example")


def test_apk_inspection_timeout(monkeypatch, tmp_path):
    from mobilerun.clients.local import apk

    monkeypatch.setattr(apk.shutil, "which", lambda _: "/aapt")

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] > 0
        raise apk.subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(apk.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out"):
        apk.inspect_apk(tmp_path / "app.apk")


def test_android_apk_inspection_runs_off_event_loop(monkeypatch, tmp_path):
    import threading

    from mobilerun.core.devices import android

    loop_thread = threading.get_ident()
    seen = []

    def inspect(path):
        seen.append(threading.get_ident())
        return ("example", 1, "", "Example")

    monkeypatch.setattr(android, "inspect_apk", inspect)
    device = AndroidDevice(serial="test")
    device.install_app = AsyncMock()
    asyncio.run(device.install_apk(tmp_path / "a.apk"))
    assert seen and seen[0] != loop_thread


def test_cloud_upload_does_not_reuse_unverified_binary(monkeypatch, tmp_path):
    from mobilerun.clients.cloud import apps

    apk = tmp_path / "app.apk"
    apk.write_bytes(b"new binary")
    monkeypatch.setattr(apps, "inspect_apk", lambda _: ("example", 1, "1", "Example"))
    item = SimpleNamespace(id="old", bundle_id="example", display_name="Example")
    sdk = SimpleNamespace(
        apps=SimpleNamespace(
            list=AsyncMock(
                return_value=SimpleNamespace(
                    items=[item], pagination=SimpleNamespace(has_next=False)
                )
            ),
            list_versions=AsyncMock(
                return_value=SimpleNamespace(
                    data=[
                        SimpleNamespace(
                            status="available", version_code=1, version_name="1"
                        )
                    ]
                )
            ),
            create_signed_upload_url=AsyncMock(
                return_value=SimpleNamespace(
                    app_id="new",
                    version_id="v",
                    r2_upload_urls=[
                        SimpleNamespace(
                            file_name="app.apk", r2_upload_url="https://example.test"
                        )
                    ],
                )
            ),
            confirm_upload=AsyncMock(),
            mark_failed=AsyncMock(),
        )
    )
    client = AsyncMock()
    client.__aenter__.return_value = sdk
    monkeypatch.setattr(apps, "cloud_client", lambda *a, **kw: client)
    http = AsyncMock()
    http.__aenter__.return_value = http
    http.put.return_value = SimpleNamespace(raise_for_status=lambda: None)
    monkeypatch.setattr(apps.httpx, "AsyncClient", lambda **kw: http)
    result = asyncio.run(apps.upload_app(apk, billing="included"))
    assert result.id == "new"
    http.put.assert_awaited_once()


def test_extension_error_remains_first_when_cleanup_fails():
    device = AndroidDevice(serial="test")
    device._close_extensions = AsyncMock(side_effect=RuntimeError("detach"))
    device._started_emulator = True
    device._keep = False
    device.terminate = AsyncMock(side_effect=ValueError("terminate"))
    with pytest.raises(RuntimeError, match="detach"):
        asyncio.run(device.close())


def test_cloud_upload_preparation_runs_off_event_loop(monkeypatch, tmp_path):
    import threading

    from mobilerun.clients.cloud import apps

    path = tmp_path / "app.apk"
    path.write_bytes(b"apk")
    main_thread = threading.get_ident()
    seen = []

    def inspect(_):
        seen.append(threading.get_ident())
        return ("example", 1, "", "Example")

    monkeypatch.setattr(apps, "inspect_apk", inspect)
    client = AsyncMock()
    client.__aenter__.side_effect = RuntimeError("stop after preparation")
    monkeypatch.setattr(apps, "cloud_client", lambda *a, **kw: client)
    with pytest.raises(RuntimeError, match="stop after preparation"):
        asyncio.run(apps.upload_app(path, billing="included"))
    assert seen and seen[0] != main_thread
