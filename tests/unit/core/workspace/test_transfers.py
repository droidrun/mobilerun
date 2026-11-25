"""Workspace, device, and SDK file transfers with offline transports."""

import asyncio
import logging
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import httpx
import pytest
from mobilerun_sdk import AsyncMobilerun

from mobilerun import Workspace
from mobilerun.clients.cloud import files as cloud_files
from mobilerun.clients.local import login_store
from mobilerun.core import workspace
from mobilerun.core.devices.android import AndroidDevice
from mobilerun.core.devices.base import Device
from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.core.devices.components import files as device_files
from mobilerun.core.devices.components.wrapper import DeviceWrapper
from mobilerun.core.devices.pool import DevicePool
from mobilerun.errors import CloudPermissionDenied, DeviceOperationNotSupported
from mobilerun.models import App, Billing, WorkspaceFile


def test_catalog_pagination_mapping_and_delete(monkeypatch):
    client = AsyncMock()
    client.__aenter__.return_value = client
    first = SimpleNamespace(
        bundle_id="one",
        display_name="One",
        id="a",
        version=SimpleNamespace(version_name="1", version_code=3),
    )
    second = SimpleNamespace(
        bundle_id="two",
        display_name="Two",
        id="b",
        version=SimpleNamespace(version_name=None, version_code=4),
    )
    client.apps.list.side_effect = [
        SimpleNamespace(items=[first], pagination=SimpleNamespace(has_next=True)),
        SimpleNamespace(items=[second], pagination=SimpleNamespace(has_next=False)),
    ]
    calls = []

    def connect(api_key, base_url, *, billing):
        calls.append((api_key, base_url, billing))
        return client

    monkeypatch.setattr(workspace, "cloud_client", connect)
    options = dict(
        api_key="source", base_url="https://source.test", billing=Billing.NO_COMMERCE
    )
    assert asyncio.run(Workspace.list_apps(**options)) == [
        App("one", "1", "One", id="a", version_code=3),
        App("two", "", "Two", id="b", version_code=4),
    ]
    assert [call.kwargs["page"] for call in client.apps.list.await_args_list] == [1, 2]
    asyncio.run(Workspace.delete_app("a", **options))
    client.apps.delete.assert_awaited_once_with("a")
    assert calls == [("source", "https://source.test", Billing.NO_COMMERCE)] * 2
    assert client.__aexit__.await_count == 2


def test_get_app_default_billing_and_client_cleanup_on_failure(monkeypatch):
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.apps.retrieve.return_value = SimpleNamespace(
        data=SimpleNamespace(bundle_id="one", display_name="One")
    )
    calls = []

    def connect(api_key, base_url, *, billing):
        calls.append((api_key, base_url, billing))
        return client

    monkeypatch.setattr(workspace, "cloud_client", connect)
    assert asyncio.run(Workspace.get_app("a")) == App("one", "", "One", id="a")
    client.apps.retrieve.side_effect = RuntimeError("unavailable")
    with pytest.raises(RuntimeError, match="unavailable"):
        asyncio.run(Workspace.get_app("b"))
    assert calls == [(None, None, Billing.NO_COMMERCE)] * 2
    assert client.__aexit__.await_count == 2


def test_workspace_upload_app_passes_overrides(monkeypatch):
    app = App("one", "1", "One", id="a")
    upload = AsyncMock(return_value=app)
    monkeypatch.setattr(workspace.cloud_apps, "upload_app", upload)
    assert (
        asyncio.run(
            Workspace.upload_app(
                "one.apk", api_key="key", base_url="https://source.test"
            )
        )
        is app
    )
    upload.assert_awaited_once_with(
        "one.apk",
        api_key="key",
        base_url="https://source.test",
        billing=Billing.NO_COMMERCE,
    )


def test_canonical_api_key_only_and_saved_login_fallback(monkeypatch):

    monkeypatch.setattr(login_store, "load_credentials", lambda: {"api_key": "saved"})
    monkeypatch.setenv("MOBILERUN_CLOUD_API_KEY", "obsolete")
    monkeypatch.delenv("MOBILERUN_API_KEY", raising=False)
    assert login_store.resolve_cloud_api_key() == "saved"
    monkeypatch.setenv("MOBILERUN_API_KEY", "canonical")
    assert login_store.resolve_cloud_api_key() == "canonical"


def test_workspace_api_owns_client_and_downloads_into_directory(monkeypatch, tmp_path):

    item = WorkspaceFile("id", "photo.jpg", "image/jpeg", 5, datetime.now(timezone.utc))
    client = AsyncMock()
    client.__aenter__.return_value = client
    monkeypatch.setattr(workspace, "cloud_client", lambda *args, **kwargs: client)
    monkeypatch.setattr(
        workspace.cloud_files, "upload_file", AsyncMock(return_value=item)
    )
    monkeypatch.setattr(
        workspace.cloud_files, "list_files", AsyncMock(return_value=[item])
    )
    monkeypatch.setattr(
        workspace.cloud_files, "download_file", AsyncMock(return_value=b"image")
    )
    monkeypatch.setattr(workspace.cloud_files, "delete_file", AsyncMock())

    assert (
        asyncio.run(Workspace.upload_file("photo.jpg", billing=Billing.NO_COMMERCE))
        == item
    )
    assert asyncio.run(Workspace.list_files(billing=Billing.NO_COMMERCE)) == [item]
    result = asyncio.run(
        Workspace.download_file("id", tmp_path, billing=Billing.NO_COMMERCE)
    )
    assert result == tmp_path / "photo.jpg"
    assert result.read_bytes() == b"image"
    asyncio.run(Workspace.delete_file("id", billing=Billing.NO_COMMERCE))
    workspace.cloud_files.delete_file.assert_awaited_once_with(
        client, "id", billing=Billing.NO_COMMERCE
    )
    assert client.__aenter__.await_count == client.__aexit__.await_count == 4


def test_download_rejects_unknown_file(monkeypatch, tmp_path):

    client = AsyncMock()
    client.__aenter__.return_value = client
    monkeypatch.setattr(workspace, "cloud_client", lambda *args, **kwargs: client)
    monkeypatch.setattr(workspace.cloud_files, "list_files", AsyncMock(return_value=[]))
    with pytest.raises(FileNotFoundError):
        asyncio.run(
            Workspace.download_file("missing", tmp_path, billing=Billing.NO_COMMERCE)
        )


def test_base_device_rejects_file_operations():
    async def run():
        device = Device()
        with pytest.raises(DeviceOperationNotSupported):
            await device.push_file("x")
        with pytest.raises(DeviceOperationNotSupported):
            await device.list_files("/sdcard")
        with pytest.raises(DeviceOperationNotSupported):
            await device.pull_file("x", "y")
        with pytest.raises(DeviceOperationNotSupported):
            await device.delete_file("x")

    asyncio.run(run())


def test_android_push_uses_media_directory_and_scans_quoted_path(tmp_path):

    image = tmp_path / "a ' photo.png"
    image.write_bytes(b"image")
    adb = SimpleNamespace(
        sync=SimpleNamespace(push=AsyncMock()),
        shell2=AsyncMock(return_value=SimpleNamespace(returncode=0, output="")),
    )
    device = cast(
        AndroidDevice, SimpleNamespace(_ensure_connected=AsyncMock(), _adb_device=adb)
    )
    result = asyncio.run(device_files.android_push_file(device, image))
    assert result.path == "/sdcard/DCIM/Camera/a ' photo.png"
    adb.sync.push.assert_awaited_once_with(image, result.path)
    command = adb.shell2.await_args.args[0]
    assert "MEDIA_SCANNER_SCAN_FILE" in command
    assert "'\"'\"'" in command


def test_android_file_operations_and_document_destination(tmp_path):

    document = tmp_path / "a file.txt"
    document.write_bytes(b"text")

    async def entries(path):
        yield SimpleNamespace(path="a file.txt", size=4, mtime=None)

    adb = SimpleNamespace(
        sync=SimpleNamespace(
            push=AsyncMock(), pull=AsyncMock(), iter_directory=entries
        ),
        shell2=AsyncMock(return_value=SimpleNamespace(returncode=0, output="")),
    )
    device = cast(
        AndroidDevice, SimpleNamespace(_ensure_connected=AsyncMock(), _adb_device=adb)
    )
    pushed = asyncio.run(device_files.android_push_file(device, document))
    assert pushed.path == "/sdcard/Download/a file.txt"
    assert adb.shell2.await_count == 0
    overridden = asyncio.run(
        device_files.android_push_file(device, document, "/sdcard/custom")
    )
    assert overridden.path == "/sdcard/custom/a file.txt"
    listed = asyncio.run(device_files.android_list_files(device, "/sdcard/Download"))
    assert [(f.path, f.size_bytes) for f in listed] == [(pushed.path, 4)]
    output = tmp_path / "out.txt"
    assert (
        asyncio.run(device_files.android_pull_file(device, pushed.path, output))
        == output
    )
    adb.sync.pull.assert_awaited_once_with(pushed.path, output)
    asyncio.run(device_files.android_delete_file(device, pushed.path))
    assert adb.shell2.await_args.args[0] == "rm -- '/sdcard/Download/a file.txt'"


def test_android_device_delegates_file_operations(monkeypatch, tmp_path):

    device = AndroidDevice(serial="emulator-5554")
    push = AsyncMock(return_value=SimpleNamespace(path="/sdcard/a"))
    monkeypatch.setattr(device_files, "android_push_file", push)
    result = asyncio.run(device.push_file(tmp_path / "a", "/sdcard"))
    assert result.path == "/sdcard/a"
    push.assert_awaited_once_with(device, tmp_path / "a", "/sdcard")


def test_push_workspace_downloads_then_pushes(monkeypatch):

    item = WorkspaceFile("id", "image.png", "image/png", 3, datetime.now(timezone.utc))
    download = AsyncMock(return_value=b"abc")
    client = AsyncMock()
    client.__aenter__.return_value = client
    monkeypatch.setattr(workspace, "cloud_client", lambda *a, **k: client)
    monkeypatch.setattr(workspace.cloud_files, "download_file", download)
    device = Device()
    device.push_file = AsyncMock(return_value="done")
    assert asyncio.run(Workspace.push_file(item, device, "/sdcard")) == "done"
    assert download.await_args is not None
    assert device.push_file.await_args is not None
    assert download.await_args.args[1] == "id"
    assert device.push_file.await_args.args[0].name == "image.png"
    assert device.push_file.await_args.args[1] == "/sdcard"


def test_cloud_device_files_use_sdk_transport(monkeypatch, tmp_path):

    device = CloudDevice("id", api_key="test")
    device._ensure_connected = AsyncMock()
    push = AsyncMock()
    listing = AsyncMock(return_value=[])
    pull = AsyncMock(return_value=b"abc")
    remove = AsyncMock()
    monkeypatch.setattr(cloud_files, "push_device_file", push)
    monkeypatch.setattr(cloud_files, "list_device_files", listing)
    monkeypatch.setattr(cloud_files, "download_device_file", pull)
    monkeypatch.setattr(cloud_files, "delete_device_file", remove)
    path = tmp_path / "a.png"
    path.write_bytes(b"abc")
    result = asyncio.run(device.push_file(path))
    assert result.path == "/sdcard/DCIM/Camera/a.png"
    assert asyncio.run(device.list_files("/sdcard")) == []
    out = tmp_path / "copy.png"
    assert asyncio.run(device.pull_file(result.path, out)).read_bytes() == b"abc"
    asyncio.run(device.delete_file(result.path))
    assert (
        push.await_count
        == listing.await_count
        == pull.await_count
        == remove.await_count
        == 1
    )


def test_wrapper_and_pool_forward_files():

    device = Device()
    device.list_files = AsyncMock(return_value=[])
    wrapped = DeviceWrapper(device)
    assert asyncio.run(wrapped.list_files("/sdcard")) == []
    assert asyncio.run(DevicePool([wrapped]).list_files("/sdcard")) == [[]]
    assert device.list_files.await_count == 2


def test_upload_confirms_only_after_put_and_cancels_on_failure(monkeypatch, tmp_path):

    path = tmp_path / "photo.jpg"
    path.write_bytes(b"image")
    now = datetime.now(timezone.utc)
    resource = SimpleNamespace(
        upload_url=AsyncMock(
            return_value=SimpleNamespace(file_id="f1", put_url="https://put")
        ),
        confirm=AsyncMock(
            return_value=SimpleNamespace(
                file_id="f1",
                filename=path.name,
                mime_type="image/jpeg",
                size_bytes=5,
                created_at=now,
            )
        ),
        cancel_pending=AsyncMock(),
    )
    client = SimpleNamespace(files=resource)
    http = AsyncMock()
    http.__aenter__.return_value = http
    http.put.return_value = SimpleNamespace(raise_for_status=lambda: None)
    monkeypatch.setattr(cloud_files.httpx, "AsyncClient", lambda: http)
    result = asyncio.run(
        cloud_files.upload_file(client, path, billing=Billing.NO_COMMERCE)
    )
    assert (result.id, result.name, result.size_bytes) == ("f1", "photo.jpg", 5)
    resource.upload_url.assert_awaited_once_with(
        filename="photo.jpg", mime_type="image/jpeg", size_bytes=5, zone="user"
    )
    http.put.assert_awaited_once_with(
        "https://put",
        content=b"image",
        headers={"Content-Type": "image/jpeg"},
        timeout=None,
    )
    resource.confirm.assert_awaited_once_with("f1")
    resource.cancel_pending.assert_not_awaited()
    http.put.side_effect = RuntimeError("failure")
    with pytest.raises(RuntimeError, match="failure"):
        asyncio.run(cloud_files.upload_file(client, path, billing=Billing.NO_COMMERCE))
    resource.cancel_pending.assert_awaited_once_with("f1")
    resource.confirm.assert_awaited_once_with("f1")


def test_list_download_and_delete(monkeypatch, tmp_path):

    now = datetime.now(timezone.utc)
    item = SimpleNamespace(
        file_id="f1",
        filename="photo.jpg",
        mime_type="image/jpeg",
        size_bytes=5,
        created_at=now,
    )
    raw = SimpleNamespace(read=AsyncMock(return_value=b"image"))
    resource = SimpleNamespace(
        list=AsyncMock(return_value=SimpleNamespace(files=[item])),
        delete=AsyncMock(),
        with_raw_response=SimpleNamespace(download=AsyncMock(return_value=raw)),
    )
    client = SimpleNamespace(files=resource)
    items = asyncio.run(cloud_files.list_files(client, billing=Billing.NO_COMMERCE))
    assert items[0].name == "photo.jpg"
    assert (
        asyncio.run(
            cloud_files.download_file(client, "f1", billing=Billing.NO_COMMERCE)
        )
        == b"image"
    )
    asyncio.run(cloud_files.delete_file(client, "f1", billing=Billing.NO_COMMERCE))
    resource.list.assert_awaited_once_with(zone="user")
    resource.with_raw_response.download.assert_awaited_once_with("f1")
    resource.delete.assert_awaited_once_with("f1")


def test_device_file_sdk_transport(tmp_path):

    path = tmp_path / "picture.png"
    path.write_bytes(b"png")
    item = SimpleNamespace(name="picture.png", size=3, modified_at=None)
    raw = SimpleNamespace(read=AsyncMock(return_value=b"png"))
    resource = SimpleNamespace(
        upload=AsyncMock(),
        list=AsyncMock(return_value=SimpleNamespace(files=[item])),
        delete=AsyncMock(),
        with_raw_response=SimpleNamespace(download=AsyncMock(return_value=raw)),
    )
    client = SimpleNamespace(devices=SimpleNamespace(files=resource))
    asyncio.run(
        cloud_files.push_device_file(
            client, "device", path, "/sdcard", billing=Billing.NO_COMMERCE
        )
    )
    resource.upload.assert_awaited_once_with("device", path="/sdcard", file=path)
    listed = asyncio.run(
        cloud_files.list_device_files(
            client, "device", "/sdcard", billing=Billing.NO_COMMERCE
        )
    )
    assert listed[0].path == "/sdcard/picture.png"
    assert (
        asyncio.run(
            cloud_files.download_device_file(
                client, "device", "/sdcard/picture.png", billing=Billing.NO_COMMERCE
            )
        )
        == b"png"
    )
    asyncio.run(
        cloud_files.delete_device_file(
            client, "device", "/sdcard/picture.png", billing=Billing.NO_COMMERCE
        )
    )
    resource.delete.assert_awaited_once_with("device", path="/sdcard/picture.png")


@pytest.mark.parametrize("device", [False, True])
@pytest.mark.parametrize("status", [200, 403])
def test_sdk_download_redirect_hides_signature_and_strips_auth(caplog, device, status):

    requests = []
    signature = "storage-redirect-canary"
    caplog.set_level(logging.INFO, logger="httpx")
    caplog.set_level(logging.DEBUG, logger="mobilerun_sdk._base_client")

    def respond(request):
        requests.append(request)
        if request.url.host == "api.example":
            return httpx.Response(
                302,
                headers={"Location": f"https://storage.example/signed?sig={signature}"},
            )
        return httpx.Response(
            status, content=b"\x00\xff" if status == 200 else b"denied"
        )

    async def run():
        http = httpx.AsyncClient(
            transport=httpx.MockTransport(respond), follow_redirects=True
        )
        async with AsyncMobilerun(
            api_key="test", base_url="https://api.example", http_client=http
        ) as client:
            if device:
                return await cloud_files.download_device_file(
                    client, "id", "/sdcard/test", billing=Billing.NO_COMMERCE
                )
            return await cloud_files.download_file(
                client, "id", billing=Billing.NO_COMMERCE
            )

    if status == 200:
        assert asyncio.run(run()) == b"\x00\xff"
    else:
        with pytest.raises(CloudPermissionDenied) as caught:
            asyncio.run(run())
        assert signature not in "".join(traceback.format_exception(caught.value))
    assert len(requests) == 2
    assert "authorization" not in requests[1].headers
    assert requests[1].url.params["sig"] == signature
    assert signature not in caplog.text
