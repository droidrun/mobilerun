import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.devices.android import AndroidDevice
from mobilerun.core.devices.base import Device
from mobilerun.errors import DeviceOperationNotSupported


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
    from mobilerun.core.devices.components import files

    image = tmp_path / "a ' photo.png"
    image.write_bytes(b"image")
    adb = SimpleNamespace(
        sync=SimpleNamespace(push=AsyncMock()),
        shell2=AsyncMock(return_value=SimpleNamespace(returncode=0, output="")),
    )
    device = cast(
        AndroidDevice, SimpleNamespace(_ensure_connected=AsyncMock(), _adb_device=adb)
    )
    result = asyncio.run(files.android_push_file(device, image))
    assert result.path == "/sdcard/DCIM/Camera/a ' photo.png"
    adb.sync.push.assert_awaited_once_with(image, result.path)
    command = adb.shell2.await_args.args[0]
    assert "MEDIA_SCANNER_SCAN_FILE" in command
    assert "'\"'\"'" in command


def test_android_file_operations_and_document_destination(tmp_path):
    from mobilerun.core.devices.components import files

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
    pushed = asyncio.run(files.android_push_file(device, document))
    assert pushed.path == "/sdcard/Download/a file.txt"
    assert adb.shell2.await_count == 0
    overridden = asyncio.run(
        files.android_push_file(device, document, "/sdcard/custom")
    )
    assert overridden.path == "/sdcard/custom/a file.txt"
    listed = asyncio.run(files.android_list_files(device, "/sdcard/Download"))
    assert [(f.path, f.size_bytes) for f in listed] == [(pushed.path, 4)]
    output = tmp_path / "out.txt"
    assert asyncio.run(files.android_pull_file(device, pushed.path, output)) == output
    adb.sync.pull.assert_awaited_once_with(pushed.path, output)
    asyncio.run(files.android_delete_file(device, pushed.path))
    assert adb.shell2.await_args.args[0] == "rm -- '/sdcard/Download/a file.txt'"


def test_android_device_delegates_file_operations(monkeypatch, tmp_path):
    from mobilerun.core.devices.android import AndroidDevice
    from mobilerun.core.devices.components import files

    device = AndroidDevice(serial="emulator-5554")
    push = AsyncMock(return_value=SimpleNamespace(path="/sdcard/a"))
    monkeypatch.setattr(files, "android_push_file", push)
    result = asyncio.run(device.push_file(tmp_path / "a", "/sdcard"))
    assert result.path == "/sdcard/a"
    push.assert_awaited_once_with(device, tmp_path / "a", "/sdcard")


def test_push_workspace_downloads_then_pushes(monkeypatch):
    from datetime import datetime, timezone

    from mobilerun.core import workspace
    from mobilerun.core.devices.base import Device
    from mobilerun.models import WorkspaceFile

    item = WorkspaceFile("id", "image.png", "image/png", 3, datetime.now(timezone.utc))
    download = AsyncMock(return_value=b"abc")
    client = AsyncMock()
    client.__aenter__.return_value = client
    monkeypatch.setattr(workspace, "cloud_client", lambda *a, **k: client)
    monkeypatch.setattr(workspace.cloud_files, "download_file", download)
    device = Device()
    device.push_file = AsyncMock(return_value="done")
    assert asyncio.run(workspace.Workspace.push_file(item, device, "/sdcard")) == "done"
    assert download.await_args is not None
    assert device.push_file.await_args is not None
    assert download.await_args.args[1] == "id"
    assert device.push_file.await_args.args[0].name == "image.png"
    assert device.push_file.await_args.args[1] == "/sdcard"


def test_cloud_device_files_use_sdk_transport(monkeypatch, tmp_path):
    from mobilerun.clients.cloud import files as cloud_files
    from mobilerun.core.devices.cloud import CloudDevice

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
    from mobilerun.core.devices.components.wrapper import DeviceWrapper
    from mobilerun.core.devices.pool import DevicePool

    device = Device()
    device.list_files = AsyncMock(return_value=[])
    wrapped = DeviceWrapper(device)
    assert asyncio.run(wrapped.list_files("/sdcard")) == []
    assert asyncio.run(DevicePool([wrapped]).list_files("/sdcard")) == [[]]
    assert device.list_files.await_count == 2
