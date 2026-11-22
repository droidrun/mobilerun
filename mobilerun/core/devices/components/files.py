"""File operations for Android devices."""

import mimetypes
import posixpath
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from mobilerun.clients.cloud import files as cloud_files
from mobilerun.models.files import DeviceFile

if TYPE_CHECKING:
    from mobilerun.core.devices.android import AndroidDevice
    from mobilerun.core.devices.cloud import CloudDevice


def destination(path: Path, dest: str | None) -> tuple[str, bool]:
    mime = mimetypes.guess_type(path.name)[0] or ""
    media = mime.startswith(("image/", "video/"))
    return dest or ("/sdcard/DCIM/Camera" if media else "/sdcard/Download"), media


async def android_push_file(
    device: "AndroidDevice", path: str | Path, dest: str | None = None
) -> DeviceFile:
    path = Path(path)
    directory, media = destination(path, dest)
    remote = posixpath.join(directory, path.name)
    await device._ensure_connected()
    adb = device._adb_device
    if adb is None:
        raise RuntimeError("Android device is not connected")
    await adb.sync.push(path, remote)
    if media:
        result = await adb.shell2(
            "am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d "
            + shlex.quote("file://" + remote)
        )
        if result.returncode:
            raise RuntimeError(f"Media scan failed: {result.output}")
    return DeviceFile(remote, path.name, path.stat().st_size, None)


async def android_list_files(
    device: "AndroidDevice", directory: str
) -> list[DeviceFile]:
    await device._ensure_connected()
    adb = device._adb_device
    if adb is None:
        raise RuntimeError("Android device is not connected")
    return [
        DeviceFile(
            posixpath.join(directory, item.path), item.path, item.size, item.mtime
        )
        async for item in adb.sync.iter_directory(directory)
        if item.path not in {".", ".."}
    ]


async def android_pull_file(
    device: "AndroidDevice", remote_path: str, local_path: str | Path
) -> Path:
    await device._ensure_connected()
    adb = device._adb_device
    if adb is None:
        raise RuntimeError("Android device is not connected")
    destination = Path(local_path)
    if destination.is_dir():
        destination = destination / posixpath.basename(remote_path)
    await adb.sync.pull(remote_path, destination)
    return destination


async def android_delete_file(device: "AndroidDevice", remote_path: str) -> None:
    await device._ensure_connected()
    adb = device._adb_device
    if adb is None:
        raise RuntimeError("Android device is not connected")
    result = await adb.shell2("rm -- " + shlex.quote(remote_path))
    if result.returncode:
        raise RuntimeError(f"File deletion failed: {result.output}")


async def cloud_push_file(
    device: "CloudDevice", path: str | Path, dest: str | None
) -> DeviceFile:
    path = Path(path)
    directory, _ = destination(path, dest)
    await device._ensure_connected()
    await cloud_files.push_device_file(
        device._write_client,
        device._require_device_id(),
        path,
        directory,
        billing=device._billing,
    )
    return DeviceFile(
        posixpath.join(directory, path.name), path.name, path.stat().st_size, None
    )


async def cloud_list_files(device: "CloudDevice", directory: str) -> list[DeviceFile]:
    await device._ensure_connected()
    return await cloud_files.list_device_files(
        device._client, device._require_device_id(), directory, billing=device._billing
    )


async def cloud_pull_file(
    device: "CloudDevice", remote_path: str, local_path: str | Path
) -> Path:
    await device._ensure_connected()
    data = await cloud_files.download_device_file(
        device._client,
        device._require_device_id(),
        remote_path,
        billing=device._billing,
    )
    path = Path(local_path)
    if path.is_dir():
        path = path / posixpath.basename(remote_path)
    path.write_bytes(data)
    return path


async def cloud_delete_file(device: "CloudDevice", remote_path: str) -> None:
    await device._ensure_connected()
    await cloud_files.delete_device_file(
        device._write_client,
        device._require_device_id(),
        remote_path,
        billing=device._billing,
    )
