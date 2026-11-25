"""Cloud workspace file transport."""

import mimetypes
import posixpath
from pathlib import Path

import httpx
from mobilerun_sdk import APIError, AsyncMobilerun
from mobilerun_sdk.types.file_confirm_response import FileConfirmResponse
from mobilerun_sdk.types.file_list_response import File

from mobilerun.clients.cloud.billing import requires
from mobilerun.clients.cloud.errors import cloud_error
from mobilerun.clients.cloud.signed_http import _signed_client, _signed_transfer_logs
from mobilerun.models import Billing, DeviceFile, WorkspaceFile


def _file(item: File | FileConfirmResponse) -> WorkspaceFile:
    return WorkspaceFile(
        item.file_id,
        item.filename,
        item.mime_type,
        int(item.size_bytes),
        item.created_at,
    )


@requires(Billing.NO_COMMERCE)
async def upload_file(
    client: AsyncMobilerun, path: str | Path, *, billing: Billing
) -> WorkspaceFile:
    path = Path(path)
    content = path.read_bytes()
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    try:
        signed = await client.files.upload_url(
            filename=path.name,
            mime_type=mime_type,
            size_bytes=len(content),
            zone="user",
        )
    except APIError as exc:
        raise cloud_error(exc, "Prepare file upload") from exc
    try:
        async with _signed_client(httpx.AsyncClient()) as http:
            response = await http.put(
                signed.put_url,
                content=content,
                headers={"Content-Type": mime_type},
                timeout=None,
            )
            response.raise_for_status()
        return _file(await client.files.confirm(signed.file_id))
    except BaseException as exc:
        try:
            await client.files.cancel_pending(signed.file_id)
        except Exception:
            pass
        if isinstance(exc, (APIError, httpx.HTTPError, httpx.InvalidURL)):
            raise cloud_error(exc, "Upload file") from None
        raise


@requires(Billing.NO_COMMERCE)
async def list_files(
    client: AsyncMobilerun, *, billing: Billing
) -> list[WorkspaceFile]:
    try:
        result = await client.files.list(zone="user")
        return [_file(item) for item in result.files]
    except APIError as exc:
        raise cloud_error(exc, "List files") from exc


@requires(Billing.NO_COMMERCE)
async def download_file(
    client: AsyncMobilerun, file_id: str, *, billing: Billing
) -> bytes:
    try:
        with _signed_transfer_logs():
            response = await client.files.with_raw_response.download(file_id)
            return await response.read()
    except (APIError, httpx.HTTPError) as exc:
        raise cloud_error(exc, "Download file") from None


@requires(Billing.NO_COMMERCE)
async def delete_file(
    client: AsyncMobilerun, file_id: str, *, billing: Billing
) -> None:
    try:
        await client.files.delete(file_id)
    except APIError as exc:
        raise cloud_error(exc, "Delete file") from exc


@requires(Billing.NO_COMMERCE)
async def push_device_file(
    client: AsyncMobilerun,
    device_id: str,
    path: Path,
    directory: str,
    *,
    billing: Billing,
) -> None:
    try:
        await client.devices.files.upload(device_id, path=directory, file=path)
    except APIError as exc:
        raise cloud_error(exc, "Upload device file", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def list_device_files(
    client: AsyncMobilerun, device_id: str, directory: str, *, billing: Billing
) -> list[DeviceFile]:
    try:
        result = await client.devices.files.list(device_id, path=directory)
        return [
            DeviceFile(
                posixpath.join(directory, item.name),
                item.name,
                item.size,
                item.modified_at,
            )
            for item in result.files or []
        ]
    except APIError as exc:
        raise cloud_error(exc, "List device files", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def download_device_file(
    client: AsyncMobilerun, device_id: str, remote_path: str, *, billing: Billing
) -> bytes:
    try:
        with _signed_transfer_logs():
            response = await client.devices.files.with_raw_response.download(
                device_id, path=remote_path
            )
            return await response.read()
    except (APIError, httpx.HTTPError) as exc:
        raise cloud_error(exc, "Download device file", device=True) from None


@requires(Billing.NO_COMMERCE)
async def delete_device_file(
    client: AsyncMobilerun, device_id: str, remote_path: str, *, billing: Billing
) -> None:
    try:
        await client.devices.files.delete(device_id, path=remote_path)
    except APIError as exc:
        raise cloud_error(exc, "Delete device file", device=True) from exc
