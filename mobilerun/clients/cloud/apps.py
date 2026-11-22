"""Upload Android packages and install cloud apps; neither is billable."""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TypedDict, Unpack

import httpx
from mobilerun_sdk import APIError, AsyncMobilerun
from mobilerun_sdk.types.app_list_response import Item
from mobilerun_sdk.types.app_retrieve_response import AppRetrieveResponse
from mobilerun_sdk.types.devices.app_list_installs_response import (
    AppListInstallsResponse,
)

from mobilerun.clients.cloud.billing import requires
from mobilerun.clients.cloud.client import cloud_client
from mobilerun.clients.cloud.errors import cloud_error
from mobilerun.clients.local.apk import inspect_apk
from mobilerun.models.apps import App
from mobilerun.models.billing import Billing

InstallParams = TypedDict(
    "InstallParams",
    {
        "bundle_id": str,
        "package_name": str,
        "background": bool,
        "country": str,
        "version_code": int,
        "x_device_display_id": int,
    },
    total=False,
)
UninstallParams = TypedDict("UninstallParams", {"device_id": str})


@requires(Billing.NO_COMMERCE)
async def get_app(
    client: AsyncMobilerun, app_id: str, *, billing: Billing
) -> AppRetrieveResponse:
    """Retrieve an app library entry."""
    try:
        return await client.apps.retrieve(app_id)
    except APIError as exc:
        raise cloud_error(exc, "Retrieve cloud app") from exc


async def _items(client: AsyncMobilerun) -> AsyncIterator[Item]:
    """Iterate every app library page."""
    page = 1
    try:
        while True:
            result = await client.apps.list(page=page, page_size=100)
            for item in result.items:
                yield item
            if not result.pagination.has_next:
                break
            page += 1
    except APIError as exc:
        raise cloud_error(exc, "List cloud apps") from exc


@requires(Billing.NO_COMMERCE)
async def list_apps(client: AsyncMobilerun, *, billing: Billing) -> list[App]:
    return [
        App(
            item.bundle_id,
            item.version.version_name or "",
            item.display_name,
            id=item.id,
            version_code=item.version.version_code,
        )
        async for item in _items(client)
    ]


@requires(Billing.NO_COMMERCE)
async def delete_app(client: AsyncMobilerun, app_id: str, *, billing: Billing) -> None:
    try:
        await client.apps.delete(app_id)
    except APIError as exc:
        raise cloud_error(exc, "Delete workspace app") from exc


@requires(Billing.NO_COMMERCE)
async def upload_app(
    path: str | Path,
    *,
    billing: Billing,
    api_key: str | None = None,
    base_url: str | None = None,
) -> App:
    """Upload an APK; version metadata alone cannot verify identical content."""
    path = Path(path)

    def prepare():
        metadata = inspect_apk(path)
        content = path.read_bytes()
        return metadata, content, hashlib.sha256(content).hexdigest()

    (package, code, version, name), content, digest = await asyncio.to_thread(prepare)
    async with cloud_client(api_key, base_url, billing=billing) as client:
        try:
            signed = await client.apps.create_signed_upload_url(
                bundle_id=package,
                display_name=name,
                version_code=code,
                version_name=version,
                platform="android",
                files=[
                    {
                        "file_name": path.name,
                        "content_type": "application/vnd.android.package-archive",
                        "sha256": digest,
                    }
                ],
            )
        except APIError as exc:
            raise cloud_error(exc, "Prepare cloud app upload") from exc
        try:
            url = next(
                entry.r2_upload_url
                for entry in signed.r2_upload_urls
                if entry.file_name == path.name
            )
            async with httpx.AsyncClient() as http:
                response = await http.put(
                    url,
                    content=content,
                    headers={"Content-Type": "application/vnd.android.package-archive"},
                    timeout=None,
                )
                response.raise_for_status()
            await client.apps.confirm_upload(signed.version_id)
        except BaseException as exc:
            try:
                await client.apps.mark_failed(signed.version_id)
            except APIError:
                pass
            if isinstance(exc, (APIError, httpx.HTTPError)):
                raise cloud_error(exc, "Upload cloud app") from exc
            raise
        return App(
            package, version, name, id=signed.app_id, version_code=code, path=path
        )


@requires(Billing.NO_COMMERCE)
async def list_installs(
    client: AsyncMobilerun, device_id: str, *, billing: Billing
) -> AppListInstallsResponse:
    """Poll app installations."""
    try:
        return await client.devices.apps.list_installs(device_id)
    except APIError as exc:
        raise cloud_error(exc, "List phone app installs", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def install(
    client: AsyncMobilerun,
    device_id: str,
    *,
    billing: Billing,
    **params: Unpack[InstallParams],
) -> None:
    """Install an app on an existing phone."""
    try:
        return await client.devices.apps.install(device_id, **params)
    except APIError as exc:
        raise cloud_error(exc, "Install phone app", device=True) from exc


@requires(Billing.NO_COMMERCE)
async def uninstall(
    client: AsyncMobilerun,
    package: str,
    *,
    billing: Billing,
    **params: Unpack[UninstallParams],
) -> None:
    """Uninstall an app from a phone."""
    try:
        # The deployed endpoint requires a JSON body even though the generated
        # SDK describes DELETE as bodyless. An empty object has no side effects.
        return await client.devices.apps.delete(package, extra_body={}, **params)
    except APIError as exc:
        raise cloud_error(exc, "Uninstall phone app", device=True) from exc
