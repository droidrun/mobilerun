"""Account-wide files and applications, using default or explicit credentials."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from mobilerun.models import App, Billing, DeviceFile, WorkspaceFile

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device


def __getattr__(name: str) -> Any:
    if name == "cloud_client":
        return import_module("mobilerun.clients.cloud.client").cloud_client
    if name in ("cloud_apps", "cloud_files"):
        return import_module(f"mobilerun.clients.cloud.{name[6:]}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class Workspace:
    """Manage assets in an account, independently of any particular device.

    Credentials default to MOBILERUN_API_KEY, then saved login. Overrides select
    the source workspace; a destination device keeps its own account and billing.
    """

    @classmethod
    async def upload_file(
        cls,
        path: str | Path,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        billing: Billing = Billing.NO_COMMERCE,
    ) -> WorkspaceFile:
        from mobilerun.core.workspace import cloud_client, cloud_files

        async with cloud_client(api_key, base_url, billing=billing) as client:
            return await cloud_files.upload_file(client, path, billing=billing)

    @classmethod
    async def list_files(
        cls,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        billing: Billing = Billing.NO_COMMERCE,
    ) -> list[WorkspaceFile]:
        from mobilerun.core.workspace import cloud_client, cloud_files

        async with cloud_client(api_key, base_url, billing=billing) as client:
            return await cloud_files.list_files(client, billing=billing)

    @classmethod
    async def download_file(
        cls,
        file_id: str,
        dest: str | Path = ".",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        billing: Billing = Billing.NO_COMMERCE,
    ) -> Path:
        from mobilerun.core.workspace import cloud_client, cloud_files

        async with cloud_client(api_key, base_url, billing=billing) as client:
            items = await cloud_files.list_files(client, billing=billing)
            item = next((item for item in items if item.id == file_id), None)
            if item is None:
                raise FileNotFoundError(f"Workspace file not found: {file_id}")
            destination = Path(dest)
            if destination.is_dir() or str(dest).endswith("/"):
                destination.mkdir(parents=True, exist_ok=True)
                destination /= Path(item.name).name
            data = await cloud_files.download_file(client, file_id, billing=billing)
            destination.write_bytes(data)
            return destination

    @classmethod
    async def delete_file(
        cls,
        file_id: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        billing: Billing = Billing.NO_COMMERCE,
    ) -> None:
        from mobilerun.core.workspace import cloud_client, cloud_files

        async with cloud_client(api_key, base_url, billing=billing) as client:
            await cloud_files.delete_file(client, file_id, billing=billing)

    @classmethod
    async def push_file(
        cls,
        file: WorkspaceFile,
        device: Device,
        dest: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        billing: Billing = Billing.NO_COMMERCE,
    ) -> DeviceFile:
        from mobilerun.core.workspace import cloud_client, cloud_files

        async with cloud_client(api_key, base_url, billing=billing) as client:
            data = await cloud_files.download_file(client, file.id, billing=billing)
        with TemporaryDirectory() as directory:
            path = Path(directory) / Path(file.name).name
            path.write_bytes(data)
            return await device.push_file(path, dest)

    @classmethod
    async def upload_app(
        cls,
        path: str | Path,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        billing: Billing = Billing.NO_COMMERCE,
    ) -> App:
        from mobilerun.core.workspace import cloud_apps

        return await cloud_apps.upload_app(
            path, api_key=api_key, base_url=base_url, billing=billing
        )

    @classmethod
    async def list_apps(
        cls,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        billing: Billing = Billing.NO_COMMERCE,
    ) -> list[App]:
        from mobilerun.core.workspace import cloud_apps, cloud_client

        async with cloud_client(api_key, base_url, billing=billing) as client:
            return await cloud_apps.list_apps(client, billing=billing)

    @classmethod
    async def get_app(
        cls,
        app_id: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        billing: Billing = Billing.NO_COMMERCE,
    ) -> App:
        from mobilerun.core.workspace import cloud_apps, cloud_client

        async with cloud_client(api_key, base_url, billing=billing) as client:
            result = await cloud_apps.get_app(client, app_id, billing=billing)
            return App(result.data.bundle_id, "", result.data.display_name, id=app_id)

    @classmethod
    async def delete_app(
        cls,
        app_id: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        billing: Billing = Billing.NO_COMMERCE,
    ) -> None:
        """Delete an uploaded app and its files/metadata, not a phone install."""
        from mobilerun.core.workspace import cloud_apps, cloud_client

        async with cloud_client(api_key, base_url, billing=billing) as client:
            await cloud_apps.delete_app(client, app_id, billing=billing)
