"""Cloud app installation and bounded completion checks."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Unpack

from mobilerun.clients.cloud import apps
from mobilerun.clients.cloud.apps import upload_app
from mobilerun.clients.cloud.client import AsyncMobilerun
from mobilerun.config.settings import get_settings
from mobilerun.core.devices.base import InstallOptions
from mobilerun.core.devices.components.cloud_lifecycle import _CloudLifecycle
from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.models.apps import App
from mobilerun.models.billing import Billing

logger = logging.getLogger(__name__)


class _CloudApps(_CloudLifecycle):
    _client: AsyncMobilerun
    _billing: Billing
    device_id: str | None

    async def install_apk(
        self, path: str | Path, **kwargs: Unpack[InstallOptions]
    ) -> App:
        await self._ensure_connected()
        self._install_timeout(dict(kwargs))
        app = await upload_app(
            path,
            billing=self._billing,
            api_key=self._client.api_key,
            base_url=str(self._client.base_url),
        )
        return await self.install_app(app, **kwargs)

    @staticmethod
    def _install_timeout(kwargs: dict[str, Any]) -> float:
        timeout = kwargs.pop("timeout", None)
        if timeout is None:
            timeout = get_settings().MOBILERUN_CLOUD_APP_INSTALL_TIMEOUT_SECONDS
        if timeout <= 0:
            raise ValueError("Cloud app install timeout must be positive")
        if kwargs.pop("reinstall", False):
            raise DeviceOperationNotSupported(
                "Cloud app installs do not support the reinstall option"
            )
        if not kwargs.pop("grant_permissions", True):
            raise DeviceOperationNotSupported(
                "Cloud app installs do not support changing permission grants"
            )
        if kwargs:
            raise TypeError(f"Unsupported cloud install options: {', '.join(kwargs)}")
        return timeout

    async def install_app(self, app: App, **kwargs: Unpack[InstallOptions]) -> App:
        await self._ensure_connected()
        timeout = self._install_timeout(dict(kwargs))
        if not app.id:
            if app.path is None:
                raise ValueError(
                    "This app has no local APK or cloud id; use install_apk with a path"
                )
            app = await upload_app(
                app.path,
                billing=self._billing,
                api_key=self._client.api_key,
                base_url=str(self._client.base_url),
            )
        params: dict[str, Any] = {"package_name": app.package}
        if app.version_code is not None:
            params["version_code"] = app.version_code
        try:
            async with asyncio.timeout(timeout):
                previous = await apps.list_installs(
                    self._client, self._require_device_id(), billing=self._billing
                )
                previous_starts = {
                    item.started_at
                    for item in (previous.installs or [])
                    if item.app_id == app.package
                }
                await apps.install(
                    self._client,
                    self._require_device_id(),
                    background=True,
                    billing=self._billing,
                    **params,
                )
                while True:
                    result = await apps.list_installs(
                        self._client, self._require_device_id(), billing=self._billing
                    )
                    attempts = [
                        item
                        for item in (result.installs or [])
                        if item.app_id == app.package
                        and item.started_at not in previous_starts
                    ]
                    if attempts:
                        attempt = max(attempts, key=lambda item: item.started_at)
                        logger.info("Installing %s: %s", app.package, attempt.status)
                        if attempt.status == "failed":
                            raise RuntimeError(
                                f"App install failed for {app.package}: {attempt.error_class or 'unknown error'}"
                            )
                        if attempt.status == "succeeded":
                            installed_apps = await self.list_apps(include_system=True)
                            if any(
                                item.get("package_name") == app.package
                                and (
                                    (
                                        app.version_code is not None
                                        and item.get("version_code") == app.version_code
                                    )
                                    or (
                                        app.version_code is None
                                        and (
                                            not app.version
                                            or item.get("version_name") == app.version
                                        )
                                    )
                                )
                                for item in installed_apps
                            ):
                                return app
                    await asyncio.sleep(0.5)
        except TimeoutError as exc:
            raise TimeoutError(
                f"App install timed out for {app.package} after {timeout}s"
            ) from exc

    async def uninstall_app(self, package: str) -> str:
        await self._ensure_connected()
        await apps.uninstall(
            self._client,
            package,
            billing=self._billing,
            device_id=self._require_device_id(),
        )
        return f"Uninstalled {package}"
