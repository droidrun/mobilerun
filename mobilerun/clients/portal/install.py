import asyncio
import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import cast

from async_adbutils import AdbDevice
from mobilerun_core_local.driver.android.portal import (
    DOWNLOAD_BASE,
    PORTAL_PACKAGE_NAME,
    check_portal_accessibility,
    download_portal_apk,
    download_versioned_portal_apk,
    enable_portal_accessibility,
    portal_content_uri,
    setup_portal,
)

from mobilerun.clients.portal.setup import portal_version_kwargs
from mobilerun.errors import PortalApkNotFound, PortalInstallFailed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortalSetupResult:
    installed: bool
    accessibility_enabled: bool
    needs_manual_accessibility: bool
    apk_source: str | None = None
    accessibility_error: str | None = None


async def wait_for_portal_service(
    device_obj: AdbDevice, timeout: float = 30.0, interval: float = 1.0
) -> bool:
    """Wait until Portal's accessibility service responds."""
    uri = portal_content_uri(PORTAL_PACKAGE_NAME, "ping")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            result = cast(str, await device_obj.shell(f"content query --uri {uri}"))
            if '"status":"success"' in result:
                return True
            if not await check_portal_accessibility(device_obj):
                await enable_portal_accessibility(device_obj)
        except (OSError, RuntimeError):
            logger.warning("Portal service probe failed; retrying.")
        await asyncio.sleep(interval)
    return False


async def install_portal(
    device_obj: AdbDevice,
    *,
    apk_path: str | None = None,
    version: str | None = None,
    latest: bool = False,
    in_place: bool = False,
    debug: bool = False,
    wait_timeout: float = 30.0,
) -> PortalSetupResult:
    """Install Portal and enable its accessibility service without rendering output."""
    if apk_path:
        apk_context = nullcontext(apk_path)
    elif version:
        normalized = f"v{version.lstrip('v')}"
        apk_context = download_versioned_portal_apk(normalized, DOWNLOAD_BASE, debug)
    elif latest:
        apk_context = download_portal_apk(debug)
    else:
        success = await setup_portal(
            device_obj, debug, **portal_version_kwargs(setup_portal, in_place=in_place)
        )
        if not success:
            raise PortalInstallFailed("Setup failed")
        try:
            enabled = await wait_for_portal_service(device_obj, timeout=wait_timeout)
        except Exception:
            enabled = False
        if not enabled:
            await device_obj.shell(
                "am start -a android.settings.ACCESSIBILITY_SETTINGS"
            )
        return PortalSetupResult(True, enabled, not enabled)

    with apk_context as downloaded_path:
        if not os.path.exists(downloaded_path):
            raise PortalApkNotFound(f"APK file not found at {downloaded_path}")
        try:
            await device_obj.install(
                downloaded_path,
                nolaunch=True,
                uninstall=True,
                flags=["-g"],
                silent=not debug,
            )
        except Exception as exc:
            raise PortalInstallFailed(str(exc), apk_path=downloaded_path) from exc

        accessibility_error = None
        try:
            await enable_portal_accessibility(device_obj)
            enabled = await wait_for_portal_service(device_obj, timeout=wait_timeout)
            if not enabled:
                raise RuntimeError("the service did not respond in time")
        except Exception as exc:
            accessibility_error = str(exc)
            await device_obj.shell(
                "am start -a android.settings.ACCESSIBILITY_SETTINGS"
            )
            enabled = False
        return PortalSetupResult(
            True, enabled, not enabled, downloaded_path, accessibility_error
        )
