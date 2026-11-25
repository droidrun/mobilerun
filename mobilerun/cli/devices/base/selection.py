"""Construct devices for direct CLI actions."""

import uuid

from async_adbutils import adb
from mobilerun_core_local.driver.android.portal import ensure_portal_ready

from mobilerun.clients.portal.http import (
    create_android_portal_http_driver,
    is_portal_http_url,
)
from mobilerun.clients.portal.setup import portal_version_kwargs
from mobilerun.config.loader import ConfigLoader
from mobilerun.config.user_config import MobileConfig
from mobilerun.core.devices import (
    AndroidDevice,
    Device,
    IOSDevice,
    RemoteDevice,
)
from mobilerun.core.devices.components.wrapper import DriverDevice
from mobilerun.core.state.options import android_state_provider


def detect_device_kind(target: str) -> str:
    """Infer a device type from an unambiguous positional target."""
    if target.lower().startswith(("http://", "https://")):
        raise ValueError(
            "URL target: add -i/--ios, -a/--android (Portal HTTP) or -v/--remote"
        )
    try:
        uuid.UUID(target)
    except (ValueError, AttributeError):
        return "android"
    return "cloud"


async def connect_target(
    target: str,
    kind: str | None = None,
    tcp: bool | None = None,
    base_url: str | None = None,
    config: MobileConfig | None = None,
    allow_auto_setup: bool = True,
) -> Device:
    """Build and connect the explicitly selected device."""
    kind = kind or detect_device_kind(target)
    if kind not in {"android", "ios", "cloud", "visual_remote"}:
        raise ValueError(f"Unknown device type: {kind}")
    if tcp is not None and kind != "android":
        raise ValueError("--tcp/--no-tcp may only be used with Android.")
    if base_url is not None and kind != "cloud":
        raise ValueError("--base-url may only be used with cloud.")
    device: Device
    if kind == "android":
        if is_portal_http_url(target):
            token = (config or ConfigLoader.load(None)).device.resolve_auth_token()
            driver = create_android_portal_http_driver(target, token)

            async def close_portal_client() -> None:
                if driver._client is not None:
                    await driver._client.aclose()
                    driver._client = None

            device = DriverDevice(
                driver,
                state_provider_factory=android_state_provider,
                close=close_portal_client,
            )
        else:
            if (
                allow_auto_setup
                and config is not None
                and config.device.auto_setup
                and config.device.portal_mode != "disabled"
            ):
                await ensure_portal_ready(
                    await adb.device(serial=target),
                    debug=config.logging.debug,
                    **portal_version_kwargs(ensure_portal_ready),
                )
            if config is None:
                device = AndroidDevice(serial=target, use_tcp=tcp)
            else:
                device = AndroidDevice(
                    serial=target, use_tcp=tcp, portal_mode=config.device.portal_mode
                )
    elif kind == "ios":
        token = config.device.resolve_auth_token() if config is not None else None
        device = IOSDevice(target, token=token)
    elif kind == "cloud":
        # lazy: the cloud driver is an optional dependency.
        from mobilerun.core.devices.cloud import CloudDevice

        device = CloudDevice(device_id=target, base_url=base_url)
    else:
        if config is None:
            device = RemoteDevice(url=target)
        else:
            device = RemoteDevice(url=target, device_id=config.device.device_id)
    try:
        await device.connect()
    except BaseException as exc:
        try:
            await device.close()
        except BaseException as cleanup_error:
            exc.add_note(f"Device cleanup failed: {type(cleanup_error).__name__}")
        raise
    return device
