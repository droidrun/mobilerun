"""Resolve CLI config and device selection before invoking a backend."""

from dataclasses import dataclass

import click
from mobilerun_core import Mobilerun
from mobilerun_core.detect import is_uuid, resolve_backend
from mobilerun_core_local.driver.ios import discover_ios_device, validate_ios_portal_url

from mobilerun.agent.utils.android_device import (
    AndroidDeviceSelectionError,
    is_portal_http_url,
    resolve_android_serial,
)
from mobilerun.cli.options import DEFAULT_API_URL
from mobilerun.config_manager import ConfigLoader, MobileConfig


def resolve_cloud_api_key():
    import os

    from mobilerun.cli import deviceauth

    credentials = deviceauth.load_credentials() or {}
    return (
        os.environ.get("MOBILERUN_CLOUD_API_KEY")
        or credentials.get("api_key")
        or credentials.get("access_token")
    )


def core_client(api_url=DEFAULT_API_URL):
    return Mobilerun(bearer=resolve_cloud_api_key(), base_url=api_url)


@dataclass
class Selection:
    config: MobileConfig
    device: str | None
    platform: str
    backend: str


async def select_device(
    device=None,
    config_path=None,
    tcp=None,
    platform=None,
    target="local",
    *,
    discover=True,
    config=None,
):
    config = config if config is not None else ConfigLoader.load(config_path)
    selected = device if device is not None else config.device.serial
    selected_platform = platform or config.device.platform
    if tcp is not None:
        config.device.use_tcp = tcp
    if not selected:
        if target == "cloud":
            raise click.ClickException(
                "Cloud device required: pass -d <id>; find ids with `mobilerun device list -t cloud`."
            )
        if not discover and selected_platform != "ios":
            return Selection(config, None, selected_platform, "local-android-adb")
        if selected_platform == "ios":
            selected = await discover_ios_device()
        else:
            try:
                selected = await resolve_android_serial()
            except AndroidDeviceSelectionError as exc:
                raise click.ClickException(str(exc)) from exc
    url = selected if is_portal_http_url(selected) else None
    if selected_platform == "ios" and not is_uuid(selected):
        url = validate_ios_portal_url(selected)
        selected = url
    backend = resolve_backend(
        selected,
        platform=selected_platform,
        url=url,
        backend="local-android-adb" if not url and not is_uuid(selected) else None,
    )
    return Selection(config, selected, selected_platform, backend)
