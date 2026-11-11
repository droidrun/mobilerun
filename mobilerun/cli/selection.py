"""Resolve CLI config and device selection before invoking a backend."""

from __future__ import annotations

from dataclasses import dataclass, field

import click

from mobilerun.cli import options
from mobilerun.cli.options import DEFAULT_API_URL


@dataclass(frozen=True)
class CloudCredential:
    value: str | None = field(repr=False)
    source: str | None
    saved_sign_in: bool


def resolve_cloud_credential():
    """Resolve the cloud bearer and its source without changing precedence."""
    import os

    from mobilerun.cli import deviceauth

    credentials = deviceauth.load_credentials() or {}
    saved = credentials.get("api_key") or credentials.get("access_token")
    env_key = os.environ.get("MOBILERUN_CLOUD_API_KEY")
    if env_key:
        return CloudCredential(env_key, "env", bool(saved))
    return CloudCredential(saved, "login" if saved else None, bool(saved))


def resolve_cloud_api_key():
    return resolve_cloud_credential().value


def core_client(api_url=DEFAULT_API_URL, *, cloud=False):
    import mobilerun_core

    return mobilerun_core.Mobilerun(
        bearer=require_cloud_api_key() if cloud else resolve_cloud_api_key(), base_url=api_url
    )


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
    from mobilerun_core.detect import is_uuid, resolve_backend
    from mobilerun_core_local.driver.ios import discover_ios_device, validate_ios_portal_url

    from mobilerun.agent.utils.android_device import (
        AndroidDeviceSelectionError,
        is_portal_http_url,
        resolve_android_serial,
    )
    from mobilerun.config_manager import ConfigLoader

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
            options.require_adb()
            return Selection(config, None, selected_platform, "local-android-adb")
        if selected_platform == "ios":
            selected = await discover_ios_device()
        else:
            options.require_adb()
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
    if backend == "local-android-adb":
        options.require_adb()
    return Selection(config, selected, selected_platform, backend)


def require_cloud_api_key():
    key = resolve_cloud_api_key()
    if not key:
        raise click.ClickException(
            "No cloud API key found. Set MOBILERUN_CLOUD_API_KEY or run `mobilerun auth login`."
        )
    return key


def portal_token(selection):
    from mobilerun.config_manager.config_manager import PORTAL_TOKEN_ENV_VARS

    token = selection.config.device.resolve_auth_token(selection.platform)
    if selection.backend == "local-android-http" and not token:
        raise click.ClickException(
            f"Portal URL requires {PORTAL_TOKEN_ENV_VARS[selection.platform]} or device.auth_token in config"
        )
    return token


async def _native_driver(selection, config, api_url):
    """Connect the framework driver used by agents and macro replay."""
    from async_adbutils import adb
    from mobilerun_core_local.driver.android import AndroidDriver
    from mobilerun_core_local.driver.android.portal import ensure_portal_ready
    from mobilerun_core_local.driver.cloud import CloudDriver
    from mobilerun_core_local.driver.ios import create_ios_driver

    from mobilerun.agent.utils.android_device import create_android_portal_http_driver
    from mobilerun.agent.utils.portal_setup import portal_version_kwargs

    if selection.backend == "cloud":
        driver = CloudDriver(device_id=selection.device, api_key=require_cloud_api_key(), base_url=api_url)
    elif selection.backend == "local-ios-http":
        driver = await create_ios_driver(selection.device, token=portal_token(selection))
    elif selection.backend == "local-android-http":
        driver = create_android_portal_http_driver(selection.device, portal_token(selection))
    else:
        options.require_adb()
        if config.device.auto_setup and config.device.portal_mode != "disabled":
            device = await adb.device(serial=selection.device)
            await ensure_portal_ready(device, debug=False, **portal_version_kwargs(ensure_portal_ready))
        driver = AndroidDriver(serial=selection.device, use_tcp=config.device.use_tcp,
                               portal_mode=config.device.portal_mode)
    await driver.connect()
    return driver


async def teardown_native_driver(driver):
    """Disable the framework keyboard after macro replay on local Android."""
    from mobilerun_core_local.driver.android import AndroidDriver
    from mobilerun_core_local.driver.android.portal import PORTAL_PACKAGE_NAME, portal_ime_id

    if isinstance(driver, AndroidDriver) and driver.device:
        try:
            await driver.device.shell(f"ime disable {portal_ime_id(PORTAL_PACKAGE_NAME)}")
        except Exception:
            pass


from mobilerun.cli.options import _lazy_imports  # noqa: E402

__getattr__ = _lazy_imports(globals(), {
    'mobilerun_core': ('mobilerun_core', None),
    'is_uuid': ('mobilerun_core.detect', 'is_uuid'),
    'resolve_backend': ('mobilerun_core.detect', 'resolve_backend'),
    'discover_ios_device': ('mobilerun_core_local.driver.ios', 'discover_ios_device'),
    'validate_ios_portal_url': ('mobilerun_core_local.driver.ios', 'validate_ios_portal_url'),
    'AndroidDeviceSelectionError': ('mobilerun.agent.utils.android_device', 'AndroidDeviceSelectionError'),
    'is_portal_http_url': ('mobilerun.agent.utils.android_device', 'is_portal_http_url'),
    'resolve_android_serial': ('mobilerun.agent.utils.android_device', 'resolve_android_serial'),
    'ConfigLoader': ('mobilerun.config_manager', 'ConfigLoader'),
    'MobileConfig': ('mobilerun.config_manager', 'MobileConfig'),
    'PORTAL_TOKEN_ENV_VARS': ('mobilerun.config_manager.config_manager', 'PORTAL_TOKEN_ENV_VARS'),
})

from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    import mobilerun_core as mobilerun_core
    from mobilerun_core.detect import is_uuid as is_uuid
    from mobilerun_core.detect import resolve_backend as resolve_backend
    from mobilerun_core_local.driver.ios import discover_ios_device as discover_ios_device
    from mobilerun_core_local.driver.ios import validate_ios_portal_url as validate_ios_portal_url

    from mobilerun.agent.utils.android_device import (
        AndroidDeviceSelectionError as AndroidDeviceSelectionError,
    )
    from mobilerun.agent.utils.android_device import is_portal_http_url as is_portal_http_url
    from mobilerun.agent.utils.android_device import (
        resolve_android_serial as resolve_android_serial,
    )
    from mobilerun.config_manager import ConfigLoader as ConfigLoader
    from mobilerun.config_manager import MobileConfig as MobileConfig
    from mobilerun.config_manager.config_manager import (
        PORTAL_TOKEN_ENV_VARS as PORTAL_TOKEN_ENV_VARS,
    )
