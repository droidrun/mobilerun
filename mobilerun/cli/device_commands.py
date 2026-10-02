"""Direct device action CLI commands.

Provides ``mobilerun device <action>`` subcommands that bypass the LLM agent
and talk directly to the device driver.
"""

import asyncio
import json
import os
import re
import tempfile
from functools import wraps
from typing import Optional

import click
from async_adbutils import adb
from mobilerun_core_local.driver.android import AndroidDriver
from mobilerun_core_local.driver.android.portal import ensure_portal_ready
from mobilerun_core_local.driver.ios import (
    IOSPortalHttpDriver,
    create_ios_driver,
    discover_ios_device,
    validate_ios_portal_url,
)
from rich.console import Console

from mobilerun.agent.utils.portal_setup import portal_version_kwargs
from mobilerun.config_manager import ConfigLoader
from mobilerun.tools.filters import ConciseFilter
from mobilerun.tools.formatters import IndexedFormatter
from mobilerun.tools.ui.ios_provider import IOSStateProvider
from mobilerun.tools.ui.provider import AndroidStateProvider

console = Console()

DEFAULT_CLOUD_BASE_URL = "https://api.mobilerun.ai/v1"
CLOUD_API_KEY_ENV = "MOBILERUN_CLOUD_API_KEY"

# Cloud device ids are UUIDs (RFC 4122); ADB serials never are. Used to
# auto-route `-d <id>` to the cloud driver when the user didn't pass `--cloud`.
_CLOUD_DEVICE_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _looks_like_cloud_device_id(value: Optional[str]) -> bool:
    return bool(value and _CLOUD_DEVICE_ID_RE.match(value))


def resolve_cloud_api_key() -> Optional[str]:
    """Resolve the Mobilerun cloud API key for ``--cloud`` commands.

    Order: the ``MOBILERUN_CLOUD_API_KEY`` env var, then the credential persisted by
    ``mobilerun login`` (``<config dir>/credentials/mobilerun-cloud.json``).
    """
    env_key = os.environ.get(CLOUD_API_KEY_ENV)
    if env_key:
        return env_key
    try:
        from mobilerun.config_manager.credential_paths import OAUTH_CREDENTIAL_DIR

        cred_path = OAUTH_CREDENTIAL_DIR / "mobilerun-cloud.json"
        if cred_path.exists():
            data = json.loads(cred_path.read_text())
            key = data.get("api_key") or data.get("access_token")
            if key:
                return key
    except Exception:
        pass
    return None


def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


def device_options(f):
    """Common device options for all action commands."""
    f = click.option(
        "--device",
        "-d",
        help="Device serial/IP (local) or cloud device UUID "
        "(auto-detected when authenticated)",
        default=None,
    )(f)
    f = click.option(
        "--config", "-c", "config_path", help="Path to config file", default=None
    )(f)
    f = click.option("--tcp/--no-tcp", default=None, help="Use TCP communication")(f)
    f = click.option("--ios", is_flag=True, default=False, help="Target iOS device")(f)
    f = click.option(
        "--cloud",
        is_flag=True,
        default=False,
        help="Target a Mobilerun cloud device instead of a local one",
    )(f)
    f = click.option(
        "--device-id",
        "device_id",
        default=None,
        help="Cloud device id (-d also works; auto-detected when value is a UUID)",
    )(f)
    f = click.option(
        "--base-url",
        "base_url",
        default=None,
        help=f"Cloud API base URL (default {DEFAULT_CLOUD_BASE_URL})",
    )(f)
    return f


async def _create_driver(
    device: Optional[str],
    config_path: Optional[str],
    tcp: Optional[bool],
    ios: bool,
    cloud: bool = False,
    device_id: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """Create and connect a device driver based on CLI options."""
    # Auto-route to cloud when -d/--device-id is a UUID and a credential exists.
    # Mirrors `mobilerun devices` (which includes cloud when authenticated), so
    # users don't have to remember `--cloud` on every per-action subcommand.
    if not cloud and not ios:
        candidate = device_id or device
        if _looks_like_cloud_device_id(candidate) and resolve_cloud_api_key():
            cloud = True

    if cloud:
        if ios:
            raise click.ClickException("--cloud and --ios are mutually exclusive.")
        cloud_device_id = device_id or device
        if not cloud_device_id:
            raise click.ClickException(
                "Cloud device id required: pass -d <id> or --device-id <id>."
            )
        api_key = resolve_cloud_api_key()
        if not api_key:
            raise click.ClickException(
                f"No cloud API key found. Set {CLOUD_API_KEY_ENV} or run "
                "`mobilerun login`."
            )
        from mobilerun_core_local.driver.cloud import CloudDriver

        driver = CloudDriver(
            device_id=cloud_device_id,
            api_key=api_key,
            base_url=base_url or DEFAULT_CLOUD_BASE_URL,
        )
        await driver.connect()
        return driver, False

    config = ConfigLoader.load(config_path)

    if device is not None:
        config.device.serial = device
    if tcp is not None:
        config.device.use_tcp = tcp
    if ios:
        config.device.platform = "ios"

    is_ios = config.device.platform.lower() == "ios"

    if is_ios:
        if config.device.serial:
            url = validate_ios_portal_url(config.device.serial)
        else:
            url = await discover_ios_device()
        driver = await create_ios_driver(url, token=config.device.resolve_auth_token())
        await driver.connect()
        return driver, True

    serial = config.device.serial
    if serial is None:
        devices = await adb.list()
        if not devices:
            raise click.ClickException("No connected Android devices found.")
        serial = devices[0].serial

    if config.device.auto_setup and config.device.portal_mode != "disabled":
        device_obj = await adb.device(serial=serial)
        await ensure_portal_ready(
            device_obj, debug=False, **portal_version_kwargs(ensure_portal_ready)
        )

    driver = AndroidDriver(
        serial=serial,
        use_tcp=config.device.use_tcp,
        portal_mode=config.device.portal_mode,
    )
    await driver.connect()
    return driver, False


async def _teardown_android(driver):
    """Disable Mobilerun keyboard after direct command execution."""
    if isinstance(driver, AndroidDriver) and driver.device:
        from mobilerun_core_local.driver.android.portal import (
            PORTAL_PACKAGE_NAME,
            portal_ime_id,
        )

        try:
            ime = portal_ime_id(PORTAL_PACKAGE_NAME)
            await driver.device.shell(f"ime disable {ime}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Click group
# ---------------------------------------------------------------------------


@click.group()
def device_cli():
    """Direct device actions (screenshot, tap, swipe, etc.)."""
    pass


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@device_cli.command()
@device_options
@coro
async def screenshot(device, config_path, tcp, ios, cloud, device_id, base_url):
    """Take a screenshot and print the saved file path to stdout."""
    driver, _ = await _create_driver(
        device, config_path, tcp, ios, cloud, device_id, base_url
    )
    try:
        png_bytes = await driver.screenshot()
        fd, path = tempfile.mkstemp(prefix="mobilerun_", suffix=".png")
        try:
            os.write(fd, png_bytes)
        finally:
            os.close(fd)
        click.echo(path)
    finally:
        await _teardown_android(driver)


@device_cli.command()
@device_options
@coro
async def ui(device, config_path, tcp, ios, cloud, device_id, base_url):
    """Print the UI accessibility tree with element bounds for targeting."""
    driver, is_ios = await _create_driver(
        device, config_path, tcp, ios, cloud, device_id, base_url
    )
    try:
        if is_ios and not isinstance(driver, IOSPortalHttpDriver):
            provider = IOSStateProvider(driver)
        else:
            provider = AndroidStateProvider(
                driver,
                tree_filter=ConciseFilter(),
                tree_formatter=IndexedFormatter(),
            )
        state = await provider.get_state()
        click.echo(state.formatted_text)
        if state.phone_state:
            click.echo(f"\nPhone state: {state.phone_state}")
    finally:
        await _teardown_android(driver)


@device_cli.command()
@click.argument("x", type=int)
@click.argument("y", type=int)
@device_options
@coro
async def tap(x, y, device, config_path, tcp, ios, cloud, device_id, base_url):
    """Tap at screen coordinates."""
    driver, _ = await _create_driver(
        device, config_path, tcp, ios, cloud, device_id, base_url
    )
    try:
        await driver.tap(x, y)
        click.echo(f"Tapped ({x}, {y})")
    finally:
        await _teardown_android(driver)


@device_cli.command("swipe")
@click.argument("x1", type=int)
@click.argument("y1", type=int)
@click.argument("x2", type=int)
@click.argument("y2", type=int)
@click.option(
    "--duration", type=float, default=1.0, show_default=True, help="Duration in seconds"
)
@device_options
@coro
async def swipe_cmd(
    x1, y1, x2, y2, duration, device, config_path, tcp, ios, cloud, device_id, base_url
):
    """Swipe from (x1, y1) to (x2, y2)."""
    driver, _ = await _create_driver(
        device, config_path, tcp, ios, cloud, device_id, base_url
    )
    try:
        await driver.swipe(x1, y1, x2, y2, duration_ms=int(duration * 1000))
        click.echo(f"Swiped ({x1}, {y1}) -> ({x2}, {y2})")
    finally:
        await _teardown_android(driver)


@device_cli.command("long-press")
@click.argument("x", type=int)
@click.argument("y", type=int)
@device_options
@coro
async def long_press(x, y, device, config_path, tcp, ios, cloud, device_id, base_url):
    """Long press at screen coordinates."""
    if ios:
        raise click.ClickException("long-press is not supported on iOS")
    driver, _ = await _create_driver(
        device, config_path, tcp, ios, cloud, device_id, base_url
    )
    try:
        await driver.swipe(x, y, x, y, 1000)
        click.echo(f"Long pressed ({x}, {y})")
    finally:
        await _teardown_android(driver)


@device_cli.command("type")
@click.argument("text")
@click.option("--clear", is_flag=True, default=False, help="Clear field before typing")
@device_options
@coro
async def type_text(
    text, clear, device, config_path, tcp, ios, cloud, device_id, base_url
):
    """Type text into the currently focused field. Use 'tap' first to focus."""
    driver, _ = await _create_driver(
        device, config_path, tcp, ios, cloud, device_id, base_url
    )
    try:
        success = await driver.input_text(text, clear)
        if success:
            click.echo(f"Typed: {text}")
        else:
            raise click.ClickException("Failed to type text")
    finally:
        await _teardown_android(driver)


@device_cli.command()
@click.argument(
    "button", type=click.Choice(["back", "home", "enter"], case_sensitive=False)
)
@device_options
@coro
async def press(button, device, config_path, tcp, ios, cloud, device_id, base_url):
    """Press a system button."""
    driver, _ = await _create_driver(
        device, config_path, tcp, ios, cloud, device_id, base_url
    )
    try:
        await driver.press_button(button)
        click.echo(f"Pressed {button}")
    finally:
        await _teardown_android(driver)


@device_cli.command()
@click.option("--system/--no-system", default=False, help="Include system apps")
@device_options
@coro
async def apps(system, device, config_path, tcp, ios, cloud, device_id, base_url):
    """List installed apps."""
    driver, _ = await _create_driver(
        device, config_path, tcp, ios, cloud, device_id, base_url
    )
    try:
        app_list = await driver.get_apps(include_system=system)
        for app in app_list:
            label = app.get("label") or ""
            package = (
                app.get("package")
                or app.get("package_name")
                or app.get("packageName")
                or ""
            )
            if label and label != package:
                click.echo(f"{package}  ({label})")
            else:
                click.echo(package)
    finally:
        await _teardown_android(driver)


@device_cli.command()
@click.argument("package")
@device_options
@coro
async def start(package, device, config_path, tcp, ios, cloud, device_id, base_url):
    """Launch an app by package name."""
    driver, _ = await _create_driver(
        device, config_path, tcp, ios, cloud, device_id, base_url
    )
    try:
        result = await driver.start_app(package)
        click.echo(result)
    finally:
        await _teardown_android(driver)
