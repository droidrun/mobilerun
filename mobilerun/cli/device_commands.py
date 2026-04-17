"""Direct device action CLI commands.

Provides ``mobilerun device <action>`` subcommands that bypass the LLM agent
and talk directly to the device driver.
"""

import asyncio
import os
import tempfile
from functools import wraps
from typing import Optional

import click
from async_adbutils import adb
from rich.console import Console

from mobilerun.config_manager import ConfigLoader
from mobilerun.portal import ensure_portal_ready
from mobilerun.tools.driver.android import AndroidDriver
from mobilerun.tools.driver.ios import (
    IOSDriver,
    discover_ios_portal,
    validate_ios_portal_url,
)
from mobilerun.tools.filters import ConciseFilter
from mobilerun.tools.formatters import IndexedFormatter
from mobilerun.tools.ui.ios_provider import IOSStateProvider
from mobilerun.tools.ui.provider import AndroidStateProvider

console = Console()


def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


def device_options(f):
    """Common device options for all action commands."""
    f = click.option(
        "--device", "-d", help="Device serial number or IP address", default=None
    )(f)
    f = click.option(
        "--config", "-c", "config_path", help="Path to config file", default=None
    )(f)
    f = click.option("--tcp/--no-tcp", default=None, help="Use TCP communication")(f)
    f = click.option("--ios", is_flag=True, default=False, help="Target iOS device")(f)
    return f


async def _create_driver(
    device: Optional[str],
    config_path: Optional[str],
    tcp: Optional[bool],
    ios: bool,
):
    """Create and connect a device driver based on CLI options."""
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
            url = await discover_ios_portal()
        driver = IOSDriver(url=url)
        await driver.connect()
        return driver, True

    serial = config.device.serial
    if serial is None:
        devices = await adb.list()
        if not devices:
            raise click.ClickException("No connected Android devices found.")
        serial = devices[0].serial

    if config.device.auto_setup:
        device_obj = await adb.device(serial=serial)
        await ensure_portal_ready(device_obj, debug=False)

    driver = AndroidDriver(serial=serial, use_tcp=config.device.use_tcp)
    await driver.connect()
    return driver, False


async def _teardown_android(driver):
    """Disable Mobilerun keyboard after direct command execution."""
    if isinstance(driver, AndroidDriver) and driver.device:
        from mobilerun.portal import PORTAL_PACKAGE_NAME, portal_ime_id
        try:
            ime = portal_ime_id(PORTAL_PACKAGE_NAME)
            await driver.device.shell(
                f"ime disable {ime}"
            )
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
async def screenshot(device, config_path, tcp, ios):
    """Take a screenshot and print the saved file path to stdout."""
    driver, _ = await _create_driver(device, config_path, tcp, ios)
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
async def ui(device, config_path, tcp, ios):
    """Print the UI accessibility tree with element bounds for targeting."""
    driver, is_ios = await _create_driver(device, config_path, tcp, ios)
    try:
        if is_ios:
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
async def tap(x, y, device, config_path, tcp, ios):
    """Tap at screen coordinates."""
    driver, _ = await _create_driver(device, config_path, tcp, ios)
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
async def swipe_cmd(x1, y1, x2, y2, duration, device, config_path, tcp, ios):
    """Swipe from (x1, y1) to (x2, y2)."""
    driver, _ = await _create_driver(device, config_path, tcp, ios)
    try:
        await driver.swipe(x1, y1, x2, y2, duration_ms=duration * 1000)
        click.echo(f"Swiped ({x1}, {y1}) -> ({x2}, {y2})")
    finally:
        await _teardown_android(driver)


@device_cli.command("long-press")
@click.argument("x", type=int)
@click.argument("y", type=int)
@device_options
@coro
async def long_press(x, y, device, config_path, tcp, ios):
    """Long press at screen coordinates."""
    if ios:
        raise click.ClickException("long-press is not supported on iOS")
    driver, _ = await _create_driver(device, config_path, tcp, ios)
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
async def type_text(text, clear, device, config_path, tcp, ios):
    """Type text into the currently focused field. Use 'tap' first to focus."""
    driver, _ = await _create_driver(device, config_path, tcp, ios)
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
async def press(button, device, config_path, tcp, ios):
    """Press a system button."""
    driver, _ = await _create_driver(device, config_path, tcp, ios)
    try:
        await driver.press_button(button)
        click.echo(f"Pressed {button}")
    finally:
        await _teardown_android(driver)


@device_cli.command()
@click.option("--system/--no-system", default=False, help="Include system apps")
@device_options
@coro
async def apps(system, device, config_path, tcp, ios):
    """List installed apps."""
    driver, _ = await _create_driver(device, config_path, tcp, ios)
    try:
        app_list = await driver.get_apps(include_system=system)
        for app in app_list:
            label = app.get("label", "")
            package = app.get("package", "")
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
async def start(package, device, config_path, tcp, ios):
    """Launch an app by package name."""
    driver, _ = await _create_driver(device, config_path, tcp, ios)
    try:
        result = await driver.start_app(package)
        click.echo(result)
    finally:
        await _teardown_android(driver)
