"""Direct device action CLI commands.

Provides ``mobilerun device <action>`` subcommands that bypass the LLM agent
and talk directly to the device driver.
"""

import asyncio
import json
import logging
import os
import sys
import tempfile

import click
from async_adbutils import adb
from mobilerun_core_local.driver.android import AndroidDriver
from mobilerun_core_local.driver.android.portal import ensure_portal_ready
from mobilerun_core_local.driver.ios import (
    IOSPortalHttpDriver,
    create_ios_driver,
)
from rich.console import Console

from mobilerun.agent.utils.android_device import (
    create_android_portal_http_driver,
    is_portal_http_url,
)
from mobilerun.agent.utils.portal_setup import portal_version_kwargs
from mobilerun.cli.options import Group, coro, device_options, emit, json_option, verbose_option
from mobilerun.cli.selection import core_client, resolve_cloud_api_key, select_device
from mobilerun.log_handlers import configure_logging
from mobilerun.tools.filters import ConciseFilter
from mobilerun.tools.formatters import IndexedFormatter
from mobilerun.tools.ui.ios_provider import IOSStateProvider
from mobilerun.tools.ui.provider import AndroidStateProvider

console = Console()

DEFAULT_CLOUD_BASE_URL = "https://api.mobilerun.ai/v1"
CLOUD_API_KEY_ENV = "MOBILERUN_CLOUD_API_KEY"



class CoreDriver:
    """Expose the core facade through the framework's async state-provider interface."""

    def __init__(self, device, backend):
        self.core = device
        self.backend = backend

    async def screenshot(self):
        import base64
        from pathlib import Path

        value = await asyncio.to_thread(self.core.screenshot)
        if self.backend == "cloud":
            path = Path(value)
            try:
                return path.read_bytes()
            finally:
                path.unlink(missing_ok=True)
        return base64.b64decode(value)

    async def get_ui_tree(self, **kwargs):
        return await asyncio.to_thread(self.core.ui)

    async def tap(self, x, y):
        await asyncio.to_thread(self.core.tap, x, y, humanize=False)

    async def swipe(self, x1, y1, x2, y2, duration_ms=1000):
        await asyncio.to_thread(self.core.swipe, x1, y1, x2, y2, ms=duration_ms, humanize=False)

    async def input_text(self, text, clear=False):
        await asyncio.to_thread(self.core.type, text, clear=clear, humanize=False)
        return True

    async def press_button(self, button):
        await asyncio.to_thread(self.core.key, button)

    async def get_apps(self, include_system=False):
        return await asyncio.to_thread(self.core.list_apps, include_system_apps=include_system)

    async def start_app(self, package):
        await asyncio.to_thread(self.core.start_app, package)
        return f"Started {package}"


async def connect_driver(device=None, config_path=None, tcp=None, platform=None,
                         target="local", api_url=DEFAULT_CLOUD_BASE_URL, verbose=None, framework=False):
    _log_progress_to_stderr()
    selection = await select_device(device, config_path, tcp, platform, target)
    config = selection.config
    config.device.serial = selection.device
    config.device.platform = selection.platform
    if selection.backend == "local-android-adb":
        # The facade does not expose auto-setup, explicit TCP, or keyboard teardown.
        return await _connect_local_driver(selection.device, config, tcp, False)
    client = core_client(api_url)
    if selection.backend == "cloud" and not resolve_cloud_api_key():
        raise click.ClickException(
            "No cloud API key found. Set MOBILERUN_CLOUD_API_KEY or run `mobilerun auth login`."
        )
    if framework:
        # Preserve input_text's boolean failure result and the macro engine's
        # async driver verbs (core.type discards the underlying boolean).
        if selection.backend == "cloud":
            from mobilerun_core_local.driver.cloud import CloudDriver

            driver = CloudDriver(device_id=selection.device, api_key=resolve_cloud_api_key(), base_url=api_url)
        elif selection.backend == "local-ios-http":
            driver = await create_ios_driver(selection.device, token=config.device.resolve_auth_token())
        else:
            driver = create_android_portal_http_driver(selection.device, config.device.resolve_auth_token())
        await driver.connect()
        return driver, selection.platform == "ios"
    connected = client.connect(
        selection.device, backend=selection.backend, platform=selection.platform,
        url=selection.device if is_portal_http_url(selection.device) else None,
        token=config.device.resolve_auth_token(),
    )
    return CoreDriver(connected, selection.backend), selection.platform == "ios"


async def _connect_local_driver(device, config, tcp, ios):

    serial = device
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


def _log_progress_to_stderr() -> None:
    """Show INFO progress (e.g. a slow UI-state fetch) on stderr.

    Without this, Python's last-resort handler drops everything below
    WARNING, so a slow device looks like a silent hang. stdout stays
    reserved for the command's output.
    """
    if logging.getLogger("mobilerun").handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    configure_logging(debug=False, handler=handler)


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


@click.group(cls=Group)
@verbose_option
def device_cli(verbose):
    """Direct device actions (screenshot, tap, swipe, etc.)."""
    pass


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@device_cli.command()
@json_option
@device_options
@coro
async def screenshot(device, config_path, tcp, platform, target, api_url, verbose, json_output):
    """Take a screenshot and print the saved file path to stdout."""
    driver, _ = await connect_driver(
        device, config_path, tcp, platform, target, api_url, verbose
    )
    try:
        png_bytes = await driver.screenshot()
        fd, path = tempfile.mkstemp(prefix="mobilerun_", suffix=".png")
        try:
            os.write(fd, png_bytes)
        finally:
            os.close(fd)
        emit({"path": path} if json_output else path, json_output=json_output)
    finally:
        await _teardown_android(driver)


@device_cli.command()
@json_option
@device_options
@coro
async def ui(device, config_path, tcp, platform, target, api_url, verbose, json_output):
    """Print the UI accessibility tree with element bounds for targeting."""
    _log_progress_to_stderr()
    driver, is_ios = await connect_driver(
        device, config_path, tcp, platform, target, api_url, verbose
    )
    try:
        if isinstance(driver, CoreDriver):
            state = await driver.get_ui_tree()
            if json_output:
                emit(state, json_output=True)
            else:
                tree = state.get("a11y_tree", state)
                emit(tree if isinstance(tree, str) else json.dumps(state, indent=2))
            return
        if is_ios and not isinstance(driver, IOSPortalHttpDriver):
            provider = IOSStateProvider(driver)
        else:
            provider = AndroidStateProvider(
                driver,
                tree_filter=ConciseFilter(),
                tree_formatter=IndexedFormatter(),
            )
        state = await provider.get_state()
        if json_output:
            emit({"ui": state.formatted_text, "phone_state": state.phone_state}, json_output=True)
        else:
            emit(state.formatted_text)
            if state.phone_state:
                emit(f"\nPhone state: {state.phone_state}")
    finally:
        await _teardown_android(driver)


@device_cli.command()
@click.argument("x", type=int)
@click.argument("y", type=int)
@device_options
@coro
async def tap(x, y, device, config_path, tcp, platform, target, api_url, verbose):
    """Tap at screen coordinates."""
    driver, _ = await connect_driver(
        device, config_path, tcp, platform, target, api_url, verbose
    )
    try:
        await driver.tap(x, y)
        click.echo(f"Tapped ({x}, {y})", err=True)
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
    x1, y1, x2, y2, duration, device, config_path, tcp, platform, target, api_url, verbose
):
    """Swipe from (x1, y1) to (x2, y2)."""
    driver, _ = await connect_driver(
        device, config_path, tcp, platform, target, api_url, verbose
    )
    try:
        await driver.swipe(x1, y1, x2, y2, duration_ms=int(duration * 1000))
        click.echo(f"Swiped ({x1}, {y1}) -> ({x2}, {y2})", err=True)
    finally:
        await _teardown_android(driver)


@device_cli.command("long-press")
@click.argument("x", type=int)
@click.argument("y", type=int)
@device_options
@coro
async def long_press(x, y, device, config_path, tcp, platform, target, api_url, verbose):
    """Long press at screen coordinates."""
    if platform == "ios":
        raise click.ClickException("long-press is not supported on iOS")
    driver, is_ios = await connect_driver(
        device, config_path, tcp, platform, target, api_url, verbose
    )
    try:
        if is_ios:
            raise click.ClickException("long-press is not supported on iOS")
        await driver.swipe(x, y, x, y, 1000)
        click.echo(f"Long pressed ({x}, {y})", err=True)
    finally:
        await _teardown_android(driver)


@device_cli.command("type")
@click.argument("text")
@click.option("--clear", is_flag=True, default=False, help="Clear field before typing")
@device_options
@coro
async def type_text(
    text, clear, device, config_path, tcp, platform, target, api_url, verbose
):
    """Type text into the currently focused field. Use 'tap' first to focus."""
    driver, _ = await connect_driver(
        device, config_path, tcp, platform, target, api_url, verbose, framework=True
    )
    try:
        success = await driver.input_text(text, clear)
        if success:
            click.echo(f"Typed: {text}", err=True)
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
async def press(button, device, config_path, tcp, platform, target, api_url, verbose):
    """Press a system button."""
    driver, _ = await connect_driver(
        device, config_path, tcp, platform, target, api_url, verbose
    )
    try:
        await driver.press_button(button)
        click.echo(f"Pressed {button}", err=True)
    finally:
        await _teardown_android(driver)


@device_cli.command()
@click.option("--system/--no-system", default=False, help="Include system apps")
@json_option
@device_options
@coro
async def apps(system, device, config_path, tcp, platform, target, api_url, verbose, json_output):
    """List installed apps."""
    driver, _ = await connect_driver(
        device, config_path, tcp, platform, target, api_url, verbose
    )
    try:
        app_list = await driver.get_apps(include_system=system)
        if json_output:
            emit(app_list, json_output=True)
            return
        for app in app_list:
            label = app.get("label") or ""
            package = (
                app.get("package")
                or app.get("package_name")
                or app.get("packageName")
                or app.get("app_id")
                or app.get("bundle_id")
                or ""
            )
            if label and label != package:
                emit(f"{package}  ({label})")
            else:
                emit(package)
    finally:
        await _teardown_android(driver)


@device_cli.command()
@click.argument("package")
@device_options
@coro
async def start(package, device, config_path, tcp, platform, target, api_url, verbose):
    """Launch an app by package name."""
    driver, _ = await connect_driver(
        device, config_path, tcp, platform, target, api_url, verbose
    )
    try:
        result = await driver.start_app(package)
        click.echo(result, err=True)
    finally:
        await _teardown_android(driver)

