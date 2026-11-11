"""Synchronous device actions through mobilerun-core."""

import asyncio
import base64
import json
import tempfile
from contextlib import contextmanager

import click

from mobilerun.agent.utils.android_device import is_portal_http_url
from mobilerun.cli.options import Group, device_options, emit, json_option, verbose_option
from mobilerun.cli.selection import core_client, portal_token, select_device


@contextmanager
def device_driver(params=None, *, selection=None):
    """Resolve Click's device options once and yield the connected core Device."""
    params = params if params is not None else click.get_current_context().params
    if selection is None:
        selection = asyncio.run(select_device(
            device=params.get("device"), config_path=params.get("config_path"),
            platform=params.get("platform"), target=params.get("target", "local"),
        ))
    token = portal_token(selection)
    client = core_client(params["api_url"], cloud=selection.backend == "cloud")
    try:
        connected = client.connect(
            selection.device, backend=selection.backend, platform=selection.platform,
            url=selection.device if is_portal_http_url(selection.device) else None,
            token=token,
        )
        try:
            yield connected
        finally:
            close = getattr(connected, "close", None)
            if close:
                close()
    finally:
        close = getattr(client, "close", None)
        if close:
            close()


def probe_device(params, *, selection=None):
    with device_driver(params, selection=selection) as device:
        device.ui()


@click.group(cls=Group)
@verbose_option
def device_cli(verbose):
    """Direct device actions (screenshot, tap, swipe, etc.)."""


@device_cli.command()
@json_option
@device_options
def screenshot(json_output, **params):
    """Take a screenshot and print the saved file path to stdout."""
    with device_driver(params) as device:
        image = device.screenshot()
        # Core returns a file for cloud screenshots, base64 PNG for local ones.
        if image.startswith(("/", "./")):
            path = image
        else:
            with tempfile.NamedTemporaryFile(prefix="mobilerun_", suffix=".png", delete=False) as file:
                file.write(base64.b64decode(image))
                path = file.name
        emit({"path": path} if json_output else path, json_output=json_output)


@device_cli.command()
@json_option
@device_options
def ui(json_output, **params):
    """Print the UI state returned by the device."""
    with device_driver(params) as device:
        state = json.loads(device.ui_json())
        emit(state if json_output else json.dumps(state, indent=2), json_output=json_output)


@device_cli.command()
@click.argument("x", type=int)
@click.argument("y", type=int)
@device_options
def tap(x, y, **params):
    """Tap at screen coordinates."""
    with device_driver(params) as device:
        device.tap(x, y, humanize=False)
    click.echo(f"Tapped ({x}, {y})", err=True)


@device_cli.command("swipe")
@click.argument("x1", type=int)
@click.argument("y1", type=int)
@click.argument("x2", type=int)
@click.argument("y2", type=int)
@click.option("--duration", type=float, default=1.0, show_default=True, help="Duration in seconds.")
@device_options
def swipe_cmd(x1, y1, x2, y2, duration, **params):
    """Swipe from (x1, y1) to (x2, y2)."""
    with device_driver(params) as device:
        device.swipe(x1, y1, x2, y2, ms=int(duration * 1000), humanize=False)
    click.echo(f"Swiped ({x1}, {y1}) -> ({x2}, {y2})", err=True)


@device_cli.command("long-press")
@click.argument("x", type=int)
@click.argument("y", type=int)
@device_options
def long_press(x, y, **params):
    """Long press at screen coordinates."""
    with device_driver(params) as device:
        device.swipe(x, y, x, y, ms=1000, humanize=False)
    click.echo(f"Long pressed ({x}, {y})", err=True)


@device_cli.command("type")
@click.argument("text")
@click.option("--clear", is_flag=True, help="Clear field before typing.")
@device_options
def type_text(text, clear, **params):
    """Type text into the focused field."""
    with device_driver(params) as device:
        device.type(text, clear=clear, humanize=False)
    click.echo("Text typed.", err=True)


@device_cli.command()
@click.argument("button", type=click.Choice(["back", "home", "enter"], case_sensitive=False))
@device_options
def press(button, **params):
    """Press a system button."""
    with device_driver(params) as device:
        device.key(button)
    click.echo(f"Pressed {button}", err=True)


@device_cli.command()
@click.option("--system", is_flag=True, help="Include system apps.")
@json_option
@device_options
def apps(system, json_output, **params):
    """List installed apps."""
    with device_driver(params) as device:
        result = device.list_apps(include_system_apps=system)
        emit(result if json_output else json.dumps(result, indent=2), json_output=json_output)


@device_cli.command()
@click.argument("package")
@device_options
def start(package, **params):
    """Launch an app by package name."""
    with device_driver(params) as device:
        device.start_app(package)
    click.echo(f"Started {package}", err=True)
