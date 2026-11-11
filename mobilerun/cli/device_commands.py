"""Synchronous device actions through mobilerun-core."""

import asyncio
import base64
import json
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager

import click

from mobilerun.agent.utils.android_device import is_portal_http_url
from mobilerun.cli.options import (
    Group,
    _ui_cell,
    device_options,
    emit,
    emit_rows,
    examples,
    json_option,
    verbose_option,
)
from mobilerun.cli.selection import core_client, portal_token, select_device

UI_FLAGS = ("clickable", "checkable", "checked", "editable", "focused", "scrollable", "selected")


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


@click.group(
    cls=Group,
    epilog=examples(
        'mobilerun device list',
        'mobilerun device pair',
        'mobilerun device ui -d DEVICE',
    ),
)
@verbose_option
def device_cli(verbose):
    """Discover, pair, set up, inspect, and control devices."""


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


def _field(node, *keys):
    for key in keys:
        value = node.get(key)
        if value and (not isinstance(value, str) or value.strip()):
            return value
    return ""


def _ltrb(bounds):
    if not isinstance(bounds, Mapping) or not bounds:
        return ""
    if all(key in bounds for key in ("x", "y", "width", "height")):
        values = (
            bounds["x"], bounds["y"],
            bounds["x"] + bounds["width"], bounds["y"] + bounds["height"],
        )
    else:
        values = tuple(bounds.get(key, "") for key in ("left", "top", "right", "bottom"))
    return ",".join(_ui_cell(value) for value in values)


def ui_lines(state, all_elements=False):
    """Render core's depth-first node list as six tab-separated columns."""
    for index, node in enumerate(state.nodes):
        flags = [
            flag for flag in UI_FLAGS
            if _field(node, flag, "is" + flag.capitalize(), "is_" + flag)
        ]
        if node.get("disabled") or any(
            node.get(key) is False for key in ("enabled", "isEnabled", "is_enabled")
        ):
            flags.append("disabled")
        text = _field(node, "text", "contentDescription", "content_description")
        resource = _field(node, "resourceId", "resource_id", "accessibilityIdentifier")
        if not (all_elements or text or resource or flags):
            continue
        class_name = _field(node, "className", "class_name")
        bounds = _field(node, "bounds", "boundsInScreen", "bounds_in_screen", "rect", "frame")
        yield "\t".join((
            str(index), _ui_cell(class_name.rsplit(".", 1)[-1]), _ui_cell(text),
            _ui_cell(resource), _ltrb(bounds), ",".join(flags),
        ))


@device_cli.command(
    epilog=examples(
        'mobilerun device ui -d DEVICE',
        'mobilerun device ui -d DEVICE -j',
    ),
)
@json_option
@click.option("--all", "all_elements", is_flag=True, help="Include empty layout containers.")
@device_options
def ui(json_output, all_elements, **params):
    """Print tab-separated UI elements, or the raw state with --json."""
    if json_output and all_elements:
        raise click.UsageError("--all cannot be combined with --json")
    with device_driver(params) as device:
        if json_output:
            emit(json.loads(device.ui_json()), json_output=True)
            return
        state = device.ui()
        if (getattr(state, "dead_tree", False)
                or getattr(state, "a11y_tree_available", True) is False):
            raise click.ClickException("The device UI tree is unavailable.")
        for line in ui_lines(state, all_elements):
            emit(line)
        phone = state.get("phone_state") or {}
        context = state.get("device_context") or {}
        screen = context.get("screen_bounds") or context.get("display_metrics") or {}
        app = phone.get("currentApp") or phone.get("packageName") or "unknown"
        width = screen.get("width", screen.get("widthPixels", "?"))
        height = screen.get("height", screen.get("heightPixels", "?"))
        click.echo(f"App: {_ui_cell(app)}; screen: {_ui_cell(width)}x{_ui_cell(height)}", err=True)


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
        if json_output:
            emit(result, json_output=True)
        else:
            emit_rows(("APP_ID", "LABEL", "VERSION", "VERSION_CODE", "SYSTEM"), (
                (_field(app, "app_id", "package_name", "bundle_id", "package"),
                 app.get("label", ""), app.get("version_name", ""),
                 app.get("version_code", ""), str(app.get("is_system_app", False)).lower())
                for app in result
            ))


@device_cli.command()
@click.argument("package")
@device_options
def start(package, **params):
    """Launch an app by package name."""
    with device_driver(params) as device:
        device.start_app(package)
    click.echo(f"Started {package}", err=True)
