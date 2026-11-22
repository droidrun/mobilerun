"""Choose a default device, optionally connecting an ADB network address."""

import re
import uuid

import click

from mobilerun.cli.base import render
from async_adbutils import adb

from mobilerun.cli.base.menu import is_interactive
from mobilerun.cli.base.prompts import SelectChoice, select_prompt_async as select_prompt, text_prompt_async as text_prompt
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.default import save_default_device
from mobilerun.cli.devices.base.discovery import discover_devices
from mobilerun.errors.base import MobilerunError


@click.command(short_help="Choose default device")
@click.argument("target", required=False)
@coro
async def connect(target: str | None) -> None:
    """Choose the device Mobilerun uses by default (picker, ID, or host:port for network devices)."""
    if target is None and not is_interactive():
        raise click.ClickException("Pass a device id or host:port.")
    devices = await discover_devices()
    if target is None:
        choices = [
            SelectChoice(
                device.id,
                f"{device.id}"
                + (f" ({device.name})" if device.name else "")
                + (f" [{device.state}]" if device.state else "")
                + f" {device.source}"
                + (" (not ready)" if not device.ready else ""),
            )
            for device in devices
        ]
        choices.append(
            SelectChoice("__add_address__", "Add device by address (host:port)…")
        )
        try:
            target = await select_prompt("Select a device", choices)
            if target is None:
                click.get_current_context().exit(0)
            if target == "__add_address__":
                target = await text_prompt("Device address (host[:port])")
        except (KeyboardInterrupt, click.Abort, EOFError):
            click.get_current_context().exit(130)
    if not any(device.id == target for device in devices):
        try:
            uuid.UUID(target)
        except ValueError:
            is_id = False
        else:
            is_id = True
        if is_id or not re.fullmatch(
            r"(?:[A-Za-z0-9][A-Za-z0-9.-]*|\[[0-9A-Fa-f:]+\])(?::[0-9]{1,5})?", target
        ):
            raise click.ClickException(f"Device {target} is not available.")
        try:
            response = await adb.connect(target)
        except MobilerunError as exc:
            raise click.ClickException(f"Failed to connect to {target}: {exc}") from exc
        prefixes = ("connected to ", "already connected to ")
        prefix = next((item for item in prefixes if response.startswith(item)), None)
        if prefix is None:
            raise click.ClickException(f"Failed to connect to {target}: {response}")
        target = response[len(prefix) :].strip()
    save_default_device(target)
    render.success(f"Connected {target} (default device)")
