"""Device command tree and hidden 0.7 compatibility aliases."""

from copy import copy
from functools import wraps

import click

from mobilerun.cli.base.menu import MenuGroup
from mobilerun.cli.devices.action import action
from mobilerun.cli.devices.app import app
from mobilerun.cli.devices.cloud import cloud
from mobilerun.cli.devices.connect import connect
from mobilerun.cli.devices.disconnect import disconnect
from mobilerun.cli.devices.file import file
from mobilerun.cli.devices.input import input
from mobilerun.cli.devices.list import devices
from mobilerun.cli.devices.ping import ping
from mobilerun.cli.devices.screen import screen
from mobilerun.cli.devices.settings import settings
from mobilerun.cli.devices.setup import setup


@click.group(cls=MenuGroup)
def device() -> None:
    """Manage and control devices."""


for command in (
    devices,
    connect,
    disconnect,
    ping,
    setup,
    app,
    file,
    input,
    screen,
    settings,
    action,
    cloud,
):
    device.add_command(command)


def _legacy_alias(name, group, replacement):
    original = group.commands[replacement]
    alias = copy(original)
    alias.hidden = True
    alias.name = name
    alias.params = [copy(param) for param in original.params]
    for param in alias.params:
        if isinstance(param, click.Option) and param.name == "cloud":
            param.opts = [opt for opt in param.opts if opt != "-c"]
        elif isinstance(param, click.Option) and param.name == "config_path":
            param.opts = [*param.opts, "-c"]
    alias.params.append(
        click.Option(["--device-id", "legacy_device_id"], hidden=True, is_eager=True)
    )

    @wraps(original.callback)
    def invoke(*args, **kwargs):
        click.echo(
            f"Deprecated: device {name}; use device {group.name} {replacement}.",
            err=True,
        )
        legacy_id = kwargs.pop("legacy_device_id", None)
        if legacy_id is not None:
            supplied = kwargs.get("target") or kwargs.get("device_target")
            if supplied is not None and supplied != legacy_id:
                raise click.UsageError(
                    "--device-id and DEVICE/--device refer to different devices."
                )
            kwargs["device_target"] = legacy_id
            kwargs["cloud"] = True
        return original.callback(*args, **kwargs)

    alias.callback = invoke
    device.add_command(alias)


for name, group, replacement in (
    ("screenshot", screen, "screenshot"),
    ("ui", screen, "ui"),
    ("tap", input, "tap"),
    ("swipe", input, "swipe"),
    ("long-press", input, "long-press"),
    ("type", input, "type"),
    ("press", input, "press"),
    ("apps", app, "list"),
    ("start", app, "start"),
):
    _legacy_alias(name, group, replacement)
