"""Device command tree."""

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
