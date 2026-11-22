import click

from mobilerun.cli.base import render
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command("get")
@device_command
@coro
async def language_get(target: DeviceTarget) -> None:
    """Print the device language."""
    async with target.open() as device:
        render.kv([("Language", await device.get_language())])
