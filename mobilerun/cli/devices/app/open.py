import click

from mobilerun.cli.base import render

from mobilerun.cli.base.menu import PromptingArgument
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command("open")
@click.argument("url", cls=PromptingArgument)
@device_command
@coro
async def open_url(url: str, target: DeviceTarget) -> None:
    """Open a URL or deep link."""
    async with target.open() as device:
        await device.open_url(url)
        render.success(f"Opened {url} on {target}")
