import click

from mobilerun.cli.base import render
from mobilerun.cli.base.menu import PromptingArgument
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command("set")
@click.argument("latitude", type=float, cls=PromptingArgument)
@click.argument("longitude", type=float, cls=PromptingArgument)
@device_command
@coro
async def location_set(latitude: float, longitude: float, target: DeviceTarget) -> None:
    """Set latitude and longitude."""
    async with target.open() as device:
        await device.set_location(latitude, longitude)
        render.success(f"Location set to {latitude}, {longitude} on {target}")
