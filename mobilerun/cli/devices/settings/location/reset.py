import click

from mobilerun.cli.base import render

from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command("reset")
@device_command
@coro
async def location_reset(target: DeviceTarget) -> None:
    """Reset the device location."""
    async with target.open() as device:
        await device.reset_location()
        render.success(f"Location reset on {target}")
