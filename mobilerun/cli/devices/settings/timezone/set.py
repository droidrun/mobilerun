import click

from mobilerun.cli.base import render
from mobilerun.cli.base.menu import PromptingArgument
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command("set")
@click.argument("timezone", cls=PromptingArgument)
@device_command
@coro
async def timezone_set(timezone: str, target: DeviceTarget) -> None:
    """Set the IANA timezone."""
    async with target.open() as device:
        await device.set_timezone(timezone)
        render.success(f"Timezone set to {timezone} on {target}")
