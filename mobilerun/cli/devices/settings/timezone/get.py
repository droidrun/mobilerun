import click

from mobilerun.cli.base import render
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command("get")
@device_command
@coro
async def timezone_get(target: DeviceTarget) -> None:
    """Print the IANA timezone."""
    async with target.open() as device:
        value = await device.get_timezone()
        render.kv([("Timezone", value if value is not None else "Not available")])
