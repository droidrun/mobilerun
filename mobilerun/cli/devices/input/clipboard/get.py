import click

from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command("get")
@device_command
@coro
async def clipboard_get(target: DeviceTarget) -> None:
    """Print clipboard text."""
    async with target.open() as device:
        click.echo(await device.get_clipboard())
