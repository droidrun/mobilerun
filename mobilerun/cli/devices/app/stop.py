import click

from mobilerun.cli.base import render

from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.cli.devices.base.pickers import choose_app


@click.command()
@click.argument("package", required=False)
@device_command
@coro
async def stop(package: str | None, target: DeviceTarget) -> None:
    """Stop an installed app."""
    async with target.open() as device:
        package = await choose_app(device, package)
        await device.stop_app(package)
        render.success(f"Stopped {package} on {target}")
