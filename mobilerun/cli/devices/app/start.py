import click

from mobilerun.cli.base import render
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.cli.devices.base.pickers import choose_app


@click.command()
@click.argument("package", required=False)
@device_command
@coro
async def start(
    package: str | None,
    target: DeviceTarget,
) -> None:
    """Launch an app by package name."""
    async with target.open() as device:
        package = await choose_app(device, package)
        await device.start_app(package)
        render.success(f"{target}: Started {package}")
