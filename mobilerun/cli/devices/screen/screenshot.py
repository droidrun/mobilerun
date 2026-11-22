import os
import tempfile

import click

from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command()
@device_command
@coro
async def screenshot(
    target: DeviceTarget,
) -> None:
    """Take a screenshot and print the saved file path to stdout."""
    async with target.open() as device:
        png_bytes = await device.screenshot()
        fd, path = tempfile.mkstemp(prefix="mobilerun_", suffix=".png")
        try:
            os.write(fd, png_bytes)
        finally:
            os.close(fd)
        click.echo(path)
