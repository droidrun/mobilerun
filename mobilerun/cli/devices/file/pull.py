from pathlib import Path

import click

from mobilerun.cli.base.menu import PromptingArgument
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command("pull")
@click.argument("remote", cls=PromptingArgument)
@click.argument("local", required=False, default=".", type=click.Path(path_type=Path))
@device_command
@coro
async def pull(remote: str, local: Path, target: DeviceTarget) -> None:
    """Pull a file from a device."""
    click.echo(f"Pulling from {target}…", err=True)
    async with target.open() as device:
        click.echo(await device.pull_file(remote, local))
