import click

from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command("get")
@device_command
@coro
async def clipboard_get(target: DeviceTarget) -> None:
    """Print clipboard text from a cloud device."""
    if target.kind != "cloud":
        raise click.UsageError("Clipboard commands require a cloud device ID.")
    async with target.open() as device:
        if device is None:
            raise click.UsageError("Missing argument 'DEVICE'.")
        click.echo(await device.get_clipboard())
