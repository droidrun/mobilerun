import click

from mobilerun.cli.base import render
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command("type")
@click.argument("text")
@click.option("--clear", is_flag=True, default=False, help="Clear field before typing")
@device_command
@coro
async def type_text(
    text: str,
    clear: bool,
    target: DeviceTarget,
) -> None:
    """Type text into the currently focused field. Use 'tap' first to focus."""
    async with target.open() as device:
        success = await device.input_text(text, clear)
        if success:
            render.success(f"{target}: Typed text")
        else:
            raise click.ClickException("Failed to type text")
