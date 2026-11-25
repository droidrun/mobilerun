import click

from mobilerun.cli.base import render
from mobilerun.cli.base.menu import PromptingArgument
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command("set")
@click.argument("text", cls=PromptingArgument)
@device_command
@coro
async def clipboard_set(text: str, target: DeviceTarget) -> None:
    """Set clipboard text on a cloud device."""
    async with target.open() as device:
        await device.set_clipboard(text)
        render.success(f"Clipboard set on {target}")
