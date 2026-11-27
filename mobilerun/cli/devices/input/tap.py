import click

from mobilerun.cli.base import render
from mobilerun.cli.base.menu import PromptingArgument
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command()
@click.argument("x", type=int, cls=PromptingArgument, prompt_label="X coordinate")
@click.argument("y", type=int, cls=PromptingArgument, prompt_label="Y coordinate")
@device_command
@coro
async def tap(
    x: int,
    y: int,
    target: DeviceTarget,
) -> None:
    """Tap at screen coordinates."""
    async with target.open() as device:
        await device.tap(x, y)
        render.success(f"{target}: Tapped ({x}, {y})")
