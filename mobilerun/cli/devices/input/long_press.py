import click

from mobilerun.cli.base import render
from mobilerun.cli.base.menu import PromptingArgument
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command("long-press")
@click.argument("x", type=int, cls=PromptingArgument, prompt_label="X coordinate")
@click.argument("y", type=int, cls=PromptingArgument, prompt_label="Y coordinate")
@device_command
@coro
async def long_press(
    x: int,
    y: int,
    target: DeviceTarget,
) -> None:
    """Long press at screen coordinates."""
    if target.kind == "ios":
        raise click.ClickException("long-press is not supported on iOS")
    async with target.open() as device:
        await device.swipe(x, y, x, y, 1000)
        render.success(f"{target}: Long pressed ({x}, {y})")
