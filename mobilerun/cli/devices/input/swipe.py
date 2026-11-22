import click

from mobilerun.cli.base import render
from mobilerun.cli.base.menu import PromptingArgument
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command("swipe")
@click.argument(
    "x1", type=int, cls=PromptingArgument, prompt_label="X coordinate (start)"
)
@click.argument(
    "y1", type=int, cls=PromptingArgument, prompt_label="Y coordinate (start)"
)
@click.argument(
    "x2", type=int, cls=PromptingArgument, prompt_label="X coordinate (end)"
)
@click.argument(
    "y2", type=int, cls=PromptingArgument, prompt_label="Y coordinate (end)"
)
@click.option(
    "--duration", type=float, default=1.0, show_default=True, help="Duration in seconds"
)
@device_command
@coro
async def swipe_cmd(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    duration: float,
    target: DeviceTarget,
) -> None:
    """Swipe from (x1, y1) to (x2, y2)."""
    async with target.open() as device:
        await device.swipe(x1, y1, x2, y2, duration_ms=int(duration * 1000))
        render.success(f"{target}: Swiped ({x1}, {y1}) -> ({x2}, {y2})")
