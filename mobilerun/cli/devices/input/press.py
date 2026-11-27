import click

from mobilerun.cli.base import render
from mobilerun.cli.base.menu import ChoiceArgument
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command()
@click.argument(
    "button",
    cls=ChoiceArgument,
    type=click.Choice(["back", "home", "enter"], case_sensitive=False),
)
@device_command
@coro
async def press(
    button: str,
    target: DeviceTarget,
) -> None:
    """Press a system button."""
    async with target.open() as device:
        await device.press_button(button)
        render.success(f"{target}: Pressed {button}")
