import click

from mobilerun.cli.base import render
from mobilerun.cli.base.menu import PromptingArgument
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command("set")
@click.argument("locale", cls=PromptingArgument)
@device_command
@coro
async def language_set(locale: str, target: DeviceTarget) -> None:
    """Set a locale such as de-DE."""
    async with target.open() as device:
        await device.set_language(locale)
        render.success(f"Language set to {locale} on {target}")
