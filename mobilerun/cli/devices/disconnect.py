"""Clear the default device or disconnect a specific ADB TCP device."""

import re

import click
from async_adbutils import adb

from mobilerun.cli.base import render
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.default import save_default_device
from mobilerun.config.loader import ConfigLoader
from mobilerun.errors.base import MobilerunError


@click.command()
@click.argument("target", required=False)
@coro
async def disconnect(target: str | None) -> None:
    """Clear the default, or disconnect a specific network device."""
    default = ConfigLoader.load(None).device.serial
    if target is None:
        if default is None:
            click.echo("No default device.")
            return
        target = default
    is_tcp = (
        re.fullmatch(
            r"(?:[A-Za-z0-9][A-Za-z0-9.-]*|\[[0-9A-Fa-f:]+\]):[0-9]{1,5}", target
        )
        is not None
    )
    if target != default and not is_tcp:
        raise click.ClickException(f"{target} is not a TCP device address.")
    if is_tcp:
        try:
            response = await adb.disconnect(target, raise_error=True)
        except MobilerunError as exc:
            raise click.ClickException(f"Failed to disconnect {target}: {exc}") from exc
        if not response or not response.startswith("disconnected "):
            raise click.ClickException(f"Failed to disconnect {target}: {response}")
    if target == default:
        save_default_device(None)
    render.success(f"Disconnected {target}")
