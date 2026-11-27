import json

import click

from mobilerun.cli.base import render
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command("get")
@click.option("--json", "as_json", is_flag=True)
@device_command
@coro
async def location_get(as_json: bool, target: DeviceTarget) -> None:
    """Print the device location."""
    async with target.open() as device:
        result = await device.get_location()
        if as_json:
            click.echo(
                json.dumps(
                    None
                    if result is None
                    else {
                        "latitude": result.latitude,
                        "longitude": result.longitude,
                    }
                )
            )
        elif result is None:
            render.kv([("Location", "Not available")])
        else:
            render.kv([("Location", f"{result.latitude:.4f}, {result.longitude:.4f}")])
