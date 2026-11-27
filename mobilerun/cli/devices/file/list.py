import json
from dataclasses import asdict

import click

from mobilerun.cli.base import render
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command("list")
@click.argument("directory", required=False, default="/sdcard/Download")
@click.option("--json", "as_json", is_flag=True)
@device_command
@coro
async def list_command(directory: str, as_json: bool, target: DeviceTarget) -> None:
    """List files on a device."""
    async with target.open() as device:
        entries = await device.list_files(directory)
        if as_json:
            click.echo(json.dumps([asdict(entry) for entry in entries], default=str))
        else:
            if not entries:
                render.empty("files")
            else:
                render.heading(len(entries), "file")
                render.table([[entry.path] for entry in entries])
