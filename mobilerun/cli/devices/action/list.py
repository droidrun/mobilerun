import json

import click

from mobilerun.cli.base import render
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command("list")
@click.option("--json", "json_output", is_flag=True, help="Output JSON.")
@device_command
@coro
async def actions(json_output: bool, target: DeviceTarget) -> None:
    """List callable device and extension actions."""
    async with target.open() as device:
        entries = device.actions.list(audience="cli")
        if json_output:
            click.echo(
                json.dumps(
                    [
                        {
                            "name": entry.name,
                            "description": entry.description,
                            "input_schema": entry.input_schema,
                        }
                        for entry in entries
                    ],
                    ensure_ascii=False,
                )
            )
        else:
            if not entries:
                render.empty("actions")
            else:
                render.heading(len(entries), "action")
                render.table(
                    [
                        [entry.name, render.secondary(entry.description)]
                        for entry in entries
                    ]
                )
