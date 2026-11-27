import json
from dataclasses import asdict

import click

from mobilerun.cli.base import render
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command("extensions")
@click.option("--json", "json_output", is_flag=True, help="Output JSON.")
@device_command
@coro
async def extensions(json_output: bool, target: DeviceTarget) -> None:
    """List bound extensions and their context."""
    async with target.open() as device:
        entries = []
        for name, extension in device.extensions.items():
            entries.append(
                {
                    "name": name,
                    "kind": type(extension).__name__,
                    "exclusive": extension.exclusive,
                    "context": [asdict(item) for item in await extension.context()],
                }
            )
        if json_output:
            click.echo(json.dumps(entries, ensure_ascii=False))
        else:
            if not entries:
                render.empty("extensions")
                return
            render.heading(len(entries), "extension")
            for entry in entries:
                render.table(
                    [
                        [
                            entry["name"],
                            render.secondary(entry["kind"]),
                            render.secondary(
                                f"exclusive: {entry['exclusive']}"
                                if entry["exclusive"]
                                else ""
                            ),
                        ]
                    ]
                )
                render.kv(
                    (f"  {item['kind']}", json.dumps(item["data"], ensure_ascii=False))
                    for item in entry["context"]
                )
