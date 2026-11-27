import json

import click

from mobilerun.cli.base import render
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command("list")
@click.option("--system/--no-system", default=False, help="Include system apps")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@device_command
@coro
async def apps(
    system: bool,
    as_json: bool,
    target: DeviceTarget,
) -> None:
    """List installed apps."""
    async with target.open() as device:
        app_list = await device.list_apps(include_system=system)
        if as_json:
            click.echo(json.dumps(app_list, ensure_ascii=False))
            return
        if not app_list:
            render.empty("apps")
            return
        render.heading(len(app_list), "app")
        rows = []
        for app in app_list:
            label = app.get("label") or ""
            package = (
                app.get("package")
                or app.get("package_name")
                or app.get("packageName")
                or ""
            )
            rows.append(
                [
                    label or package,
                    render.secondary(package if label and label != package else ""),
                ]
            )
        render.table(rows)
