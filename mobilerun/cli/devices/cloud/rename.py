"""Rename an existing cloud phone without opening a device connection."""

import json
from typing import Any

import click

from mobilerun.cli.base import render
from mobilerun.cli.base.menu import is_interactive
from mobilerun.cli.base.prompts import text_prompt_async
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.clients.cloud import devices
from mobilerun.clients.cloud.client import cloud_client
from mobilerun.config.settings import get_settings


class _RenameCommand(click.Command):
    def invoke(self, ctx: click.Context) -> Any:
        """With -d, bind a single positional to NAME, not DEVICE."""
        params = ctx.params
        if (
            params["device_target"]
            and params["target"] is not None
            and params["name"] is None
        ):
            params["name"], params["target"] = params["target"], None
        return super().invoke(ctx)


@click.command(cls=_RenameCommand)
@click.option("--json", "as_json", is_flag=True)
@device_command(cloud_only=True)
@click.argument("name", required=False)
@coro
async def rename(target: DeviceTarget, name: str | None, as_json: bool) -> None:
    """Rename a cloud phone."""
    if name is None:
        if not is_interactive():
            raise click.UsageError("Missing argument 'NAME'.")
        name = await text_prompt_async("Name")
        if not name or not name.strip():
            click.echo("Cancelled.", err=True)
            return
    name = name.strip()
    if not name:
        raise click.UsageError("NAME must not be empty.")
    billing = get_settings().MOBILERUN_BILLING
    async with cloud_client(base_url=target.base_url, billing=billing) as client:
        await devices.set_name(target.identifier, name, billing=billing, client=client)
    if as_json:
        click.echo(json.dumps({"id": target.identifier, "name": name}))
    else:
        render.success(f"Renamed {target} to {name}")
