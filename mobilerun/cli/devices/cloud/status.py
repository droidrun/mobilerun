import click

from mobilerun.cli.base import render
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.clients.cloud import devices
from mobilerun.clients.cloud.client import cloud_client
from mobilerun.config.settings import get_settings


@click.command()
@click.option("--json", "as_json", is_flag=True)
@device_command(cloud_only=True)
@coro
async def status(target: DeviceTarget, as_json: bool) -> None:
    """Print cloud phone status."""
    billing = get_settings().MOBILERUN_BILLING
    async with cloud_client(base_url=target.base_url, billing=billing) as client:
        info = await devices.retrieve(target.identifier, billing=billing, client=client)
    if as_json:
        click.echo(info.model_dump_json())
    else:
        name = getattr(info, "name", None)
        render.table(
            [
                [
                    name or str(target),
                    render.status(info.state),
                    render.secondary(str(target)) if name else "",
                ]
            ]
        )
