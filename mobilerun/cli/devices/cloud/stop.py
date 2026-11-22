import click

from mobilerun.cli.base import render
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.clients.cloud import devices
from mobilerun.clients.cloud.client import cloud_client
from mobilerun.config.settings import get_settings


@click.command()
@device_command(cloud_only=True)
@coro
async def stop(target: DeviceTarget) -> None:
    """Stop a cloud phone without deleting it."""
    billing = get_settings().MOBILERUN_BILLING
    async with cloud_client(base_url=target.base_url, billing=billing) as client:
        await devices.stop(target.identifier, billing=billing, client=client)
    render.success(f"Stopped {target}")
