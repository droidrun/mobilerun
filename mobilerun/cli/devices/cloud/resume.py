import click

from mobilerun.cli.base import render
from mobilerun.cli.base.billing import creation_warning
from mobilerun.cli.base.run import coro
from mobilerun.cli.base.safety import confirm_action_async
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.clients.cloud import devices
from mobilerun.clients.cloud.client import cloud_client
from mobilerun.config.settings import get_settings


@click.command()
@click.option("--yes", "-y", is_flag=True, help="Confirm cloud phone charges.")
@device_command(cloud_only=True)
@coro
async def resume(target: DeviceTarget, yes: bool) -> None:
    """Resume a stopped cloud phone."""
    billing = get_settings().MOBILERUN_BILLING
    await confirm_action_async(
        "resume cloud phone", billing.value, creation_warning(billing.value), yes=yes
    )
    async with cloud_client(base_url=target.base_url, billing=billing) as client:
        await devices.resume(target.identifier, billing=billing, client=client)
    render.success(f"Resumed {target}")
