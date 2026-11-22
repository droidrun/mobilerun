import click

from mobilerun.cli.base import render
from mobilerun.cli.base.run import coro
from mobilerun.cli.base.safety import confirm_action_async
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.clients.cloud import devices
from mobilerun.clients.cloud.client import cloud_client
from mobilerun.config.settings import get_settings


@click.command()
@click.option("--yes", "-y", is_flag=True, help="Confirm permanent phone deletion.")
@device_command(cloud_only=True)
@coro
async def terminate(target: DeviceTarget, yes: bool) -> None:
    """Permanently delete a cloud phone and its data."""
    billing = get_settings().MOBILERUN_BILLING
    await confirm_action_async(
        "terminate cloud phone",
        billing.value,
        f"Permanently delete {target} and its data.",
        yes=yes,
    )
    async with cloud_client(base_url=target.base_url, billing=billing) as client:
        await devices.terminate(target.identifier, billing=billing, client=client)
    render.success(f"Terminated {target}")
