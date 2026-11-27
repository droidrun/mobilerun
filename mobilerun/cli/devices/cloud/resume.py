from typing import cast

import click

from mobilerun.cli.base import render
from mobilerun.cli.base.billing import billing_option, invoke_with_billing
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.clients.cloud import devices
from mobilerun.clients.cloud.client import cloud_client
from mobilerun.models.billing import Billing


@click.command()
@click.option(
    "--yes", "-y", is_flag=True, help="Confirm metered resume without prompting."
)
@billing_option
@device_command(cloud_only=True)
@coro
async def resume(target: DeviceTarget, yes: bool, billing: str) -> None:
    """Resume a stopped cloud phone."""
    device_id = cast(str, target.identifier)  # @device_command requires a cloud ID.
    mode = Billing(billing)
    async with cloud_client(base_url=target.base_url, billing=mode) as client:
        operation, needs_paid = await devices._prepare_resume(
            device_id, billing=mode, client=client
        )
        await invoke_with_billing(
            operation,
            mode,
            yes=yes,
            confirm_paid="Resume cloud phone" if needs_paid else None,
        )
    render.success(f"Resumed {target}")
