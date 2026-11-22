import click

from mobilerun.cli.base import render
from mobilerun.cli.base.menu import PromptingArgument
from mobilerun.cli.base.run import coro
from mobilerun.cli.base.safety import confirm_action_async
from mobilerun.cli.devices.base.options import DeviceTarget, device_command


@click.command("rm")
@click.argument("remote", cls=PromptingArgument)
@click.option("--yes", "-y", is_flag=True)
@device_command
@coro
async def rm(remote: str, yes: bool, target: DeviceTarget) -> None:
    """Delete a file from a device."""
    await confirm_action_async(
        "delete device file", "no_commerce", f"Delete {remote}.", yes=yes
    )
    async with target.open() as device:
        await device.delete_file(remote)
        render.success(f"Deleted {remote} from {target}")
