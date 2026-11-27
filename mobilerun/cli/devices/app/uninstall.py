import click

from mobilerun.cli.base import render
from mobilerun.cli.base.run import coro
from mobilerun.cli.base.safety import confirm_action_async
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.cli.devices.base.pickers import choose_app


@click.command()
@click.argument("package", required=False)
@click.option("--yes", "-y", is_flag=True, help="Confirm uninstall without prompting.")
@device_command
@coro
async def uninstall(
    package: str | None,
    yes: bool,
    target: DeviceTarget,
) -> None:
    """Uninstall an app by package name."""
    confirmed = package is not None
    if confirmed:
        await confirm_action_async(
            f"uninstall {package}",
            "no_commerce",
            f"Uninstall {package} from {target}; app data will be deleted.",
            yes=yes,
        )
    async with target.open() as device:
        package = await choose_app(device, package)
        if not confirmed:
            await confirm_action_async(
                f"uninstall {package}",
                "no_commerce",
                f"Uninstall {package} from {target}; app data will be deleted.",
                yes=yes,
            )
        result = await device.uninstall_app(package)
        if isinstance(result, str) and result.lower().startswith(("failure", "error")):
            raise click.ClickException(f"{target}: {result}")
        render.success(f"Uninstalled {package} from {target}")
