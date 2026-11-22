import click

from mobilerun.cli.base import render
from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.cli.devices.base.source import SourceCommand, source_option
from mobilerun.core.workspace import Workspace


@click.command(cls=SourceCommand)
@click.argument("apk", required=False, type=click.Path(exists=True, dir_okay=False))
@click.option("--app-id", help="Install a workspace app by ID.")
@click.option("--reinstall", is_flag=True, help="Replace an existing installation")
@click.option("--grant/--no-grant", default=True, help="Grant runtime permissions")
@source_option("apk", "app_id", apps=True)
@device_command
@coro
async def install(
    apk: str | None,
    app_id: str | None,
    reinstall: bool,
    grant: bool,
    target: DeviceTarget,
) -> None:
    """Install an app on a device."""
    click.echo(f"Installing on {target}…", err=True)
    async with target.open() as device:
        if app_id:
            app = await Workspace.get_app(app_id)
            result = await device.install_app(app)
        else:
            assert apk is not None
            result = await device.install_apk(
                apk, reinstall=reinstall, grant_permissions=grant
            )
        detail = f"{result.package} {result.version}" if result is not None else apk
        render.success(f"{target}: Installed {detail}")
