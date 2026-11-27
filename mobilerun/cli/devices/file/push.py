import click

from mobilerun.cli.base.run import coro
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.cli.devices.base.source import SourceCommand, source_option
from mobilerun.core.workspace import Workspace


@click.command("push", cls=SourceCommand)
@click.argument("path", required=False, type=click.Path(exists=True, dir_okay=False))
@click.option("--dest")
@click.option("--file-id", help="Push a workspace file by ID.")
@source_option("path", "file_id", apps=False)
@device_command
@coro
async def push(
    path: str | None, file_id: str | None, dest: str | None, target: DeviceTarget
) -> None:
    """Push a local or workspace file to a device."""
    item = None
    if file_id:
        items = await Workspace.list_files()
        item = next((item for item in items if item.id == file_id), None)
        if item is None:
            raise click.ClickException("Workspace file not found")
    click.echo(f"Pushing to {target}…", err=True)
    async with target.open() as device:
        if file_id:
            result = await Workspace.push_file(item, device, dest)
        else:
            result = await device.push_file(path, dest)
        click.echo(result.path)
