"""Sandbox commands."""

import click

from mobilerun.cli.base.menu import MenuGroup
from mobilerun.utils.paths import local_store_path

from . import commands


@click.group(cls=MenuGroup)
def sandbox() -> None:
    """Deploy and run cloud functions."""


@sandbox.command("path")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
def path(as_json: bool) -> None:
    """Show the local sandbox directory."""
    from mobilerun.sandbox.settings import get_sandbox_settings

    settings = get_sandbox_settings()
    info = local_store_path(
        settings.sandbox_directory,
        overridden=settings.MOBILERUN_SANDBOX_DIRECTORY is not None,
    )
    click.echo(info.format(as_json))


for command in (
    commands.deploy_command,
    commands.list_command,
    commands.fire_command,
    commands.runs_command,
    commands.enable_command,
    commands.disable_command,
):
    sandbox.add_command(command)
