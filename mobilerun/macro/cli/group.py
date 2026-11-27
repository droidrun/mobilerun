"""Macro commands."""

import click

from mobilerun.cli.base.menu import MenuGroup
from mobilerun.macro.cli.list import list as list_command
from mobilerun.macro.cli.replay import replay
from mobilerun.macro.settings import get_macro_settings
from mobilerun.utils.paths import local_store_path


@click.group(cls=MenuGroup)
def macro() -> None:
    """Replay recorded automation sequences."""


@macro.command("path")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
def path(as_json: bool) -> None:
    """Show the local macro directory."""
    settings = get_macro_settings()
    info = local_store_path(
        settings.macro_directory,
        overridden=settings.MOBILERUN_MACRO_DIRECTORY is not None,
    )
    click.echo(info.format(as_json))


macro.add_command(list_command)
macro.add_command(replay)
