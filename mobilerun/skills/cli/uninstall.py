from pathlib import Path

import click

from mobilerun.cli.base.handling import handle_errors
from mobilerun.skills import uninstall

from .common import options


@click.command("uninstall")
@click.argument("names", nargs=-1, required=True)
@options
@handle_errors
def uninstall_command(
    names: tuple[str, ...], agent: str, project: bool, directory: Path | None
) -> None:
    """Remove installed skills."""
    for result in uninstall(
        list(names), agent, "project" if project else "user", directory
    ):
        click.echo(f"{result.name}: {result.status} ({result.path})")
