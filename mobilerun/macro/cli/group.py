"""Macro commands."""

import click

from mobilerun.cli.base.menu import MenuGroup
from mobilerun.macro.cli.list import list as list_command
from mobilerun.macro.cli.replay import replay


@click.group(cls=MenuGroup)
def macro() -> None:
    """Replay recorded automation sequences."""


macro.add_command(list_command)
macro.add_command(replay)
