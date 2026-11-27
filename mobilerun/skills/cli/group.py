"""Commands for installing bundled coding-agent skills."""

import click

from mobilerun.cli.base.menu import MenuGroup

from .install import install_command
from .list import list_command
from .show import show_command
from .uninstall import uninstall_command


@click.group(cls=MenuGroup)
def skills() -> None:
    """Manage skills for coding agents."""


skills.add_command(list_command)
skills.add_command(install_command)
skills.add_command(uninstall_command)
skills.add_command(show_command)
