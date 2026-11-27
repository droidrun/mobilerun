"""Authentication commands."""

import click

from mobilerun.cli.base.menu import MenuGroup

from .login import login
from .logout import logout
from .status import status


@click.group(cls=MenuGroup)
def auth() -> None:
    """Manage cloud authentication."""


auth.add_command(login)
auth.add_command(logout)
auth.add_command(status, name="status")
