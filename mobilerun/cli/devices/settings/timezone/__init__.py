import click

from mobilerun.cli.base.menu import MenuGroup

from .get import timezone_get
from .set import timezone_set


@click.group(cls=MenuGroup)
def timezone() -> None:
    """Read or set the device timezone."""


timezone.add_command(timezone_get)
timezone.add_command(timezone_set)
