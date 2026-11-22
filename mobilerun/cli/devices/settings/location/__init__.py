import click

from mobilerun.cli.base.menu import MenuGroup
from .get import location_get
from .set import location_set
from .reset import location_reset


@click.group(cls=MenuGroup)
def location() -> None:
    """Read, set or reset the device location."""


for command in (location_get, location_set, location_reset):
    location.add_command(command)
