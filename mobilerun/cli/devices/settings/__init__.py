import click

from mobilerun.cli.base.menu import MenuGroup

from .language import language
from .location import location
from .timezone import timezone


@click.group(cls=MenuGroup)
def settings():
    """Manage location, language and timezone."""


for command in (location, language, timezone):
    settings.add_command(command)
