import click

from mobilerun.cli.base.menu import MenuGroup
from .location import location
from .language import language
from .timezone import timezone


@click.group(cls=MenuGroup)
def settings():
    """Manage location, language and timezone."""


for command in (location, language, timezone):
    settings.add_command(command)
