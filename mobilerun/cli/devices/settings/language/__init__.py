import click

from mobilerun.cli.base.menu import MenuGroup

from .get import language_get
from .set import language_set


@click.group(cls=MenuGroup)
def language() -> None:
    """Read or set the device language."""


language.add_command(language_get)
language.add_command(language_set)
