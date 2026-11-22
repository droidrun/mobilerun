import click

from mobilerun.cli.base.menu import MenuGroup

from .list import list_command
from .pull import pull
from .push import push
from .rm import rm


@click.group(cls=MenuGroup)
def file():
    """Transfer and manage device files."""


for command in (list_command, push, pull, rm):
    file.add_command(command)
