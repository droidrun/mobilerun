import click

from mobilerun.cli.base.menu import MenuGroup

from .call import call
from .extensions import extensions
from .list import actions


@click.group(cls=MenuGroup)
def action():
    """Inspect and invoke device and extension actions."""


for command in (actions, call, extensions):
    action.add_command(command)
