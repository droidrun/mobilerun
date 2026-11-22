import click

from mobilerun.cli.base.menu import MenuGroup
from .list import actions
from .call import call
from .extensions import extensions


@click.group(cls=MenuGroup)
def action():
    """Inspect and invoke device and extension actions."""


for command in (actions, call, extensions):
    action.add_command(command)
