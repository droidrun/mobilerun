import click

from mobilerun.cli.base.menu import MenuGroup

from .screenshot import screenshot
from .ui import ui


@click.group(cls=MenuGroup)
def screen():
    """Capture screenshots and inspect the UI."""


screen.add_command(screenshot)
screen.add_command(ui)
