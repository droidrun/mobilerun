import click

from mobilerun.cli.base.menu import MenuGroup

from .clipboard import clipboard
from .long_press import long_press
from .press import press
from .swipe import swipe_cmd
from .tap import tap
from .type import type_text


@click.group(cls=MenuGroup)
def input():
    """Send input and manage the clipboard."""


for command in (tap, long_press, swipe_cmd, type_text, press, clipboard):
    input.add_command(command)
