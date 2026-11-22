import click

from mobilerun.cli.base.menu import MenuGroup
from .tap import tap
from .long_press import long_press
from .swipe import swipe_cmd
from .type import type_text
from .press import press
from .clipboard import clipboard


@click.group(cls=MenuGroup)
def input():
    """Send input and manage the clipboard."""


for command in (tap, long_press, swipe_cmd, type_text, press, clipboard):
    input.add_command(command)
