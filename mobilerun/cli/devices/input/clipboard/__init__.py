import click

from mobilerun.cli.base.menu import MenuGroup

from .get import clipboard_get
from .set import clipboard_set


@click.group(cls=MenuGroup, context_settings={"help_option_names": ["-h", "--help"]})
def clipboard() -> None:
    """Read or set the device clipboard."""


clipboard.add_command(clipboard_get)
clipboard.add_command(clipboard_set)
