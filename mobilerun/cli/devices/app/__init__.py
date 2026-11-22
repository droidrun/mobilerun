import click

from mobilerun.cli.base.menu import MenuGroup
from .list import apps
from .install import install
from .uninstall import uninstall
from .start import start
from .stop import stop
from .open import open_url


@click.group(cls=MenuGroup)
def app():
    """Manage installed apps and open links."""


for command in (apps, install, uninstall, start, stop, open_url):
    app.add_command(command)
