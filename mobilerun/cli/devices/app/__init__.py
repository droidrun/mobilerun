import click

from mobilerun.cli.base.menu import MenuGroup

from .install import install
from .list import apps
from .open import open_url
from .start import start
from .stop import stop
from .uninstall import uninstall


@click.group(cls=MenuGroup)
def app():
    """Manage installed apps and open links."""


for command in (apps, install, uninstall, start, stop, open_url):
    app.add_command(command)
