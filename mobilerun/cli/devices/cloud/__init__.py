import click

from mobilerun.cli.base.menu import MenuGroup

from .reboot import reboot
from .resume import resume
from .status import status
from .stop import stop
from .terminate import terminate


@click.group(cls=MenuGroup)
def cloud():
    """Manage cloud phone lifecycle without connecting to the phone."""


for command in (status, resume, stop, reboot, terminate):
    cloud.add_command(command)
