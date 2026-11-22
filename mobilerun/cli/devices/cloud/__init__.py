import click

from mobilerun.cli.base.menu import MenuGroup
from .status import status
from .resume import resume
from .stop import stop
from .reboot import reboot
from .terminate import terminate


@click.group(cls=MenuGroup)
def cloud():
    """Manage cloud phone lifecycle without connecting to the phone."""


for command in (status, resume, stop, reboot, terminate):
    cloud.add_command(command)
