"""Cloud device logout command."""

import click

from mobilerun.cli.base import render
from mobilerun.clients.local import login_store as credentials
from mobilerun.config.env import get_secrets, secret_value


@click.command()
def logout() -> None:
    """Remove the saved Mobilerun cloud login."""
    if credentials.clear_credentials():
        render.success("Logged out.")
    else:
        render.warn("Not logged in.")
    if secret_value(get_secrets().MOBILERUN_API_KEY):
        render.warn("MOBILERUN_API_KEY is still set and overrides saved credentials.")
