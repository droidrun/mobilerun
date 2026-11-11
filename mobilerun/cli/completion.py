"""Print Click's shell completion scripts without installing them."""

import click
from click.shell_completion import get_completion_class

from mobilerun.cli.options import CONTEXT_SETTINGS, Command, emit


@click.command(cls=Command, context_settings=CONTEXT_SETTINGS)
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def completion(shell):
    """Print the completion script for bash, zsh, or fish."""
    root = click.get_current_context().find_root().command
    complete = get_completion_class(shell)(root, {}, "mobilerun", "_MOBILERUN_COMPLETE")
    emit(complete.source())
