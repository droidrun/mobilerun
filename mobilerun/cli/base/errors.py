"""Render command failures without leaking unexpected exception details."""

import re
import traceback

import click

from mobilerun.addons.agents import is_missing_agents
from mobilerun.cli.base import render
from mobilerun.errors.base import MobilerunError
from mobilerun.errors.cloud import CloudAuthRequired


def _line(value: object) -> str:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(value))
    return " ".join("".join(c if c.isprintable() else " " for c in text).split())


def render_error(error: BaseException, debug: bool = False) -> None:
    if debug:
        traceback.print_exception(error)
    elif isinstance(error, CloudAuthRequired) or isinstance(
        error.__cause__, CloudAuthRequired
    ):
        click.echo(
            "Error: Not logged in. Run `mobilerun account login` or set MOBILERUN_API_KEY.",
            err=True,
        )
    elif is_missing_agents(error):
        click.echo(
            "Error: Agent commands need the mobilerun-agents add-on: pip install mobilerun-agents",
            err=True,
        )
    elif (
        isinstance(error, ModuleNotFoundError)
        and error.name
        and error.name.split(".")[0] in {"anthropic", "langfuse", "openinference"}
    ):
        extra = "anthropic" if error.name.split(".")[0] == "anthropic" else "langfuse"
        click.echo(
            f"Error: Optional dependency missing. Install with: pip install 'mobilerun-agents[{extra}]'",
            err=True,
        )
    elif isinstance(error, TimeoutError) and not isinstance(error, MobilerunError):
        click.echo(f"Error: {_line(error) or 'Timed out'}", err=True)
    elif isinstance(error, ExceptionGroup):
        click.echo(_line(error.message), err=True)
        for cause in error.exceptions:
            # Exception notes exist only after add_note has been called.
            notes = getattr(cause, "__notes__", [])
            label = f"{_line(notes[0])}: " if notes else ""
            message = (
                _line(cause)
                if isinstance(cause, MobilerunError)
                else "Unexpected error (use --debug for details)"
            )
            render.error(f"{label}{message}")
    elif isinstance(error, (MobilerunError, click.ClickException)):
        click.echo(f"Error: {_line(error)}", err=True)
    else:
        click.echo("Error: unexpected failure (use --debug for details)", err=True)
