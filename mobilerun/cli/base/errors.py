"""Render command failures without leaking unexpected exception details."""

import re
import traceback

import click

from mobilerun.cli.base import render
from mobilerun.errors.base import MobilerunError
from mobilerun.errors.cloud import CloudAuthRequired


def _line(value: object) -> str:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(value))
    return " ".join("".join(c if c.isprintable() else " " for c in text).split())


def safe_error_message(error: BaseException) -> str:
    """Return the same sanitized message shown for an error without rendering it."""
    if isinstance(error, CloudAuthRequired) or isinstance(
        error.__cause__, CloudAuthRequired
    ):
        return "Not logged in. Run `mobilerun account login` or set MOBILERUN_API_KEY."
    if (
        isinstance(error, ModuleNotFoundError)
        and error.name
        and error.name.split(".")[0] in {"anthropic", "langfuse", "openinference"}
    ):
        extra = "anthropic" if error.name.split(".")[0] == "anthropic" else "langfuse"
        return f"Optional dependency missing. Install with: pip install 'mobilerun-agents[{extra}]'"
    if isinstance(error, TimeoutError) and not isinstance(error, MobilerunError):
        return _line(error) or "Timed out"
    if isinstance(error, ExceptionGroup):
        return _line(error.message)
    if isinstance(error, (MobilerunError, click.ClickException)):
        return _line(error)
    return "unexpected failure (use --debug for details)"


def render_error(error: BaseException, debug: bool = False) -> None:
    if debug:
        traceback.print_exception(error)
    elif isinstance(error, CloudAuthRequired) or isinstance(
        error.__cause__, CloudAuthRequired
    ):
        click.echo(f"Error: {safe_error_message(error)}", err=True)
    elif isinstance(error, ExceptionGroup):
        click.echo(safe_error_message(error), err=True)
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
    else:
        click.echo(f"Error: {safe_error_message(error)}", err=True)
