"""Explicit consent before purchases and destructive operations."""

import sys
from contextlib import nullcontext, redirect_stdout

import click

from mobilerun.cli.base import menu
from mobilerun.cli.base.prompts import confirm_prompt, confirm_prompt_async


def _prepare_confirmation(
    action: str, billing: str, warning: str, yes: bool, interactive: bool
) -> bool:
    if interactive:
        chargeable = action in {
            "send SMS",
            "reboot cloud phone",
        }
        click.echo(
            f"Billing: {billing}. {warning}"
            if chargeable and billing == "paid"
            else warning
        )
        return not yes
    elif not yes:
        raise click.ClickException(
            f"Refusing to {action} without confirmation; pass --yes."
        )
    return False


def confirm_action(action: str, billing: str, warning: str, *, yes: bool) -> None:
    """Require --yes in automation, or an affirmative interactive confirmation."""
    if _prepare_confirmation(
        action, billing, warning, yes, menu.is_interactive()
    ) and not confirm_prompt("Continue?", default=False):
        click.get_current_context().exit(0)


async def confirm_action_async(
    action: str, billing: str, warning: str, *, yes: bool, err: bool = False
) -> None:
    """Detect the original streams before optionally routing prompts to stderr."""
    interactive = menu.is_interactive()
    with redirect_stdout(sys.stderr) if err else nullcontext():
        if _prepare_confirmation(
            action, billing, warning, yes, interactive
        ) and not await confirm_prompt_async("Continue?", default=False):
            click.get_current_context().exit(0)
