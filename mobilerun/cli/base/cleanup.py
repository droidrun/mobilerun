"""Shared batch cleanup confirmation and per-resource reporting."""

import json
from collections.abc import Awaitable, Callable, Sequence

import click

from mobilerun.cli.base import render
from mobilerun.cli.base.errors import render_error, safe_error_message
from mobilerun.cli.base.safety import confirm_action_async


async def cleanup(
    ids: Sequence[str],
    *,
    action: str,
    noun: str,
    billing: str,
    yes: bool,
    as_json: bool,
    operation: Callable[[str], Awaitable[str]],
    warning: str = "",
) -> None:
    """Confirm once, process each target, and report all outcomes safely."""
    ids = tuple(dict.fromkeys(ids))
    count = len(ids)
    label = (
        noun
        if count == 1
        else (
            noun[:-1] + "ies"
            if noun.endswith("y")
            else noun + ("es" if noun.endswith("x") else "s")
        )
    )
    await confirm_action_async(
        f"{action.lower()} {noun}",
        billing,
        f"{action} {count} {label}: {', '.join(ids)}{warning}.",
        yes=yes,
        err=True,
    )
    results = []
    for item_id in ids:
        try:
            item_status = await operation(item_id)
        except (click.Abort, click.exceptions.Exit):
            raise
        except Exception as exc:
            message = safe_error_message(exc)
            results.append({"id": item_id, "status": "failed", "error": message})
            if not as_json:
                click.echo(f"{item_id}: ", nl=False, err=True)
                render_error(exc)
        else:
            results.append({"id": item_id, "status": item_status})
            if not as_json:
                render.success(f"{noun.capitalize()} {item_id}: {item_status}")
    if as_json:
        click.echo(json.dumps(results))
    if any(item["status"] == "failed" for item in results):
        click.get_current_context().exit(1)
