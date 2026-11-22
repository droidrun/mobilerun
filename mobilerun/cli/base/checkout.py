"""CLI-owned checkout presentation."""

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

import click

T = TypeVar("T")


async def complete_checkout(purchase: Awaitable[T]) -> T:
    try:
        return await purchase
    except TimeoutError as exc:
        for note in getattr(exc, "__notes__", ()):
            click.echo(note, err=True)
        raise click.ClickException(str(exc)) from exc
    except (asyncio.CancelledError, KeyboardInterrupt) as exc:
        for note in getattr(exc, "__notes__", ()):
            click.echo(note, err=True)
        raise
