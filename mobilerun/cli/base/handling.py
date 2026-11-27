"""Translate expected command errors into concise Click errors."""

import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import click

P = ParamSpec("P")
T = TypeVar("T")


def handle_errors(function: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return function(*args, **kwargs)
        except (ValueError, OSError) as exc:
            raise click.ClickException(str(exc)) from exc

    return wrapped
