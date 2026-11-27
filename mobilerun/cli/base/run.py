import asyncio
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import click

from mobilerun.errors import CloudAuthRequired

P = ParamSpec("P")
T = TypeVar("T")


def coro(f: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, T]:
    @wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return asyncio.run(f(*args, **kwargs))
        except CloudAuthRequired as exc:
            raise click.ClickException(str(exc)) from exc

    return wrapper
