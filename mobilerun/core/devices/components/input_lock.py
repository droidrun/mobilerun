"""Task-reentrant root-device serialization for focus-changing operations."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device

D = TypeVar("D", bound="Device")
P = ParamSpec("P")
R = TypeVar("R")

_INPUT_OPERATIONS = frozenset(
    {
        "tap",
        "swipe",
        "drag",
        "press_button",
        "press_key_code",
        "input_text",
        "start_app",
        "stop_app",
        "open_url",
        "tap_element",
        "_input_secret",
        "_input_text_raw",
        "_swipe_humanized",
        "install_app",
        "install_apk",
        "uninstall_app",
        "set_language",
    }
)


class InputOperationLock:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[Any] | None = None

    @asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        task = asyncio.current_task()
        if task is not None and self._owner is task:
            yield
            return
        async with self._lock:
            self._owner = task
            try:
                yield
            finally:
                self._owner = None


def _serialize_input(
    fn: Callable[Concatenate[D, P], Awaitable[R]],
) -> Callable[Concatenate[D, P], Awaitable[R]]:
    @wraps(fn)
    async def invoke(self: D, *args: P.args, **kwargs: P.kwargs) -> R:
        async with self._unwrap()._input_operations.hold():
            return await fn(self, *args, **kwargs)

    invoke.__dict__["_serialized_input"] = True
    return invoke


def serialize_inputs(cls: type[D]) -> type[D]:
    """Instrument base methods, overrides and inherited device mixins alike."""
    for name in _INPUT_OPERATIONS:
        method = getattr(cls, name, None)
        if inspect.iscoroutinefunction(method) and not getattr(
            method, "_serialized_input", False
        ):
            setattr(cls, name, _serialize_input(method))
    return cls
