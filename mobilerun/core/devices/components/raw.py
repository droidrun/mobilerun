"""Explicit raw I/O delegation for framework-owned backend compositions.

The backend is chosen at construction, never inferred from an instance.
Framework input semantics stay in Device; raw text always bypasses expansion.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from functools import wraps
from types import CoroutineType
from typing import Any, Concatenate, ParamSpec, TypeVar

from mobilerun_core_local.driver.base import DeviceDriver, PlatformUnsupportedError

from mobilerun.errors import DeviceOperationNotSupported

P = ParamSpec("P")
R = TypeVar("R")
B = TypeVar("B", bound="_BackendIO")


def _translate_unsupported(
    operation: Callable[P, Coroutine[Any, Any, R]],
) -> Callable[P, CoroutineType[Any, Any, R]]:
    """Translate only the raw driver's declared unsupported-operation errors."""

    @wraps(operation)
    async def invoke(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await operation(*args, **kwargs)
        except (NotImplementedError, PlatformUnsupportedError) as exc:
            raise DeviceOperationNotSupported(
                str(exc) or f"{operation.__name__} is unsupported"
            ) from exc

    return invoke


def _backend_operation(
    operation: Callable[Concatenate[B, P], Coroutine[Any, Any, R]],
) -> Callable[Concatenate[B, P], CoroutineType[Any, Any, R]]:
    translated = _translate_unsupported(operation)

    @wraps(operation)
    async def invoke(self: B, *args: P.args, **kwargs: P.kwargs) -> R:
        await self._ensure_connected()
        return await translated(self, *args, **kwargs)

    return invoke


class _BackendIO:
    _backend: type[DeviceDriver]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._backend_connect_lock = asyncio.Lock()
        super().__init__(*args, **kwargs)

    @property
    def _io_driver(self) -> DeviceDriver:
        raise DeviceOperationNotSupported("No raw backend is configured")

    @_translate_unsupported
    async def connect(self) -> None:
        await self._ensure_connected()

    @_translate_unsupported
    async def _ensure_connected(self) -> None:
        async with self._backend_connect_lock:
            await self._io_driver.ensure_connected()

    @_backend_operation
    async def tap(self, x: int, y: int) -> None:
        await self._io_driver.tap(x, y)

    @_backend_operation
    async def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000
    ) -> None:
        await self._io_driver.swipe(x1, y1, x2, y2, duration_ms)

    @_backend_operation
    async def _input_text_raw(
        self, text: str, clear: bool = False, stealth: bool = False, wpm: int = 0
    ) -> bool:
        return await self._io_driver.input_text(text, clear, stealth, wpm)

    @_backend_operation
    async def press_button(self, button: str) -> None:
        await self._io_driver.press_button(button)

    @_backend_operation
    async def press_key_code(self, key_code: int) -> None:
        await self._io_driver.press_key_code(key_code)

    @_backend_operation
    async def drag(
        self, x1: int, y1: int, x2: int, y2: int, duration: float = 3.0
    ) -> None:
        await self._io_driver.drag(x1, y1, x2, y2, duration)

    @_backend_operation
    async def start_app(self, package: str, activity: str | None = None) -> str:
        return await self._io_driver.start_app(package, activity)

    @_backend_operation
    async def stop_app(self, package: str) -> str:
        return await self._io_driver.stop_app(package)

    @_backend_operation
    async def uninstall_app(self, package: str) -> str:
        return await self._io_driver.uninstall_app(package)

    @_backend_operation
    async def list_apps(self, include_system: bool = False) -> list[dict[str, str]]:
        return await self._io_driver.get_apps(include_system)

    @_backend_operation
    async def screenshot(self, hide_overlay: bool = True) -> bytes:
        return await self._io_driver.screenshot(hide_overlay)

    @_backend_operation
    async def _input_coordinate_size(
        self, screenshot_width: int, screenshot_height: int
    ) -> tuple[int, int]:
        return await self._io_driver.input_coordinate_size(
            screenshot_width, screenshot_height
        )

    @_backend_operation
    async def _get_ui_tree(self) -> dict[str, Any]:
        return await self._io_driver.get_ui_tree()

    @_backend_operation
    async def _get_date(self) -> str:
        return await self._io_driver.get_date()
