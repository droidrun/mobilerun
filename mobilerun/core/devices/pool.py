"""Concurrent, ordered operations across a group of devices."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Mapping
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self, TypeVar, Unpack

from pydantic import BaseModel

from mobilerun.core.devices.base import Device, InstallOptions, _unwrap_device
from mobilerun.core.devices.location import Location
from mobilerun.core.extensions import Action, Extension
from mobilerun.core.extensions.actions import Audience
from mobilerun.core.state.ui_state import UIState
from mobilerun.models.apps import App
from mobilerun.models.context import ContextItem, JSONValue
from mobilerun.models.files import DeviceFile
from mobilerun.models.results import ExecutionResult, ExpectationResult

if TYPE_CHECKING:
    from mobilerun.core.mailbox.base import Mailbox
    from mobilerun.core.phone_number.base import PhoneNumber

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")


class _PoolActions:
    def __init__(self, pool: DevicePool) -> None:
        self._pool = pool

    def list(self, *, audience: Audience = "python") -> list[tuple[Action, ...]]:
        return [device.actions.list(audience=audience) for device in self._pool.devices]

    async def call(
        self,
        name: str,
        arguments: Mapping[str, JSONValue],
        *,
        audience: Audience = "python",
        timeout: float | None = None,
    ) -> list[object]:
        schemas = []
        for device in self._pool.devices:
            entry = device.actions.get(name)
            if audience != "python" and not entry.public:
                raise PermissionError(f"Action {name} is restricted")
            entry.validate(arguments)
            schemas.append(entry.input_schema)
        if schemas and any(schema != schemas[0] for schema in schemas):
            raise ValueError(f"Pool action {name} has incompatible schemas")
        return await self._pool._broadcast(
            name,
            lambda device, _: device.actions.call(
                name, arguments, audience=audience, timeout=timeout
            ),
        )


class _PoolState:
    def __init__(self, pool: DevicePool) -> None:
        self._pool = pool

    async def refresh(self) -> list[UIState]:
        return await self._pool._broadcast(
            "state.refresh", lambda device, _: device.state.refresh()
        )


class DevicePool:
    """Broadcast device operations concurrently, collecting results in device order."""

    @property
    def actions(self) -> _PoolActions:
        return _PoolActions(self)

    async def context(self) -> list[tuple[ContextItem, ...]]:
        return await self._broadcast("context", lambda device, _: device.context())

    async def bind_each(
        self,
        factory: Callable[[Device], Extension],
        *,
        force: bool = False,
        close_on_device_close: bool = False,
    ) -> list[Extension]:
        extensions = [factory(device) for device in self.devices]
        if len({id(extension) for extension in extensions}) != len(extensions):
            raise ValueError("Pool binding requires separate extension instances")
        return await self._broadcast(
            "bind",
            lambda device, index: extensions[index].bind(
                device, force=force, close_on_device_close=close_on_device_close
            ),
        )

    async def unbind_extension(self, name: str) -> list[None]:
        return await self._broadcast(
            "unbind_extension", lambda device, _: device.unbind_extension(name)
        )

    # File operations are broadcast in device order.
    async def push_file(
        self, path: str | Path, dest: str | None = None
    ) -> list[DeviceFile]:
        return await self._broadcast(
            "push_file", lambda device, _: device.push_file(path, dest)
        )

    async def list_files(self, directory: str) -> list[list[DeviceFile]]:
        return await self._broadcast(
            "list_files", lambda device, _: device.list_files(directory)
        )

    async def delete_file(self, remote_path: str) -> list[None]:
        return await self._broadcast(
            "delete_file", lambda device, _: device.delete_file(remote_path)
        )

    # Environment controls are broadcast in device order.
    async def get_language(self) -> list[str]:
        return await self._broadcast(
            "get_language", lambda device, _: device.get_language()
        )

    async def get_timezone(self) -> list[str | None]:
        return await self._broadcast(
            "get_timezone", lambda device, _: device.get_timezone()
        )

    async def get_location(self) -> list[Location | None]:
        return await self._broadcast(
            "get_location", lambda device, _: device.get_location()
        )

    async def set_location(self, latitude: float, longitude: float) -> list[None]:
        return await self._broadcast(
            "set_location", lambda device, _: device.set_location(latitude, longitude)
        )

    async def reset_location(self) -> list[None]:
        return await self._broadcast(
            "reset_location", lambda device, _: device.reset_location()
        )

    async def set_timezone(self, timezone: str) -> list[None]:
        return await self._broadcast(
            "set_timezone", lambda device, _: device.set_timezone(timezone)
        )

    async def open_url(self, url: str) -> list[None]:
        return await self._broadcast("open_url", lambda device, _: device.open_url(url))

    @property
    def number(self) -> list[PhoneNumber]:
        return [device.number for device in self.devices]

    @property
    def mailbox(self) -> list[Mailbox]:
        return [device.mailbox for device in self.devices]

    def __init__(
        self, devices: list[Device], *, concurrency: int | None = None
    ) -> None:
        if not devices:
            raise ValueError("DevicePool requires at least one device")
        if concurrency is not None and (
            not isinstance(concurrency, int) or concurrency < 1
        ):
            raise ValueError("concurrency must be a positive integer")
        for device in devices:
            operator = device._operator
            if operator is not None:
                operator.check(_unwrap_device(device))
        self.devices = tuple(devices)
        self.state = _PoolState(self)
        self._semaphore = asyncio.Semaphore(concurrency) if concurrency else None

    def __len__(self) -> int:
        return len(self.devices)

    def __iter__(self) -> Iterator[Device]:
        return iter(self.devices)

    @property
    def platform(self) -> list[str]:
        return [device.platform for device in self.devices]

    @property
    def supported_operations(self) -> list[set[str]]:
        return [device.supported_operations for device in self.devices]

    @property
    def extensions(self) -> list[Mapping[str, Extension]]:
        return [device.extensions for device in self.devices]

    @property
    def supported_buttons(self) -> list[set[str]]:
        return [device.supported_buttons for device in self.devices]

    async def _broadcast(
        self, name: str, call: Callable[[Device, int], Awaitable[R]]
    ) -> list[R]:
        errors: dict[int, Exception] = {}

        async def invoke(device: Device, index: int) -> list[R]:
            try:
                if self._semaphore is None:
                    return [await call(device, index)]
                async with self._semaphore:
                    return [await call(device, index)]
            except Exception as exc:
                exc.add_note(f"device {index}: {type(device).__name__}")
                errors[index] = exc
                return []

        outcomes = await asyncio.gather(
            *(invoke(device, index) for index, device in enumerate(self.devices))
        )
        if errors:
            raise ExceptionGroup(
                f"{len(errors)} of {len(self)} devices failed: {name}",
                [errors[index] for index in sorted(errors)],
            )
        return [value for outcome in outcomes for value in outcome]

    async def __aenter__(self) -> Self:
        try:
            await self.connect()
        except BaseException as exc:
            try:
                await self.close()
            except Exception as cleanup_error:
                exc.add_note(f"Pool cleanup failed: {cleanup_error}")
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            await self.close()
        except Exception as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"Pool cleanup failed: {cleanup_error}")

    async def close(self) -> list[None]:
        return await self._broadcast("close", lambda device, _: device.close())

    async def connect(self) -> list[None]:
        return await self._broadcast("connect", lambda device, _: device.connect())

    async def screenshot(self, hide_overlay: bool = True) -> list[bytes]:
        return await self._broadcast(
            "screenshot", lambda device, _: device.screenshot(hide_overlay=hide_overlay)
        )

    async def tap(self, x: int, y: int) -> list[None]:
        return await self._broadcast("tap", lambda device, _: device.tap(x, y))

    async def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000
    ) -> list[None]:
        return await self._broadcast(
            "swipe",
            lambda device, _: device.swipe(x1, y1, x2, y2, duration_ms=duration_ms),
        )

    async def drag(
        self, x1: int, y1: int, x2: int, y2: int, duration: float = 3.0
    ) -> list[None]:
        return await self._broadcast(
            "drag", lambda device, _: device.drag(x1, y1, x2, y2, duration=duration)
        )

    async def input_text(
        self, text: str, clear: bool = False, stealth: bool = False, wpm: int = 0
    ) -> list[bool]:
        return await self._broadcast(
            "input_text",
            lambda device, _: device.input_text(
                text, clear=clear, stealth=stealth, wpm=wpm
            ),
        )

    async def press_button(self, button: str) -> list[None]:
        return await self._broadcast(
            "press_button", lambda device, _: device.press_button(button)
        )

    async def wait(self, seconds: float) -> list[None]:
        return await self._broadcast("wait", lambda device, _: device.wait(seconds))

    async def press_key_code(self, key_code: int) -> list[None]:
        return await self._broadcast(
            "press_key_code", lambda device, _: device.press_key_code(key_code)
        )

    async def start_app(self, package: str, activity: str | None = None) -> list[str]:
        return await self._broadcast(
            "start_app",
            lambda device, _: device.start_app(package, activity=activity),
        )

    async def stop_app(self, package: str) -> list[str]:
        return await self._broadcast(
            "stop_app", lambda device, _: device.stop_app(package)
        )

    async def install_apk(
        self, path: str | Path, **kwargs: Unpack[InstallOptions]
    ) -> list[App]:
        return await self._broadcast(
            "install_apk", lambda device, _: device.install_apk(path, **kwargs)
        )

    async def install_app(
        self, app: App, **kwargs: Unpack[InstallOptions]
    ) -> list[App]:
        return await self._broadcast(
            "install_app", lambda device, _: device.install_app(app, **kwargs)
        )

    async def uninstall_app(self, package: str) -> list[str]:
        return await self._broadcast(
            "uninstall_app", lambda device, _: device.uninstall_app(package)
        )

    async def list_apps(
        self, include_system: bool = False
    ) -> list[list[dict[str, str]]]:
        return await self._broadcast(
            "list_apps",
            lambda device, _: device.list_apps(include_system=include_system),
        )

    async def tap_element(self, index: int) -> list[None]:
        return await self._broadcast(
            "tap_element", lambda device, _: device.tap_element(index)
        )

    async def set_language(self, locale: str) -> list[None]:
        return await self._broadcast(
            "set_language", lambda device, _: device.set_language(locale)
        )

    async def set_clipboard(self, text: str) -> list[None]:
        return await self._broadcast(
            "set_clipboard", lambda device, _: device.set_clipboard(text)
        )

    async def get_clipboard(self) -> list[str]:
        return await self._broadcast(
            "get_clipboard", lambda device, _: device.get_clipboard()
        )

    async def do(
        self,
        instruction: str,
        *,
        output: type[T] | None = None,
        max_steps: int | None = None,
        operator: Any = None,
        timeout: float | None = None,
    ) -> list[ExecutionResult[T]]:
        kwargs = {"output": output, "max_steps": max_steps, "operator": operator}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return await self._broadcast(
            "do", lambda device, _: device.do(instruction, **kwargs)
        )

    async def expect(
        self, condition: str, *, operator: Any = None, timeout: float | None = None
    ) -> list[ExpectationResult]:
        kwargs = {}
        if operator is not None:
            kwargs["operator"] = operator
        if timeout is not None:
            kwargs["timeout"] = timeout
        return await self._broadcast(
            "expect", lambda device, _: device.expect(condition, **kwargs)
        )
