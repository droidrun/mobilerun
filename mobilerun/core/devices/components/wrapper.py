"""Explicit framework wrapper delegation and the raw-driver boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Unpack

from mobilerun_core_local.driver.base import DeviceDriver, PlatformUnsupportedError
from pydantic import BaseModel
from mobilerun.core.devices.location import Location

from mobilerun.core.devices.base import Device, InstallOptions
from mobilerun.core.devices.components.identifiers import _sanitize_identifier
from mobilerun.core.devices.components.raw import _translate_unsupported
from mobilerun.core.operator import Operator
from mobilerun.core.proxy.target import ProxyTarget
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.options import StateProviderFactory, StateProviderOptions
from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.models.apps import App
from mobilerun.models.files import DeviceFile
from mobilerun.models.results import ExecutionResult, ExpectationResult

T = TypeVar("T", bound=BaseModel)
D = TypeVar("D", bound=Device)

if TYPE_CHECKING:
    from mobilerun.core.devices.recording import ActionRecording
    from mobilerun.core.mailbox.base import Mailbox
    from mobilerun.core.phone_number.base import PhoneNumber


class DeviceWrapper(Device, Generic[D]):
    """A framework-only wrapper. Unknown attributes are never forwarded."""

    # File operations follow the wrapped device.
    async def push_file(self, path: str | Path, dest: str | None = None) -> DeviceFile:
        return await self._inner.push_file(path, dest)

    async def list_files(self, directory: str) -> list[DeviceFile]:
        return await self._inner.list_files(directory)

    async def pull_file(self, remote_path: str, local_path: str | Path) -> Path:
        return await self._inner.pull_file(remote_path, local_path)

    async def delete_file(self, remote_path: str) -> None:
        await self._inner.delete_file(remote_path)

    # Environment controls follow the wrapped device.
    async def get_location(self) -> Location | None:
        return await self._inner.get_location()

    async def get_language(self) -> str:
        return await self._inner.get_language()

    async def get_timezone(self) -> str | None:
        return await self._inner.get_timezone()

    async def set_location(self, latitude: float, longitude: float) -> None:
        await self._inner.set_location(latitude, longitude)

    async def reset_location(self) -> None:
        await self._inner.reset_location()

    async def set_timezone(self, timezone: str) -> None:
        await self._inner.set_timezone(timezone)

    async def open_url(self, url: str) -> None:
        await self._inner.open_url(url)

    async def set_language(self, locale: str) -> None:
        await self._inner.set_language(locale)

    async def set_clipboard(self, text: str) -> None:
        await self._inner.set_clipboard(text)

    async def get_clipboard(self) -> str:
        return await self._inner.get_clipboard()

    def __init__(
        self,
        device: D,
        *,
        state_provider: StateProvider | None = None,
        state_provider_factory: StateProviderFactory | None = None,
        state_options: StateProviderOptions | None = None,
        operator: Operator | None = None,
    ) -> None:
        self._inner: D = device
        if state_provider is not None and state_provider_factory is not None:
            raise TypeError("Pass state_provider or state_provider_factory, not both")
        if state_provider is None and state_provider_factory is None:
            state_provider = device._state_provider_override
        # Rebuild lazy providers against the OUTER wrapper, not its raw inner IO.
        factory = state_provider_factory or device._state_provider_factory
        options = state_options or device._default_provider_options
        super().__init__(
            state_provider=state_provider,
            state_provider_factory=factory if state_provider is None else None,
            state_options=options,
            operator=operator if operator is not None else device._operator,
        )

    def _unwrap(self) -> Device:
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError("Device wrapper is not initialized")
        return inner._unwrap()

    @property
    def platform(self) -> str:
        return self._inner.platform

    @property
    def supported_operations(self) -> set[str]:
        return self._inner.supported_operations

    @property
    def cloud_id(self) -> str | None:
        return self._inner.cloud_id

    @property
    def supported_buttons(self) -> set[str]:
        return self._inner.supported_buttons

    @property
    def _is_cloud(self) -> bool:
        return self._inner._is_cloud

    @property
    def identifier(self) -> str | None:
        return self._inner.identifier

    async def close(self) -> None:
        await self._inner.close()

    async def connect(self) -> None:
        await self._inner.connect()

    async def _ensure_connected(self) -> None:
        await self._inner._ensure_connected()

    @property
    def number(self) -> PhoneNumber:
        return self._inner.number

    @property
    def mailbox(self) -> Mailbox:
        return self._inner.mailbox

    @property
    def _proxy_target(self) -> ProxyTarget:
        return self._inner._proxy_target

    def _default_operator(self) -> Operator | None:
        return self._inner._default_operator()

    async def do(
        self,
        instruction: str,
        *,
        output: type[T] | None = None,
        max_steps: int | None = None,
        operator: Operator | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult[T]:
        return await Device.do(
            self,
            instruction,
            output=output,
            max_steps=max_steps,
            operator=operator,
            timeout=timeout,
        )

    async def expect(
        self,
        condition: str,
        *,
        operator: Operator | None = None,
        timeout: float | None = None,
    ) -> ExpectationResult:
        return await Device.expect(self, condition, operator=operator, timeout=timeout)

    @property
    def _recording(self) -> ActionRecording | None:
        return self._inner._recording

    @property
    def _state_recovery(self) -> Callable[[], Awaitable[None]] | None:
        return self._inner._state_recovery

    def _create_state_provider(
        self,
        options: StateProviderOptions | None = None,
        *,
        device: Device | None = None,
    ) -> StateProvider:
        return self._inner._create_state_provider(options, device=device or self)

    async def _input_coordinate_size(
        self, screenshot_width: int, screenshot_height: int
    ) -> tuple[int, int]:
        return await self._inner._input_coordinate_size(
            screenshot_width, screenshot_height
        )

    async def input_text(
        self, text: str, clear: bool = False, stealth: bool = False, wpm: int = 0
    ) -> bool:
        return await self._inner.input_text(text, clear, stealth, wpm)

    async def _input_text_raw(
        self, text: str, clear: bool = False, stealth: bool = False, wpm: int = 0
    ) -> bool:
        return await self._inner._input_text_raw(text, clear, stealth, wpm)

    async def wait(self, seconds: float) -> None:
        await self._inner.wait(seconds)

    async def tap(self, x: int, y: int) -> None:
        await self._inner.tap(x, y)

    async def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000
    ) -> None:
        await self._inner.swipe(x1, y1, x2, y2, duration_ms)

    async def drag(
        self, x1: int, y1: int, x2: int, y2: int, duration: float = 3.0
    ) -> None:
        await self._inner.drag(x1, y1, x2, y2, duration)

    async def press_button(self, button: str) -> None:
        await self._inner.press_button(button)

    async def press_key_code(self, key_code: int) -> None:
        await self._inner.press_key_code(key_code)

    async def install_apk(
        self, path: str | Path, **kwargs: Unpack[InstallOptions]
    ) -> App:
        return await self._inner.install_apk(path, **kwargs)

    async def install_app(self, app: App, **kwargs: Unpack[InstallOptions]) -> App:
        return await self._inner.install_app(app, **kwargs)

    async def _swipe_humanized(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000
    ) -> None:
        await self._inner._swipe_humanized(x1, y1, x2, y2, duration_ms)

    async def start_app(self, package: str, activity: str | None = None) -> str:
        return await self._inner.start_app(package, activity)

    async def stop_app(self, package: str) -> str:
        return await self._inner.stop_app(package)

    async def uninstall_app(self, package: str) -> str:
        return await self._inner.uninstall_app(package)

    async def list_apps(self, include_system: bool = False) -> list[dict[str, str]]:
        return await self._inner.list_apps(include_system)

    async def screenshot(self, hide_overlay: bool = True) -> bytes:
        return await self._inner.screenshot(hide_overlay)

    async def _get_ui_tree(self) -> dict[str, Any]:
        return await self._inner._get_ui_tree()

    async def _get_date(self) -> str:
        return await self._inner._get_date()


class DriverDevice(Device):
    """Adapt a raw driver with an explicitly paired provider and lifecycle."""

    def __init__(
        self,
        driver: DeviceDriver,
        *,
        state_provider_factory: StateProviderFactory,
        initialize: Callable[[], Awaitable[None]] | None = None,
        close: Callable[[], Awaitable[None]] | None = None,
        identifier: str | None = None,
    ) -> None:
        self._driver = driver
        self._identifier = _sanitize_identifier(identifier)
        self._initialize_driver = initialize
        self._connected = False
        self._connect_lock = asyncio.Lock()
        self._close_driver = close
        super().__init__(state_provider_factory=state_provider_factory)

    @property
    def platform(self) -> str:
        return self._driver.platform

    @property
    def identifier(self) -> str | None:
        return self._identifier

    @property
    def supported_operations(self) -> set[str]:
        return self._driver.supported

    @property
    def supported_buttons(self) -> set[str]:
        return self._driver.supported_buttons

    @_translate_unsupported
    async def connect(self) -> None:
        async with self._connect_lock:
            if not self._connected:
                if self._initialize_driver is not None:
                    await self._initialize_driver()
                else:
                    await self._driver.connect()
                self._connected = True

    async def close(self) -> None:
        async with self._closing_extensions():
            if self._close_driver is not None:
                await self._close_driver()

    def _create_state_provider(
        self,
        options: StateProviderOptions | None = None,
        *,
        device: Device | None = None,
    ) -> StateProvider:
        assert self._state_provider_factory is not None
        return self._state_provider_factory(
            device or self, options or StateProviderOptions()
        )

    @_translate_unsupported
    async def _ensure_connected(self) -> None:
        await self.connect()

    @_translate_unsupported
    async def _input_text_raw(
        self, text: str, clear: bool = False, stealth: bool = False, wpm: int = 0
    ) -> bool:
        return await self._driver.input_text(
            text, clear=clear, stealth=stealth, wpm=wpm
        )

    @_translate_unsupported
    async def tap(self, x: int, y: int) -> None:
        await self._driver.tap(x, y)

    @_translate_unsupported
    async def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000
    ) -> None:
        await self._driver.swipe(x1, y1, x2, y2, duration_ms)

    @_translate_unsupported
    async def drag(
        self, x1: int, y1: int, x2: int, y2: int, duration: float = 3.0
    ) -> None:
        await self._driver.drag(x1, y1, x2, y2, duration)

    @_translate_unsupported
    async def press_button(self, button: str) -> None:
        await self._driver.press_button(button)

    @_translate_unsupported
    async def press_key_code(self, key_code: int) -> None:
        await self._driver.press_key_code(key_code)

    @_translate_unsupported
    async def install_apk(
        self, path: str | Path, **kwargs: Unpack[InstallOptions]
    ) -> App:
        try:
            await self._driver.install_app(str(path), **kwargs)
        except PlatformUnsupportedError as exc:
            raise DeviceOperationNotSupported(str(exc)) from exc
        from mobilerun.clients.local.apk import inspect_apk

        package, code, version, name = await asyncio.to_thread(inspect_apk, Path(path))
        return App(package, version, name, version_code=code, path=Path(path))

    async def install_app(self, app: App, **kwargs: Unpack[InstallOptions]) -> App:
        if app.path is None:
            raise ValueError("This app has no local APK; use install_apk with a path")
        await self.install_apk(app.path, **kwargs)
        return app

    @_translate_unsupported
    async def start_app(self, package: str, activity: str | None = None) -> str:
        return await self._driver.start_app(package, activity)

    @_translate_unsupported
    async def stop_app(self, package: str) -> str:
        return await self._driver.stop_app(package)

    @_translate_unsupported
    async def uninstall_app(self, package: str) -> str:
        try:
            return await self._driver.uninstall_app(package)
        except PlatformUnsupportedError as exc:
            raise DeviceOperationNotSupported(str(exc)) from exc

    @_translate_unsupported
    async def list_apps(self, include_system: bool = False) -> list[dict[str, str]]:
        return await self._driver.get_apps(include_system)

    @_translate_unsupported
    async def screenshot(self, hide_overlay: bool = True) -> bytes:
        return await self._driver.screenshot(hide_overlay)

    @_translate_unsupported
    async def _input_coordinate_size(
        self, screenshot_width: int, screenshot_height: int
    ) -> tuple[int, int]:
        return await self._driver.input_coordinate_size(
            screenshot_width, screenshot_height
        )

    @_translate_unsupported
    async def _get_ui_tree(self) -> dict[str, Any]:
        return await self._driver.get_ui_tree()

    @_translate_unsupported
    async def _get_date(self) -> str:
        return await self._driver.get_date()
