"""Common functionality for framework device implementations."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Self,
    SupportsIndex,
    TypedDict,
    TypeVar,
    Unpack,
    cast,
)

from pydantic import BaseModel
from mobilerun.core.devices.location import Location

from mobilerun.core.devices.components.identifiers import _redact_secrets, _secret_input
from mobilerun.core.devices.components.input_lock import (
    InputOperationLock,
    serialize_inputs,
)
from mobilerun.core.extensions import Action, ActionSet, Extension
from mobilerun.core.extensions.actions import Declaration
from mobilerun.core.extensions.registry import Registry
from mobilerun.core.operator import Operator
from mobilerun.core.proxy.target import ProxyTarget
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.options import StateProviderFactory, StateProviderOptions
from mobilerun.core.state.view import UIStateView
from mobilerun.errors import (
    DeviceOperationNotSupported,
    ExpectationFailed,
    ExpectationInconclusive,
    NoFocusedInput,
)
from mobilerun.models.apps import App
from mobilerun.models.context import ContextItem
from mobilerun.models.files import DeviceFile
from mobilerun.models.results import (
    ExecutionResult,
    ExpectationResult,
    ExpectationVerdict,
)
from mobilerun.config.prompts import PromptLoader, load_prompt_text

T = TypeVar("T", bound=BaseModel)


class InstallOptions(TypedDict, total=False):
    timeout: float
    reinstall: bool
    grant_permissions: bool


if TYPE_CHECKING:
    from mobilerun.clients.cloud.client import AsyncMobilerun
    from mobilerun.core.devices.recording import ActionRecording
    from mobilerun.core.mailbox.base import Mailbox
    from mobilerun.core.phone_number.base import PhoneNumber


def _unwrap_device(device: Device) -> Device:
    """Find the innermost framework device, not its underlying driver."""
    return device._unwrap()


@serialize_inputs
class Device:
    """Shared UI-state access for framework devices."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        serialize_inputs(cls)

    def _number_cloud_client(self) -> AsyncMobilerun | None:
        """Cloud client for device-scoped resource operations, when available."""
        return None

    @property
    def _extension_state(self) -> Registry:
        return self._unwrap().__dict__["_extensions"]

    @property
    def extensions(self) -> Mapping[str, Extension]:
        return self._extension_state.view

    def __getattr__(self, name: str) -> Extension:
        root = self._unwrap()
        registry = root.__dict__.get("_extensions")
        if registry is not None and name in registry.view:
            return registry.view[name]
        if name == "proxy" and self.cloud_id is not None:
            raise AttributeError(
                "Cloud phone has no active proxy; await proxy.bind(phone) to configure one without automatic provisioning"
            )
        raise AttributeError(
            f"{type(self).__name__} has no extension or attribute '{name}'"
        )

    def __copy__(self) -> Self:
        raise TypeError(
            "Devices cannot be copied or pickled; retain the original handle"
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        raise TypeError(
            "Devices cannot be copied or pickled; retain the original handle"
        )

    def __reduce_ex__(self, protocol: SupportsIndex) -> Any:
        raise TypeError(
            "Devices cannot be copied or pickled; retain the original handle"
        )

    async def unbind_extension(self, name: str) -> None:
        extension = self.extensions.get(name)
        if extension is None:
            raise LookupError(f"No extension named '{name}'")
        await extension.unbind()

    @property
    def actions(self) -> ActionSet:
        return ActionSet(self)

    def _action_entries(self) -> tuple[Action, ...]:
        operations = {
            "tap": self.tap,
            "swipe": self.swipe,
            "drag": self.drag,
            "input_text": self.input_text,
            "press_button": self.press_button,
            "press_key_code": self.press_key_code,
            "start_app": self.start_app,
            "stop_app": self.stop_app,
            "open_url": self.open_url,
            "wait": self.wait,
        }
        entries = [
            Action.compile(
                "device",
                name,
                method,
                Declaration(),
                description=f"Device {name.replace('_', ' ')}",
            )
            for name, method in operations.items()
            if name in self.supported_operations or name == "wait"
        ]
        for extension in self.extensions.values():
            entries.extend(extension._action_entries())
        return tuple(entries)

    async def context(self) -> tuple[ContextItem, ...]:
        items = []
        for extension in tuple(self.extensions.values()):
            items.extend(await extension.context())
        return tuple(items)

    @property
    def cloud_id(self) -> str | None:
        return None

    async def _close_extensions(self) -> None:
        registry = self._extension_state
        async with registry.lock:
            registry.closing = True
            bindings = tuple(registry.bindings.values())
        errors = []
        for binding in reversed(bindings):
            try:
                if binding.owned:
                    await binding.extension.close()
                else:
                    await binding.extension.unbind()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("Could not close extensions", errors)

    @asynccontextmanager
    async def _closing_extensions(self) -> AsyncIterator[None]:
        error = None
        try:
            await self._close_extensions()
        except BaseException as exc:
            error = exc
        try:
            yield
        finally:
            if error is not None:
                raise error

    # File operations default to unsupported on backends without file access.
    async def push_file(self, path: str | Path, dest: str | None = None) -> DeviceFile:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support push_file"
        )

    async def list_files(self, directory: str) -> list[DeviceFile]:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support list_files"
        )

    async def pull_file(self, remote_path: str, local_path: str | Path) -> Path:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support pull_file"
        )

    async def delete_file(self, remote_path: str) -> None:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support delete_file"
        )

    # Device environment controls; unsupported backends fail explicitly.
    async def get_location(self) -> Location | None:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support get_location"
        )

    async def set_location(self, latitude: float, longitude: float) -> None:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support set_location"
        )

    async def reset_location(self) -> None:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support reset_location"
        )

    async def set_timezone(self, timezone: str) -> None:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support set_timezone"
        )

    async def get_timezone(self) -> str | None:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support get_timezone"
        )

    async def get_language(self) -> str:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support get_language"
        )

    async def open_url(self, url: str) -> None:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support open_url"
        )

    async def set_language(self, locale: str) -> None:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support set_language"
        )

    async def set_clipboard(self, text: str) -> None:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support set_clipboard"
        )

    async def get_clipboard(self) -> str:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support get_clipboard"
        )

    @property
    def _is_cloud(self) -> bool:
        return False

    @property
    def identifier(self) -> str | None:
        """Stable log metadata; never a repr containing connection secrets."""
        return None

    async def install_apk(
        self, path: str | Path, **kwargs: Unpack[InstallOptions]
    ) -> App:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support install_apk"
        )

    async def install_app(self, app: App, **kwargs: Unpack[InstallOptions]) -> App:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support install_app"
        )

    @property
    def _proxy_target(self) -> ProxyTarget:
        return ProxyTarget()

    async def _swipe_humanized(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000
    ) -> None:
        await self.swipe(x1, y1, x2, y2, duration_ms)

    def _unwrap(self) -> Device:
        return self

    @property
    def _recording(self) -> ActionRecording | None:
        return None

    @property
    def _state_recovery(self) -> Callable[[], Awaitable[None]] | None:
        return None

    async def _input_coordinate_size(
        self, screenshot_width: int, screenshot_height: int
    ) -> tuple[int, int]:
        return screenshot_width, screenshot_height

    def __await__(self) -> Generator[Any, None, Self]:
        async def ready() -> Self:
            await self.connect()
            return self

        return ready().__await__()

    async def close(self) -> None:
        """Release resources owned by this device, if any."""
        await self._close_extensions()

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    def __init__(
        self,
        *args: Any,
        state_provider: StateProvider | None = None,
        state_provider_factory: StateProviderFactory | None = None,
        state_options: StateProviderOptions | None = None,
        operator: Operator | None = None,
        **kwargs: Any,
    ) -> None:
        if "credentials" in kwargs:
            raise TypeError("Use await provider.bind(device), not credentials=...")
        super().__init__(*args, **kwargs)
        self._extensions = Registry()
        self._input_operations = InputOperationLock()
        self._operator: Operator | None = operator
        if operator is not None:
            operator.check(self)
        if state_provider is not None and state_provider_factory is not None:
            raise TypeError("Pass state_provider or state_provider_factory, not both")
        self._state_provider_factory = state_provider_factory
        self._default_provider_options = state_options or StateProviderOptions()
        self._ui_state_view = UIStateView(self, self._build_default_state_provider)
        self._state_provider_override: StateProvider | None = None
        if state_provider is not None:
            self._set_state_provider(state_provider)

    def _set_state_provider(self, state_provider: StateProvider | None) -> None:
        """Attach a provider; passing None restores the lazy built-in default."""
        self._state_provider_override = state_provider
        self._ui_state_view._attach(state_provider)

    def _set_state_provider_factory(
        self, factory: Callable[[Device], StateProvider]
    ) -> None:
        self._ui_state_view._attach_factory(factory)

    async def input_text(
        self, text: str, clear: bool = False, stealth: bool = False, wpm: int = 0
    ) -> bool:
        return await self._input_text_raw(text, clear, stealth, wpm)

    async def _input_text_raw(
        self, text: str, clear: bool = False, stealth: bool = False, wpm: int = 0
    ) -> bool:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support input_text"
        )

    async def _input_secret(self, value: str) -> bool:
        with _secret_input(value):
            try:
                phone_state = (await self.state.refresh()).phone_state
                if (
                    not isinstance(phone_state, dict)
                    or "focusedElement" not in phone_state
                    or not isinstance(phone_state.get("isEditable"), bool)
                ):
                    raise DeviceOperationNotSupported(
                        "Secret input requires a state provider that reports focused editable fields"
                    )
                if (
                    not isinstance(phone_state.get("focusedElement"), dict)
                    or not phone_state["focusedElement"]
                    or not phone_state["isEditable"]
                ):
                    raise NoFocusedInput(
                        "No text field is focused; tap the field before typing a credential"
                    )
                return await self._input_text_raw(value)
            except Exception as exc:
                message = _redact_secrets(str(exc))
                try:
                    sanitized = type(exc)(message)
                except Exception:
                    sanitized = RuntimeError(message)
            raise sanitized from None

    async def wait(self, seconds: float) -> None:
        """Pause; unlike a plain sleep, wrappers can record this operation."""
        await asyncio.sleep(seconds)

    @property
    def number(self) -> PhoneNumber:
        return cast("PhoneNumber", self._resource_lookup("sms", "phone_number"))

    @property
    def mailbox(self) -> Mailbox:
        return cast("Mailbox", self._resource_lookup("email", "mailbox"))

    def _resource_lookup(self, name: str, role: str) -> Extension:
        default = self.extensions.get(name)
        if default is not None and default.role == role:
            return default
        matches = [item for item in self.extensions.values() if item.role == role]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise DeviceOperationNotSupported(f"No {name} extension is bound")
        raise DeviceOperationNotSupported(
            f"Multiple {name} extensions are bound; select one by name"
        )

    def _default_operator(self) -> Operator | None:
        return None

    async def do(
        self,
        instruction: str,
        *,
        output: type[T] | None = None,
        max_steps: int | None = None,
        operator: Operator | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult[T]:
        selected = operator if operator is not None else self._operator
        if selected is None:
            selected = self._default_operator()
        if selected is None:
            selected = Operator()
        selected.check(self)
        kwargs: dict[str, Any] = {"output": output, "max_steps": max_steps}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return await selected.execute(self, instruction, **kwargs)

    async def expect(
        self,
        condition: str,
        *,
        operator: Operator | None = None,
        timeout: float | None = None,
    ) -> ExpectationResult:
        kwargs: dict[str, Any] = {}
        if operator is not None:
            operator.check(self)
            kwargs["operator"] = operator
        if timeout is not None:
            kwargs["timeout"] = timeout
        instruction = PromptLoader.render_template(
            load_prompt_text("devices/expect.jinja2"), {"condition": condition}
        )
        task = await self.do(instruction, output=ExpectationVerdict, **kwargs)
        if not task.success or task.output is None:
            raise ExpectationInconclusive(condition, task)
        result = ExpectationResult(
            passed=task.output.passed,
            condition=condition,
            reason=task.output.reason,
            task=task,
        )
        if not result.passed:
            raise ExpectationFailed(result)
        return result

    def _create_state_provider(
        self,
        options: StateProviderOptions | None = None,
        *,
        device: Device | None = None,
    ) -> StateProvider:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support _create_state_provider"
        )

    def _build_default_state_provider(self) -> StateProvider:
        if self._state_provider_factory is not None:
            return self._state_provider_factory(self, self._default_provider_options)
        return self._create_state_provider(self._default_provider_options)

    def _reset_state_provider(
        self, options: StateProviderOptions | None = None
    ) -> None:
        """Set options for a fresh lazy default provider, replacing any override."""
        self._default_provider_options = options or StateProviderOptions()
        self._ui_state_view._reset_default()

    @property
    def state(self) -> UIStateView:
        return self._ui_state_view

    async def tap_element(self, index: int) -> None:
        self.state._require("element_index")
        x, y = self.state.current.get_element_coords(index)
        return await self.tap(x, y)

    # Concrete backends implement the raw I/O surface.
    @property
    def platform(self) -> str:
        return ""

    @property
    def supported_operations(self) -> set[str]:
        return set()

    @property
    def supported_buttons(self) -> set[str]:
        return set()

    async def connect(self) -> None:
        await self._ensure_connected()

    async def _ensure_connected(self) -> None:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support _ensure_connected"
        )

    async def screenshot(self, hide_overlay: bool = True) -> bytes:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support screenshot"
        )

    async def tap(self, x: int, y: int) -> None:
        raise DeviceOperationNotSupported(f"{type(self).__name__} doesn't support tap")

    async def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000
    ) -> None:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support swipe"
        )

    async def start_app(self, package: str, activity: str | None = None) -> str:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support start_app"
        )

    async def stop_app(self, package: str) -> str:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support stop_app"
        )

    async def uninstall_app(self, package: str) -> str:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support uninstall_app"
        )

    async def list_apps(self, include_system: bool = False) -> list[dict[str, str]]:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support list_apps"
        )

    async def press_button(self, button: str) -> None:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support press_button"
        )

    async def press_key_code(self, key_code: int) -> None:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support press_key_code"
        )

    async def drag(
        self, x1: int, y1: int, x2: int, y2: int, duration: float = 3.0
    ) -> None:
        raise DeviceOperationNotSupported(f"{type(self).__name__} doesn't support drag")

    async def _get_ui_tree(self) -> dict[str, Any]:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support _get_ui_tree"
        )

    async def _get_date(self) -> str:
        raise DeviceOperationNotSupported(
            f"{type(self).__name__} doesn't support _get_date"
        )
