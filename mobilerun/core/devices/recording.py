"""Record framework actions without recording expanded credential values."""

from collections.abc import Sequence
from typing import Any, Protocol, TypeVar

from mobilerun.core.devices.base import Device
from mobilerun.core.devices.components.wrapper import DeviceWrapper
from mobilerun.core.extensions.base import _inside_bound_action
from mobilerun.core.operator import Operator
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.options import StateProviderFactory, StateProviderOptions
from mobilerun.models.recording import RecordedAction

D = TypeVar("D", bound=Device)


class ActionRecording(Protocol):
    @property
    def recorded_actions(self) -> Sequence[RecordedAction]: ...


class RecordingDevice(DeviceWrapper[D]):
    def __init__(
        self,
        device: D,
        *,
        state_provider: StateProvider | None = None,
        state_provider_factory: StateProviderFactory | None = None,
        state_options: StateProviderOptions | None = None,
        operator: Operator | None = None,
    ) -> None:
        super().__init__(
            device,
            state_provider=state_provider,
            state_provider_factory=state_provider_factory,
            state_options=state_options,
            operator=operator,
        )
        self._actions: list[RecordedAction] = []
        self._extension_state.observers.append(self._record_extension)

    def _record_extension(self, name: str, arguments: dict[str, Any]) -> None:
        self._actions.append(
            {"action_type": "extension_call", "name": name, "arguments": arguments}
        )

    def _record_primitive(self, action: RecordedAction) -> None:
        if not _inside_bound_action(self):
            self._actions.append(action)

    async def close(self) -> None:
        observers = self._extension_state.observers
        if self._record_extension in observers:
            observers.remove(self._record_extension)
        await self._inner.close()

    @property
    def _recording(self) -> ActionRecording:
        return self

    @property
    def recorded_actions(self) -> Sequence[RecordedAction]:
        return self._actions

    async def tap(self, x: int, y: int) -> None:
        await self._inner.tap(x, y)
        self._record_primitive({"action_type": "tap", "x": x, "y": y})

    async def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000
    ) -> None:
        await self._inner.swipe(x1, y1, x2, y2, duration_ms)
        self._record_primitive(
            {
                "action_type": "swipe",
                "start_x": x1,
                "start_y": y1,
                "end_x": x2,
                "end_y": y2,
                "duration_ms": duration_ms,
            }
        )

    async def drag(
        self, x1: int, y1: int, x2: int, y2: int, duration: float = 3.0
    ) -> None:
        await self._inner.drag(x1, y1, x2, y2, duration)
        self._record_primitive(
            {
                "action_type": "drag",
                "start_x": x1,
                "start_y": y1,
                "end_x": x2,
                "end_y": y2,
                "duration": duration,
            }
        )

    async def _swipe_humanized(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000
    ) -> None:
        await self._inner._swipe_humanized(x1, y1, x2, y2, duration_ms)
        self._record_primitive(
            {
                "action_type": "swipe",
                "start_x": x1,
                "start_y": y1,
                "end_x": x2,
                "end_y": y2,
                "duration_ms": duration_ms,
            }
        )

    async def input_text(
        self, text: str, clear: bool = False, stealth: bool = False, wpm: int = 0
    ) -> bool:
        result = await self._inner.input_text(text, clear, stealth, wpm)
        self._record_primitive(
            {"action_type": "input_text", "text": text, "clear": clear}
        )
        return result

    async def press_button(self, button: str) -> None:
        await self._inner.press_button(button)
        self._record_primitive({"action_type": "button_press", "button": button})

    async def start_app(self, package: str, activity: str | None = None) -> str:
        result = await self._inner.start_app(package, activity)
        self._record_primitive(
            {"action_type": "start_app", "package": package, "activity": activity}
        )
        return result

    async def open_url(self, url: str) -> None:
        await self._inner.open_url(url)
        self._record_primitive({"action_type": "open_url", "url": url})

    async def stop_app(self, package: str) -> str:
        result = await self._inner.stop_app(package)
        self._record_primitive({"action_type": "stop_app", "package": package})
        return result

    async def press_key_code(self, key_code: int) -> None:
        await self._inner.press_key_code(key_code)
        self._record_primitive({"action_type": "press_key_code", "key_code": key_code})

    async def wait(self, seconds: float) -> None:
        await self._inner.wait(seconds)
        self._record_primitive({"action_type": "wait", "duration": seconds})
