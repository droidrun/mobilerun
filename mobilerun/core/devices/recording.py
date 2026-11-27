"""Explicit device wrapper for recording dispatched actions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, TypeVar

from pydantic import JsonValue

from mobilerun.config.settings import get_settings
from mobilerun.core.devices.base import Device
from mobilerun.core.devices.components.wrapper import DeviceWrapper
from mobilerun.core.extensions.base import _inside_bound_action
from mobilerun.core.operator import Operator
from mobilerun.core.runs import recording_policy
from mobilerun.core.runs.capture import FingerprintCapture
from mobilerun.core.runs.session import RecordingSession
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.options import StateProviderFactory, StateProviderOptions
from mobilerun.models import Actor, Fingerprint, RunKind, RunStatus
from mobilerun.utils.redaction import _active_secrets, _redact_secrets

D = TypeVar("D", bound=Device)
R = TypeVar("R")


class RecordingDevice(DeviceWrapper[D]):
    """Record one outcome for each dispatched primitive or outer extension call."""

    def __init__(
        self,
        device: D,
        *,
        kind: RunKind = "recording",
        source_macro: str | None = None,
        by: Actor = "script",
        runs_dir: Path | None = None,
        capture: Literal["full", "actions"] = "full",
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
        settings = get_settings()
        self._owner = device.recording_device() or self
        evidence = FingerprintCapture(
            self,
            device,
            capture == "full",
            settings.MOBILERUN_RUN_SETTLE_SECONDS,
        )
        self._session = RecordingSession(
            self,
            evidence,
            runs_dir if runs_dir is not None else settings.runs_dir,
            kind=kind,
            source_macro=source_macro,
            by=by,
            observe_extensions=self._owner is self,
        )

    def recording_device(self) -> RecordingDevice[Any]:
        return self._owner

    @property
    def run_dir(self) -> Path | None:
        return self._owner._session.writer.started_dir

    @property
    def step_index(self) -> int:
        return self._owner._session.writer.step_index

    def collect_steps(self) -> AbstractContextManager[list[int]]:
        """Collect committed rows from this scope and tasks spawned in it."""
        return self._owner._session.writer.collect_steps()

    def start_run(self) -> Path | None:
        return self._owner._session.start() if self.recording_allowed() else None

    def finish(self, status: RunStatus, failed_step: int | None = None) -> None:
        self._session.finish(status, failed_step)

    async def close(self) -> None:
        try:
            self.finish("cancelled")
        except BaseException as error:
            try:
                await self._inner.close()
            except BaseException:
                error.add_note("The wrapped device also failed to close")
            raise
        await self._inner.close()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            await self.close()
        except Exception:
            if exc is None:
                raise
            exc.add_note("Recording/device cleanup also failed")

    async def capture_fingerprint(self, *, fresh: bool = False) -> Fingerprint | None:
        if not self.recording_allowed() or not recording_policy()[0]:
            return None
        return await self._owner._session.capture.capture(fresh=fresh)

    async def _dispatch(
        self, operation: Callable[[], Awaitable[R]], action: str, **args: JsonValue
    ) -> R:
        if self._owner is not self or _inside_bound_action(self):
            return await operation()
        return await self._session.dispatch(operation, action, args)

    async def tap(self, x: int, y: int) -> None:
        await self._dispatch(partial(self._inner.tap, x, y), "tap", x=x, y=y)

    async def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000
    ) -> None:
        op = partial(self._inner.swipe, x1, y1, x2, y2, duration_ms)
        await self._gesture(op, "swipe", (x1, y1, x2, y2), duration_ms=duration_ms)

    async def _swipe_humanized(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000
    ) -> None:
        op = partial(self._inner._swipe_humanized, x1, y1, x2, y2, duration_ms)
        await self._gesture(op, "swipe", (x1, y1, x2, y2), duration_ms=duration_ms)

    async def drag(
        self, x1: int, y1: int, x2: int, y2: int, duration: float = 3.0
    ) -> None:
        op = partial(self._inner.drag, x1, y1, x2, y2, duration)
        await self._gesture(op, "drag", (x1, y1, x2, y2), duration=duration)

    async def _gesture(
        self,
        operation: Callable[[], Awaitable[None]],
        action: str,
        points: tuple[int, int, int, int],
        **duration: JsonValue,
    ) -> None:
        args = dict(zip(("start_x", "start_y", "end_x", "end_y"), points, strict=True))
        await self._dispatch(operation, action, **args, **duration)

    async def input_text(
        self, text: str, clear: bool = False, stealth: bool = False, wpm: int = 0
    ) -> bool:
        sensitive = bool(
            _active_secrets.get()
        ) or self._extension_state.matches_sensitive_argument(text)
        if sensitive:
            self._extension_state.sensitive_ui = True
            if _redact_secrets(text) != text:
                self._extension_state.known_codes.observed_secrets.add(text)
        op = partial(self._inner.input_text, text, clear, stealth, wpm)
        return await self._text(op, "***" if sensitive else text, clear, stealth, wpm)

    async def _input_text_raw(
        self, text: str, clear: bool = False, stealth: bool = False, wpm: int = 0
    ) -> bool:
        if not _active_secrets.get():
            return await super()._input_text_raw(text, clear, stealth, wpm)
        op = partial(self._inner._input_text_raw, text, clear, stealth, wpm)
        return await self._text(op, "***", clear, stealth, wpm)

    async def _text(
        self,
        operation: Callable[[], Awaitable[bool]],
        text: str,
        clear: bool,
        stealth: bool,
        wpm: int,
    ) -> bool:
        return await self._dispatch(
            operation, "input_text", text=text, clear=clear, stealth=stealth, wpm=wpm
        )

    async def press_button(self, button: str) -> None:
        await self._dispatch(
            partial(self._inner.press_button, button), "button_press", button=button
        )

    async def start_app(self, package: str, activity: str | None = None) -> str:
        return await self._dispatch(
            partial(self._inner.start_app, package, activity),
            "start_app",
            package=package,
            activity=activity,
        )

    async def open_url(self, url: str) -> None:
        await self._dispatch(partial(self._inner.open_url, url), "open_url", url=url)

    async def stop_app(self, package: str) -> str:
        return await self._dispatch(
            partial(self._inner.stop_app, package), "stop_app", package=package
        )

    async def press_key_code(self, key_code: int) -> None:
        await self._dispatch(
            partial(self._inner.press_key_code, key_code),
            "press_key_code",
            key_code=key_code,
        )

    async def wait(self, seconds: float) -> None:
        await self._dispatch(
            partial(self._inner.wait, seconds), "wait", duration=seconds
        )
