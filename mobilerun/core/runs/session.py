"""Coordinate action observation, screen evidence, and a run's row writer."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar
from uuid import uuid4

from pydantic import JsonValue

from mobilerun.core.extensions.registry import ActionOutcome
from mobilerun.models import Actor, DeviceInfo, RunKind, RunRecord, RunStatus, RunStep
from mobilerun.utils.redaction import _redact_secrets, mask_step_args

from .writer import RunWriter, recording_policy

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device
    from mobilerun.core.runs.capture import FingerprintCapture

R = TypeVar("R")


class RecordingSession:
    def __init__(
        self,
        device: Device,
        capture: FingerprintCapture,
        runs_dir: Path,
        *,
        kind: RunKind,
        source_macro: str | None,
        by: Actor,
        observe_extensions: bool,
    ) -> None:
        self._device = device
        self.capture = capture
        self._by = by
        self._accepting = True
        self._observing = observe_extensions

        def record() -> RunRecord:
            redact = device.known_codes_redactor()
            return RunRecord(
                run_id=uuid4().hex,
                kind=kind,
                source_macro=(
                    redact(_redact_secrets(source_macro))
                    if source_macro is not None
                    else None
                ),
                device=DeviceInfo(
                    platform=device.platform,
                    identifier=redact(_redact_secrets(device.identifier or "unknown")),
                ),
                started_at=datetime.now(UTC),
            )

        self.writer = RunWriter(record, runs_dir)
        if self._observing:
            device._extension_state.observers[device] = self._observe_extension

    def start(self) -> Path | None:
        if (
            self._accepting
            and self._device.recording_allowed()
            and recording_policy()[0]
        ):
            return self.writer.start()
        return None

    def finish(self, status: RunStatus, failed_step: int | None = None) -> None:
        if status == "running":
            raise ValueError("A finished run requires a terminal status")
        if not self._accepting:
            return
        self._accepting = False
        try:
            self.writer.finish(status, failed_step=failed_step)
        finally:
            if self._observing:
                self._device._extension_state.observers.pop(self._device, None)

    def _observe_extension(
        self,
        facade: Device,
        name: str,
        args: dict[str, JsonValue],
        outcome: ActionOutcome,
    ) -> AbstractAsyncContextManager[None]:
        return self._observe(
            "extension_call",
            {"name": name, "arguments": args},
            outcome,
            allowed=facade.recording_allowed(),
        )

    @asynccontextmanager
    async def _observe(
        self,
        action: str,
        args: dict[str, JsonValue],
        outcome: ActionOutcome,
        *,
        allowed: bool = True,
    ) -> AsyncIterator[None]:
        enabled, actor = recording_policy()
        if not allowed or not enabled or not self._device.recording_allowed():
            self.capture.invalidate()
        run_dir = self.start() if allowed else None
        if run_dir is None:
            yield
            return
        before = await self.capture.sample()
        started_at = datetime.now(UTC)
        try:
            yield
        finally:
            finished_at = datetime.now(UTC)
            if self._accepting:
                after = await self.capture.after(
                    cancelled=outcome.error == "CancelledError"
                )
                if self._accepting:
                    try:
                        i = self.writer.step_index + 1
                        screens: dict[
                            Literal["before", "after"],
                            dict[Literal["json", "png"], str],
                        ] = {}
                        if before_files := self.capture.save(
                            run_dir, i, "before", before
                        ):
                            screens["before"] = before_files
                        if after_files := self.capture.save(run_dir, i, "after", after):
                            screens["after"] = after_files
                        self.writer.append(
                            RunStep(
                                i=i,
                                started_at=started_at,
                                finished_at=finished_at,
                                action=action,
                                args=mask_step_args(
                                    args,
                                    contains_code=self._device._extension_state.matches_sensitive_argument,
                                ),
                                ok=outcome.error is None,
                                error=(
                                    _redact_secrets(outcome.error)
                                    if outcome.error is not None
                                    else None
                                ),
                                before=self.capture.fingerprint(before),
                                after=self.capture.fingerprint(after),
                                by=actor or self._by,
                                screens=screens or None,
                            )
                        )
                    except Exception:
                        if outcome.error is None:
                            raise  # Preserve a prior device exception.

    async def dispatch(
        self,
        operation: Callable[[], Awaitable[R]],
        action: str,
        args: dict[str, JsonValue],
    ) -> R:
        outcome = ActionOutcome()
        async with self._observe(action, args, outcome):
            try:
                result = await operation()
            except BaseException as error:
                outcome.error = type(error).__name__
                raise
            if result is False:
                outcome.error = "returned_false"
            return result
