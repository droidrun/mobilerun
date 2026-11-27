"""Deterministic, guarded replay on a caller-owned device."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from mobilerun.core.credentials import CredentialProvider
from mobilerun.core.credentials.masking import CredentialMasks, masking_scope
from mobilerun.core.devices import Device
from mobilerun.core.state.base import StateProvider
from mobilerun.errors.base import DeviceError
from mobilerun.macro.errors import MacroError, MacroMismatch, MismatchPhase
from mobilerun.macro.models import MacroStep
from mobilerun.macro.settings import get_macro_settings
from mobilerun.macro.utils import (
    capture_fingerprint,
    compare_states,
    has_masked_arguments,
)
from mobilerun.models import Fingerprint

logger = logging.getLogger("mobilerun-macro")

_OBSERVATION_ERRORS = (DeviceError, OSError, NotImplementedError)
_Credential = tuple[CredentialProvider, str | None, str, str | None]


@dataclass
class _ReplayMasks:
    masks: CredentialMasks
    prepared: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class MacroPlayer:
    """Replay steps without constructing, recording or closing the device."""

    def __init__(
        self,
        device: Device | None = None,
        delay_between_actions: float | None = None,
        state_timeout: float | None = None,
        state_threshold: float | None = None,
        state_poll_interval: float | None = None,
    ) -> None:
        settings = get_macro_settings()
        self.driver = device
        self.delay_between_actions = (
            settings.MOBILERUN_MACRO_DELAY
            if delay_between_actions is None
            else delay_between_actions
        )
        self.state_timeout = (
            settings.MOBILERUN_MACRO_STATE_TIMEOUT
            if state_timeout is None
            else state_timeout
        )
        self.state_threshold = (
            settings.MOBILERUN_MACRO_STATE_THRESHOLD
            if state_threshold is None
            else state_threshold
        )
        self.state_poll_interval = (
            settings.MOBILERUN_MACRO_POLL_INTERVAL
            if state_poll_interval is None
            else state_poll_interval
        )
        self.state_provider: StateProvider | None = None
        self.executed_steps = 0
        self._tainted = False
        self._replay_masks: ContextVar[_ReplayMasks | None] = ContextVar(
            "macro_replay_masks", default=None
        )

    @property
    def tainted(self) -> bool:
        """Whether this player may have dispatched sensitive input; sticky."""
        return self._tainted

    @contextmanager
    def replay_scope(self) -> Iterator[None]:
        """Share credential masks across sequential play_step calls, then clear them."""
        with self._mask_scope():
            yield

    @contextmanager
    def _mask_scope(self) -> Iterator[_ReplayMasks]:
        existing = self._replay_masks.get()
        if existing is not None:
            yield existing
            return
        with masking_scope() as masks:
            state = _ReplayMasks(masks)
            token = self._replay_masks.set(state)
            try:
                yield state
            finally:
                self._replay_masks.reset(token)

    def _device(self) -> Device:
        if self.driver is None:
            raise MacroError("MacroPlayer requires a Device")
        return self.driver

    async def get_current_fingerprint(self) -> Fingerprint | None:
        """Read the best available structural screen evidence, without UI text."""
        driver = self._device()
        if self.state_provider is not None:
            try:
                observed = capture_fingerprint(await self.state_provider.get_state())
            except _OBSERVATION_ERRORS:
                pass
            else:
                if observed is not None:
                    return observed
        view = driver.state
        if view is not None:
            try:
                observed = capture_fingerprint(await view.refresh())
            except _OBSERVATION_ERRORS:
                pass
            else:
                if observed is not None:
                    return observed
        try:
            return capture_fingerprint(await driver._get_ui_tree())
        except _OBSERVATION_ERRORS:
            return None

    async def _wait_for_guard(
        self, expected: Fingerprint
    ) -> tuple[bool, Fingerprint | None]:
        deadline = time.monotonic() + self.state_timeout
        while True:
            observed = await self.get_current_fingerprint()
            if compare_states(expected, observed, self.state_threshold):
                return True, observed
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False, observed
            # Engine polling is not an authored device wait.
            await asyncio.sleep(min(remaining, self.state_poll_interval))

    def _preflight(self, step: MacroStep) -> _Credential | None:
        """Validate credential and redaction constraints before device observation."""
        args: dict[str, Any] = step.args
        if has_masked_arguments(args):
            raise MacroError("Cannot replay redacted action arguments")
        driver = self._device()
        if step.action not in {"extension_call", "type_secret"}:
            return None
        if args.get("protected", False) is not False:
            raise MacroError("Protected credential replay requires host approval")
        if step.action == "extension_call":
            qualified = args.get("name")
            arguments = args.get("arguments")
            if not isinstance(qualified, str) or not isinstance(arguments, dict):
                raise MacroError("Invalid extension_call action")
            owner, dot, method = qualified.partition(".")
            if not owner or not dot or not method:
                raise MacroError("Invalid extension_call action")
            bound = driver.extensions.get(owner)
            if bound is None:
                raise MacroError("Extension is not bound for macro replay")
            entry = driver.actions.get(qualified)
            if entry.sensitive_parameters & arguments.keys():
                raise MacroError("Cannot replay sensitive extension arguments")
            if method != "type_field" or not isinstance(bound, CredentialProvider):
                return None
            vault = bound
        else:
            arguments = args
            default_vault = driver.extensions.get("vault")
            if isinstance(default_vault, CredentialProvider):
                vault = default_vault
            else:
                candidates = [
                    v
                    for v in driver.extensions.values()
                    if isinstance(v, CredentialProvider)
                ]
                if len(candidates) != 1:
                    raise MacroError("Bind one vault to replay a secret action")
                vault = candidates[0]
        package, name = arguments.get("package"), arguments.get("name")
        field = arguments.get("field", "password")
        if not any(isinstance(v, str) and v.strip() for v in (package, name)):
            raise MacroError("Credential action requires a package or name")
        if (
            (package is not None and not isinstance(package, str))
            or (name is not None and not isinstance(name, str))
            or not isinstance(field, str)
            or not field.strip()
        ):
            raise MacroError("Invalid credential reference")
        if vault.requires_approval(package, field, name):
            raise MacroError("Protected credential replay requires host approval")
        return vault, package, field, name

    async def _prepare_masks(self, masks: CredentialMasks) -> None:
        try:
            for extension in self._device().extensions.values():
                if isinstance(extension, CredentialProvider):
                    await extension.prepare_masks(masks)
            return
        except Exception:
            pass
        # Provider exceptions can contain values; don't attach their traceback.
        raise MacroError("Could not prepare credential masking")

    async def _prepare_once(self, state: _ReplayMasks) -> CredentialMasks:
        async with state.lock:
            if not state.prepared:
                await self._prepare_masks(state.masks)
                state.prepared = True
        return state.masks

    async def _dispatch(
        self, step: MacroStep, credential: _Credential | None
    ) -> object:
        driver = self._device()
        args: dict[str, Any] = step.args
        operation: Awaitable[object]
        match step.action:
            case "tap":
                operation = driver.tap(args["x"], args["y"])
            case "swipe":
                operation = driver.swipe(
                    args["start_x"],
                    args["start_y"],
                    args["end_x"],
                    args["end_y"],
                    args["duration_ms"],
                )
            case "drag":
                operation = driver.drag(
                    args["start_x"],
                    args["start_y"],
                    args["end_x"],
                    args["end_y"],
                    args["duration"],
                )
            case "input_text":
                options = {
                    key: args[key] for key in ("clear", "stealth", "wpm") if key in args
                }
                operation = driver.input_text(args["text"], **options)
            case "start_app":
                operation = driver.start_app(args["package"], args.get("activity"))
            case "stop_app":
                operation = driver.stop_app(args["package"])
            case "open_url":
                operation = driver.open_url(args["url"])
            case "press_key_code":
                operation = driver.press_key_code(args["key_code"])
            case "button_press":
                operation = driver.press_button(args["button"])
            case "wait":
                operation = driver.wait(args["duration"])
            case "extension_call":
                operation = driver.actions.call(
                    args["name"], args["arguments"], audience="operator"
                )
            case "type_secret":
                assert credential is not None
                vault, package, field, name = credential
                operation = vault.type_field(package, field, name)
            case _:
                raise MacroError("Unknown macro action")
        return await operation

    def _stop(
        self,
        step: MacroStep,
        phase: MismatchPhase,
        observed: Fingerprint | None,
        masks: CredentialMasks,
    ) -> MacroMismatch:
        if masks.active or self.tainted:
            # A repair add-on must not be handed a device with visible credentials.
            raise MacroError(
                f"Macro mismatch at step {step.i} ({phase}) during credential replay; handoff refused"
            )
        return MacroMismatch(step, phase, observed)

    async def _play_step(
        self, step: MacroStep, credential: _Credential | None, masks: CredentialMasks
    ) -> None:
        before = None
        if step.before is not None:
            matched, before = await self._wait_for_guard(step.before)
            if not matched:
                raise self._stop(step, "before", before, masks)
        try:
            try:
                ok = await self._dispatch(step, credential) is not False
            finally:
                if masks.active:
                    self._tainted = True
        except MacroError:
            raise
        except Exception:
            # Never attach backend exception text or a credential traceback.
            ok = False
        if not ok:
            raise self._stop(step, "action", before, masks)
        if step.after is not None:
            matched, after = await self._wait_for_guard(step.after)
            if not matched:
                raise self._stop(step, "after", after, masks)
        self.executed_steps += 1

    async def play_step(self, step: MacroStep) -> None:
        """Check the before guard, dispatch the action, then check the after guard."""
        credential = self._preflight(step)
        with self._mask_scope() as state:
            masks = await self._prepare_once(state)
            await self._play_step(step, credential, masks)

    async def replay_macro(
        self,
        steps: Sequence[MacroStep],
        start_from_step: int = 0,
        max_steps: int | None = None,
    ) -> bool:
        """Replay selected rows; the zero-based offset is a position, not an ID."""
        self.executed_steps = 0
        if start_from_step < 0 or (max_steps is not None and max_steps < 0):
            raise MacroError("Step limits must not be negative")
        end = None if max_steps is None else start_from_step + max_steps
        selected = steps[start_from_step:end]
        credentials = [self._preflight(step) for step in selected]
        if not selected:
            return True
        with self._mask_scope() as state:
            masks = await self._prepare_once(state)
            for index, (step, credential) in enumerate(
                zip(selected, credentials, strict=True)
            ):
                if index:
                    # Only explicit macro waits are recorded as device actions.
                    await asyncio.sleep(self.delay_between_actions)
                logger.info("Replaying macro step %s", step.i)
                await self._play_step(step, credential, masks)
        return True
