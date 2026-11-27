"""Extension-owned checks and atomic local relationship transitions."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Generator,
    Iterator,
    Sequence,
)
from contextlib import AsyncExitStack, ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

from pydantic import BaseModel

from mobilerun.core.extensions.actions import (
    Action,
    ActionInfo,
    ActionSet,
    Declaration,
    _authorize_dispatch,
    action,
    device_action,
)
from mobilerun.core.extensions.registry import ActionOutcome, Binding, invoking_facade
from mobilerun.errors.base import CloudError, MobilerunError, is_value_safe
from mobilerun.errors.operators import (
    ExtensionActionError,
    ExtensionClosed,
    ExtensionConflict,
    ExtensionNotBound,
)
from mobilerun.models.cloud_task import CloudTaskInputs
from mobilerun.models.context import ContextItem
from mobilerun.utils.codes import verification_read_codes
from mobilerun.utils.redaction import mask_step_args

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device

_active: ContextVar[Extension | None] = ContextVar("extension_action", default=None)
_sensitive_action: ContextVar[bool] = ContextVar(
    "sensitive_extension_action", default=False
)
_observed_outcome: ContextVar[ActionOutcome | None] = ContextVar(
    "observed_extension_outcome", default=None
)


def _inside_bound_action(device: Device) -> bool:
    """Whether a named action is executing on this device or a sibling facade."""
    active = _active.get()
    bound = active.device if active is not None else None
    return bound is not None and bound._extension_state is device._extension_state


def _redact_bound_arguments(device: Device) -> None:
    outcome = _observed_outcome.get()
    if outcome is not None and _inside_bound_action(device):
        outcome.redact_arguments()


class _ExtensionAccess:
    """Retain the invoking device for actions reached through a facade."""

    def __init__(self, extension: Extension, facade: Device) -> None:
        self._extension = extension
        self._facade = facade

    @property
    def __class__(self) -> type[Extension]:
        return type(self._extension)

    @property
    def actions(self) -> ActionSet:
        return ActionSet(self._extension, facade=self._facade)

    @contextmanager
    def _invocation(self) -> Iterator[None]:
        from mobilerun.core.runs import recording_scope

        token = invoking_facade.set(self._facade)
        try:
            with recording_scope(enabled=self._facade.recording_allowed()):
                yield
        finally:
            invoking_facade.reset(token)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return getattr(self._extension, name)
        with self._invocation():
            member = getattr(self._extension, name)
        if not callable(member):
            return member

        if inspect.iscoroutinefunction(member):

            @wraps(member)
            async def invoke(*args: Any, **kwargs: Any) -> Any:
                with self._invocation():
                    return self._wrap_lazy(await member(*args, **kwargs))

            return invoke

        @wraps(member)
        def call(*args: Any, **kwargs: Any) -> Any:
            with self._invocation():
                result = member(*args, **kwargs)
            if inspect.isawaitable(result):

                async def await_result() -> Any:
                    with self._invocation():
                        return self._wrap_lazy(await result)

                return await_result()
            return self._wrap_lazy(result)

        return call

    def _wrap_lazy(self, result: Any) -> Any:
        if inspect.isasyncgen(result):
            return self._iterate_async(result)
        if inspect.isgenerator(result):
            return self._iterate(result)
        return result

    def _iterate(self, iterator: Generator[Any, Any, Any]) -> Generator[Any, Any, Any]:
        advance: Callable[[], Any] = partial(next, iterator)
        try:
            while True:
                with self._invocation():
                    item = advance()
                try:
                    value = yield item
                except GeneratorExit:
                    raise
                except BaseException as error:
                    advance = partial(iterator.throw, error)
                else:
                    advance = partial(iterator.send, value)
        except StopIteration as stop:
            return stop.value
        finally:
            with self._invocation():
                iterator.close()

    async def _iterate_async(
        self, iterator: AsyncGenerator[Any, Any]
    ) -> AsyncGenerator[Any, Any]:
        advance: Callable[[], Awaitable[Any]] = iterator.__anext__
        try:
            while True:
                with self._invocation():
                    item = await advance()
                try:
                    value = yield item
                except GeneratorExit:
                    raise
                except BaseException as error:
                    advance = partial(iterator.athrow, error)
                else:
                    advance = partial(iterator.asend, value)
        except StopAsyncIteration:
            return
        finally:
            with self._invocation():
                await iterator.aclose()


def _sanitize_safe_chain(error: MobilerunError) -> None:
    """Keep safe explicit causes only; discard implicit backend contexts."""
    seen: set[int] = set()
    current: BaseException = error
    while True:
        seen.add(id(current))
        # Notes are free-text provider data, including on otherwise safe causes.
        current.__notes__ = []
        current.__context__ = None
        current.__suppress_context__ = True
        cause = current.__cause__
        if (
            not isinstance(cause, MobilerunError)
            or not is_value_safe(cause)
            or id(cause) in seen
        ):
            current.__cause__ = None
            return
        current = cause


class Extension:
    """Standalone resource linked through one shared binding coordinator."""

    default_name: ClassVar[str] = "extension"
    exclusive: ClassVar[str | None] = None
    cloud_native: ClassVar[bool] = False
    role: ClassVar[str | None] = None
    _declarations: ClassVar[dict[str, Declaration]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        declarations: dict[str, Declaration] = {}
        for base in reversed(cls.__mro__[1:]):
            declarations.update(vars(base).get("_declarations", {}))
        for name, member in vars(cls).items():
            if inspect.isfunction(member) and "_extension_action" in member.__dict__:
                declarations[name] = member.__dict__["_extension_action"]
            elif inspect.isfunction(member) and name in declarations:
                declaration = declarations[name]
                setattr(
                    cls,
                    name,
                    (device_action if declaration.requires_device else action)(
                        public=declaration.public,
                        sensitive_parameters=declaration.sensitive_parameters,
                    )(member),
                )
        cls._declarations = declarations

    def __init__(self, *, name: str | None = None) -> None:
        self._name = self.default_name if name is None else name
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", self._name):
            raise ValueError("Extension names must match [a-z][a-z0-9_]{0,31}")
        self._binding: Binding | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._action_lock = asyncio.Lock()
        self._closed = False
        self._closing = False
        self._close_mutation_started = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def device(self) -> Device | None:
        if self._binding is None:
            return None
        if self._binding.uncertain:
            raise ExtensionNotBound(f"Extension '{self.name}' binding is uncertain")
        return self._binding.facade

    @property
    def actions(self) -> ActionSet:
        return ActionSet(self)

    def _action_entries(self) -> tuple[Action, ...]:
        entries = []
        for name, declaration in self._declarations.items():
            member = inspect.getattr_static(type(self), name)
            target, retry = getattr(member, "_billing_handler", (None, None))
            entries.append(
                Action.compile(
                    self.name,
                    name,
                    member.__get__(self),
                    declaration,
                    retry_call=retry.__get__(self) if target is member else None,
                )
            )
        return tuple(entries)

    async def check_bind(self, device: Device) -> Sequence[Extension]:
        return tuple(
            other
            for other in device._extension_state.view.values()
            if self.exclusive is not None
            and other.exclusive == self.exclusive
            and other is not self
        )

    async def on_bind(self, device: Device) -> None:
        """Perform backend linking; no local registry mutation."""

    async def on_unbind(self, device: Device) -> None:
        """Remove backend linking; no local registry mutation."""

    async def _check_unbind(self, device: Device) -> None:
        """Reject a detach before starting a potentially mutating hook."""

    async def on_close(self) -> None:
        """Release only resources owned by this extension."""

    async def context(self) -> Sequence[ContextItem]:
        return ()

    async def cloud_task_inputs(self) -> CloudTaskInputs:
        """Contribute server-side task references, never secret values."""
        return CloudTaskInputs()

    def _can_switch(self, device: Device) -> bool:
        return False

    async def _reconcile_bind_failure(
        self, device: Device, failure: BaseException
    ) -> None:
        """Resource-specific read-back after an ambiguous backend mutation."""

    def _matches_remote(self, observed: object) -> bool:
        """Whether a remote binding observation identifies this resource."""
        return False

    def _observe_remote_binding(self, device: Device, observed: object | None) -> None:
        """Update resource-owned backend state after binding read-back."""

    def _check_name(self, device: Device, conflicts: Sequence[Extension] = ()) -> None:
        occupied = device._extension_state.view.get(self.name)
        if (
            self.name == "device"
            or any(self.name in vars(base) for base in type(device).__mro__)
            or self.name in vars(device)
        ):
            raise ExtensionConflict(f"Extension name '{self.name}' is reserved")
        if occupied is not None and occupied not in conflicts:
            raise ExtensionConflict(f"Extension name '{self.name}' is already bound")

    async def bind(
        self,
        device: Device,
        *,
        close_on_device_close: bool = False,
        force: bool = False,
    ) -> Self:
        if _active.get() is not None:
            raise ExtensionConflict("Binding cannot change inside an extension action")
        async with self._lifecycle_lock:
            registry = device._extension_state
            async with registry.lock:
                if self._closed or self._closing or registry.closing:
                    raise ExtensionConflict(
                        "Cannot bind a closed extension or closing device"
                    )
                if self._binding is not None:
                    if self._binding.facade._unwrap() is device._unwrap():
                        if self._binding.uncertain:
                            raise ExtensionConflict(
                                "Binding is uncertain; unbind it first"
                            )
                        return self
                    raise ExtensionConflict(
                        f"Extension '{self.name}' is already bound to another device"
                    )
                conflicts = tuple(await self.check_bind(device))
                self._check_name(device, conflicts)
                if conflicts and not force:
                    old = conflicts[0]
                    raise ExtensionConflict(
                        f"{device.identifier or 'device'} already has {self.exclusive} '{old.name}'; unbind it first"
                    )
                self._action_entries()
                removed: list[Binding] = []
                registry.force = force
                registry.switching = bool(
                    conflicts and force and self._can_switch(device)
                )
                detaching: Binding | None = None
                try:
                    for old in conflicts:
                        async with old._action_lock:
                            binding = registry.bindings[old.name]
                            await old._check_unbind(binding.facade)
                            detaching = binding
                            await old.on_unbind(binding.facade)
                            registry.bindings.pop(old.name)
                            # Keep the relationship until the transition settles:
                            # concurrent close/unbind must wait for this registry
                            # lock rather than release a rollback resource early.
                            binding.uncertain = True
                            removed.append(binding)
                            detaching = None
                    await self.on_bind(device)
                except BaseException as failure:
                    registry.switching = False
                    if detaching is not None:
                        detaching.uncertain = True
                    for binding in removed:
                        try:
                            async with binding.extension._action_lock:
                                await binding.extension.on_bind(binding.facade)
                                binding.uncertain = False
                        except BaseException:
                            binding.uncertain = True
                            failure.add_note(
                                f"Could not restore extension '{binding.extension.name}'; checking remote state"
                            )
                        registry.bindings[binding.extension.name] = binding
                        binding.extension._binding = binding
                    if detaching is not None:
                        await detaching.extension._finish_reconciliation(
                            device, failure
                        )
                    for binding in removed:
                        # A successful restore response does not prove remote
                        # truth. Read back through each restored resource's hook.
                        await binding.extension._finish_reconciliation(
                            binding.facade, failure
                        )
                    if detaching is None and not removed:
                        await self._finish_reconciliation(device, failure)
                    raise failure
                finally:
                    registry.force = False
                    registry.switching = False
                binding = Binding(self, device, close_on_device_close)
                for previous in removed:
                    previous.extension._binding = None
                registry.bindings[self.name] = binding
                self._binding = binding
        return self

    async def unbind(self) -> None:
        if _active.get() is not None:
            raise ExtensionConflict("Binding cannot change inside an extension action")
        async with self._lifecycle_lock:
            await self._unbind_locked()

    async def _finish_reconciliation(
        self, device: Device, failure: BaseException
    ) -> None:
        # Hold the registry lock until read-back finishes, including repeated
        # cancellation, so no competing transition observes stale certainty.
        reconciliation = asyncio.create_task(
            self._reconcile_bind_failure(device, failure)
        )
        while not reconciliation.done():
            try:
                await asyncio.shield(reconciliation)
            except asyncio.CancelledError:
                continue
        reconciliation.result()

    async def _unbind_locked(self, *, for_close: bool = False) -> None:
        if self._binding is None:
            return
        binding = self._binding
        registry = binding.facade._extension_state
        async with registry.lock:
            if self._binding is not binding:
                return  # A concurrent force transition already detached us.
            async with self._action_lock:
                await self._check_unbind(binding.facade)
                try:
                    if for_close:
                        self._close_mutation_started = True
                    await self.on_unbind(binding.facade)
                except BaseException as failure:
                    binding.uncertain = True
                    await self._finish_reconciliation(binding.facade, failure)
                    raise
                registry.bindings.pop(self.name)
                self._binding = None

    async def close(self) -> None:
        if _active.get() is not None:
            raise ExtensionConflict("Binding cannot change inside an extension action")
        try:
            async with self._lifecycle_lock:
                if self._closed:
                    return
                if not self._closing and self._binding is not None:
                    binding = self._binding
                    async with binding.facade._extension_state.lock:
                        if self._binding is binding:
                            await self._check_unbind(binding.facade)
                self._closing = True
                await self._unbind_locked(for_close=True)
                async with self._action_lock:
                    self._close_mutation_started = True
                    await self.on_close()
                    self._closed = True
        except BaseException:
            if not self._close_mutation_started:
                self._closing = False
            raise

    async def _invoke_action(
        self,
        fn: Callable[..., Awaitable[Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> object:
        active = _active.get()
        if active is not None and active is not self:
            raise ExtensionActionError(
                "Cross-extension nested action calls are not allowed"
            )
        nested = active is self
        if self._closed or (self._closing and not nested):
            raise ExtensionClosed(f"Extension '{self.name}' is closing or closed")
        entry = self.actions.get(f"{self.name}.{fn.__name__}")

        def device_for_call() -> Device | None:
            if self._closed or (self._closing and not nested):
                raise ExtensionClosed(f"Extension '{self.name}' is closing or closed")
            device = self.device
            if device is not None and device._extension_state.closing:
                raise ExtensionNotBound("Device is closing")
            if device is None and entry.requires_device:
                raise ExtensionNotBound(
                    f"'{entry.name}' needs a bound device; await {self.name}.bind(device) first"
                )
            return device

        from mobilerun.core.extensions.actions import iter_sensitive_values
        from mobilerun.core.runs import recording_scope
        from mobilerun.utils.redaction import secret_scope

        lock = nullcontext() if nested else self._action_lock
        async with lock, AsyncExitStack() as dispatch_scope:
            device = device_for_call()
            facade = invoking_facade.get()
            if device is not None and (
                facade is None or facade._extension_state is not device._extension_state
            ):
                facade = device
            if facade is not None and not facade.recording_allowed():
                dispatch_scope.enter_context(recording_scope(enabled=False))
            values = inspect.signature(entry._call).bind(*args[1:], **kwargs)
            values.apply_defaults()
            arguments = {
                key: value for key, value in values.arguments.items() if key != "self"
            }
            validated = entry.validate(arguments)
            _authorize_dispatch(
                ActionInfo(
                    self.role, self.name, fn.__name__, entry.supports_billing_retry
                )
            )
            token = _active.set(self)
            sensitivity_token = _sensitive_action.set(
                _sensitive_action.get() or bool(entry.sensitive_parameters)
            )
            outcome = ActionOutcome()
            outcome_token = _observed_outcome.set(outcome) if not nested else None
            try:
                with ExitStack() as secrets:
                    for key in entry.sensitive_parameters:
                        for value in iter_sensitive_values(validated[key]):
                            secrets.enter_context(secret_scope(value))
                    async with AsyncExitStack() as stack:
                        if device is not None:
                            registry = device._extension_state
                            if entry.sensitive_parameters:
                                registry.sensitive_ui = True
                                if nested:
                                    _redact_bound_arguments(device)
                            recorder = (
                                facade.recording_device()
                                if facade is not None and facade.recording_allowed()
                                else None
                            )
                            observe = (
                                registry.observers.get(recorder)
                                if recorder is not None and not nested
                                else None
                            )
                            if observe is not None:
                                model = cast(type[BaseModel], entry._arguments)
                                safe = model.model_construct(**validated).model_dump(
                                    mode="json", exclude=set(entry.sensitive_parameters)
                                )
                                safe.update(
                                    dict.fromkeys(entry.sensitive_parameters, "***")
                                )
                                safe = mask_step_args(
                                    safe,
                                    contains_code=registry.matches_sensitive_argument,
                                )
                                outcome.arguments = safe
                                await stack.enter_async_context(
                                    observe(facade or device, entry.name, safe, outcome)
                                )
                        try:
                            result = entry.validate_result(
                                await fn(self, device, **validated)
                                if entry.requires_device
                                else await fn(self, **validated)
                            )
                        except BaseException as error:
                            outcome.error = type(error).__name__
                            raise
                        if result is False:
                            outcome.error = "returned_false"
                        if device is not None:
                            device._extension_state.remember_verification_codes(
                                *verification_read_codes(self.role, fn.__name__, result)
                            )
                    return result
            except Exception as error:
                if isinstance(error, MobilerunError) and is_value_safe(error):
                    _sanitize_safe_chain(error)
                    raise
                cause_type = type(error).__name__
                category = "unexpected failure"
                if isinstance(error, TimeoutError):
                    category = "timed out"
                    timeout = validated.get("timeout")
                    if (
                        "timeout" not in entry.sensitive_parameters
                        and timeout is not None
                        and type(timeout) in (int, float)
                        and 0 <= timeout < float("inf")
                    ):
                        seconds = (
                            str(timeout) if type(timeout) is int else f"{timeout:g}"
                        )
                        category += f" after {seconds}s"
                elif isinstance(error, CloudError):
                    category = "cloud request failed"
                elif isinstance(error, ConnectionError):
                    category = "connection failed"
                wrapped = ExtensionActionError(
                    f"Action {entry.name} failed: {cause_type} ({category})",
                    cause_type=cause_type,
                )
                if (
                    logging.getLogger("mobilerun").isEnabledFor(logging.DEBUG)
                    and not _sensitive_action.get()
                    # A rejected safe-rendering claim is not a debug escape hatch.
                    and not isinstance(error, CloudError)
                ):
                    raise wrapped from error
                raise wrapped from None
            finally:
                if outcome_token is not None:
                    _observed_outcome.reset(outcome_token)
                _sensitive_action.reset(sensitivity_token)
                _active.reset(token)
