"""Typed async action declarations and consumer-independent catalogs."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Concatenate,
    Literal,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
    get_type_hints,
    overload,
)

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, create_model

from mobilerun.core.extensions.registry import invoking_facade
from mobilerun.errors.operators import ExtensionActionError
from mobilerun.models.billing import Billing
from mobilerun.models.context import JSONValue

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device
    from mobilerun.core.extensions.base import Extension

P = ParamSpec("P")
R = TypeVar("R")
S = TypeVar("S")
Audience = Literal["python", "operator", "cli"]


@dataclass(frozen=True)
class ActionInfo:
    """Value-free metadata supplied to pre-dispatch authorization guards."""

    extension_role: str | None
    extension_name: str | None
    action_name: str
    supports_billing_retry: bool


_dispatch_guards: ContextVar[tuple[Callable[[ActionInfo], None], ...]] = ContextVar(
    "extension_dispatch_guards", default=()
)


@contextmanager
def dispatch_guard(predicate: Callable[[ActionInfo], None]) -> Iterator[None]:
    """Authorize each declared action before dispatch; nested guards accumulate."""
    token = _dispatch_guards.set((*_dispatch_guards.get(), predicate))
    try:
        yield
    finally:
        _dispatch_guards.reset(token)


def _authorize_dispatch(info: ActionInfo) -> None:
    for predicate in _dispatch_guards.get():
        predicate(info)


class BillingRetry(Protocol):
    """Run a preflight-guarded operation, retrying only with authorized billing."""

    def __call__(
        self, operation: Callable[[Billing], Awaitable[R]], billing: Billing, /
    ) -> Awaitable[tuple[R, Billing]]: ...


@dataclass(frozen=True)
class Declaration:
    public: bool = True
    sensitive_parameters: frozenset[str] = frozenset()
    requires_device: bool = False


_RetryHandler = Callable[Concatenate[S, BillingRetry, P], Awaitable[R]]
_RetryCall = Callable[[BillingRetry, dict[str, Any]], Awaitable[object]]


@overload
def action(
    fn: Callable[P, Awaitable[R]],
    *,
    public: bool = True,
    sensitive_parameters: frozenset[str] = frozenset(),
) -> Callable[P, Awaitable[R]]: ...


@overload
def action(
    *, public: bool = True, sensitive_parameters: frozenset[str] = frozenset()
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]: ...


@overload
def action(
    *,
    public: bool = True,
    sensitive_parameters: frozenset[str] = frozenset(),
    billing_retry: _RetryHandler[S, P, R],
) -> Callable[
    [Callable[Concatenate[S, P], Awaitable[R]]],
    Callable[Concatenate[S, P], Awaitable[R]],
]: ...


def action(
    fn: Callable[..., Any] | None = None,
    *,
    public: bool = True,
    sensitive_parameters: frozenset[str] = frozenset(),
    billing_retry: Callable[..., Awaitable[Any]] | None = None,
) -> Any:
    """Declare a documented async extension method; direct calls stay typed."""
    decorate = _decorate(
        Declaration(public, frozenset(sensitive_parameters)),
        billing_retry=billing_retry,
    )
    return decorate if fn is None else decorate(fn)


@overload
def device_action(
    fn: Callable[Concatenate[S, Device, P], Awaitable[R]],
    *,
    public: bool = True,
    sensitive_parameters: frozenset[str] = frozenset(),
) -> Callable[Concatenate[S, P], Awaitable[R]]: ...


@overload
def device_action(
    *, public: bool = True, sensitive_parameters: frozenset[str] = frozenset()
) -> Callable[
    [Callable[Concatenate[S, Device, P], Awaitable[R]]],
    Callable[Concatenate[S, P], Awaitable[R]],
]: ...


def device_action(
    fn: Callable[..., Any] | None = None,
    *,
    public: bool = True,
    sensitive_parameters: frozenset[str] = frozenset(),
) -> Any:
    """Require and inject the bound facade, hiding it from public arguments."""
    decorate = _decorate(Declaration(public, frozenset(sensitive_parameters), True))
    return decorate if fn is None else decorate(fn)


def _decorate(
    declaration: Declaration,
    *,
    billing_retry: Callable[..., Awaitable[Any]] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError("Actions must be async methods")
        if not inspect.getdoc(fn):
            raise TypeError("Actions require a docstring")
        signature = inspect.signature(fn)
        for name, parameter in signature.parameters.items():
            if (
                name in declaration.sensitive_parameters
                and parameter.default is not parameter.empty
            ):
                raise TypeError("Sensitive action parameters cannot declare a default")

        if declaration.requires_device:
            parameters = list(signature.parameters.values())
            if (
                len(parameters) < 2
                or parameters[1].name != "device"
                or not (
                    parameters[1].annotation == "Device"
                    or parameters[1].annotation is _device_type()
                )
                or parameters[1].kind
                not in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ):
                raise TypeError(
                    "Device actions require device: Device immediately after self"
                )
            signature = signature.replace(parameters=[parameters[0], *parameters[2:]])

        @wraps(fn)
        async def invoke(*args: Any, **kwargs: Any) -> Any:
            extension = cast("Extension", args[0])
            return await extension._invoke_action(fn, args, kwargs)

        invoke.__dict__["_extension_action"] = declaration
        if billing_retry is not None:

            async def retry_call(
                extension: Extension, policy: BillingRetry, arguments: dict[str, Any]
            ) -> object:
                @wraps(fn)
                async def authorized(owner: Extension, /, **validated: Any) -> object:
                    return await billing_retry(owner, policy, **validated)

                return await extension._invoke_action(
                    authorized, (extension,), arguments
                )

            # wraps() copies metadata; only this exact wrapper grants consent handling.
            invoke.__dict__["_billing_handler"] = (invoke, retry_call)
        invoke.__dict__["__signature__"] = signature
        return invoke

    return decorate


def _device_type() -> type:
    # Lazy import avoids the Device -> Extension -> actions import cycle.
    from mobilerun.core.devices.base import Device

    return Device


@dataclass(frozen=True)
class Action:
    name: str
    owner: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    public: bool
    sensitive_parameters: frozenset[str]
    requires_device: bool
    _call: Callable[..., Awaitable[Any]] = field(repr=False, compare=False)
    _arguments: type = field(repr=False, compare=False)
    _returns: TypeAdapter[Any] = field(repr=False, compare=False)
    _retry_call: _RetryCall | None = field(default=None, repr=False, compare=False)

    @property
    def supports_billing_retry(self) -> bool:
        return self._retry_call is not None

    @classmethod
    def compile(
        cls,
        owner: str,
        name: str,
        fn: Callable[..., Awaitable[Any]],
        declaration: Declaration,
        *,
        description: str | None = None,
        retry_call: _RetryCall | None = None,
    ) -> Action:
        signature = inspect.signature(fn)
        raw = inspect.unwrap(fn)
        hints = get_type_hints(
            raw,
            globalns={**raw.__globals__, "Device": _device_type()},
            include_extras=True,
        )
        fields: dict[str, Any] = {}
        for parameter in signature.parameters.values():
            if parameter.name == "self":
                continue
            if parameter.kind in (
                parameter.POSITIONAL_ONLY,
                parameter.VAR_POSITIONAL,
                parameter.VAR_KEYWORD,
            ):
                raise TypeError(f"Action {name} has an unsupported parameter interface")
            if parameter.name not in hints:
                raise TypeError(f"Action {name} requires typed parameters")
            default = ... if parameter.default is parameter.empty else parameter.default
            fields[parameter.name] = (hints[parameter.name], default)
        if "return" not in hints:
            raise TypeError(f"Action {name} requires a return annotation")
        if not declaration.sensitive_parameters <= fields.keys():
            raise TypeError(f"Action {name} declares an unknown sensitive parameter")
        arguments = create_model(
            f"{owner}_{name}_Arguments",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )
        returns = TypeAdapter(hints["return"])
        input_schema = arguments.model_json_schema()
        _sanitize_sensitive_schema(input_schema, declaration.sensitive_parameters)
        return cls(
            f"{owner}.{name}",
            owner,
            description or inspect.getdoc(fn) or name,
            input_schema,
            returns.json_schema(),
            declaration.public,
            declaration.sensitive_parameters,
            declaration.requires_device,
            fn,
            arguments,
            returns,
            retry_call,
        )

    def validate(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        try:
            model = self._arguments.model_validate(dict(arguments))
            return {name: model.__dict__[name] for name in type(model).model_fields}
        except ValidationError:
            raise ExtensionActionError(f"Invalid arguments for {self.name}") from None

    def validate_result(self, result: object) -> object:
        try:
            return self._returns.validate_python(result)
        except ValidationError:
            raise ExtensionActionError(f"Invalid result from {self.name}") from None

    def serialize(self, result: object) -> JSONValue:
        return cast(JSONValue, self._returns.dump_python(result, mode="json"))


def iter_sensitive_values(value: object) -> Iterator[str]:
    """Yield scalar strings from nested values, including mapping keys."""
    if isinstance(value, Enum):
        yield from iter_sensitive_values(value.value)
    elif isinstance(value, BaseModel):
        yield from iter_sensitive_values(value.model_dump())
    elif is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            yield from iter_sensitive_values(getattr(value, item.name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from iter_sensitive_values(key)
            yield from iter_sensitive_values(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from iter_sensitive_values(item)
    elif value is not None:
        yield str(value)


def _sanitize_sensitive_schema(schema: dict[str, Any], names: frozenset[str]) -> None:
    definitions = schema.get("$defs", {})
    visited: set[str] = set()

    def scrub(node: Any) -> None:
        if isinstance(node, dict):
            for key in ("default", "examples", "example", "const", "enum", "pattern"):
                node.pop(key, None)
            reference = node.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                name = reference.removeprefix("#/$defs/")
                if name not in visited and name in definitions:
                    visited.add(name)
                    scrub(definitions[name])
            for child in node.values():
                scrub(child)
        elif isinstance(node, list):
            for child in node:
                scrub(child)

    for name in names:
        scrub(schema.get("properties", {}).get(name, {}))


class ActionSet:
    """A live catalog; stale callers still validate current binding state."""

    def __init__(
        self, source: Device | Extension, *, facade: Device | None = None
    ) -> None:
        self._source = source
        self._facade = facade

    def list(self, *, audience: Audience = "python") -> tuple[Action, ...]:
        return tuple(
            entry
            for entry in self._source._action_entries()
            if audience == "python" or entry.public
        )

    def get(self, name: str) -> Action:
        for entry in self.list():
            if entry.name == name:
                return entry
        raise ExtensionActionError(f"Unknown action: {name}")

    async def call(
        self,
        name: str,
        arguments: Mapping[str, JSONValue],
        *,
        audience: Audience = "python",
        timeout: float | None = None,
        billing_retry: BillingRetry | None = None,
    ) -> object:
        """Invoke a validated action, optionally providing billing consent policy."""
        entry = self.get(name)
        if audience != "python" and not entry.public:
            raise ExtensionActionError(f"Action {name} is restricted")
        validated = entry.validate(arguments)
        if entry.owner == "device":
            _authorize_dispatch(ActionInfo(None, None, name.partition(".")[2], False))
        token = invoking_facade.set(self._facade) if self._facade is not None else None
        try:
            async with asyncio.timeout(timeout):
                if billing_retry is not None:
                    if entry._retry_call is None:
                        raise ExtensionActionError(
                            f"Action {name} does not support billing retry"
                        )
                    return entry.validate_result(
                        await entry._retry_call(billing_retry, validated)
                    )
                return entry.validate_result(await entry._call(**validated))
        finally:
            if token is not None:
                invoking_facade.reset(token)
