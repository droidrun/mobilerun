"""Typed async action declarations and consumer-independent catalogs."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Concatenate,
    Literal,
    ParamSpec,
    TypeVar,
    cast,
    get_type_hints,
    overload,
)

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, create_model

from mobilerun.errors.operators import ExtensionActionError
from mobilerun.models.context import JSONValue

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device
    from mobilerun.core.extensions.base import Extension

P = ParamSpec("P")
R = TypeVar("R")
S = TypeVar("S")
Audience = Literal["python", "operator", "cli"]


@dataclass(frozen=True)
class Declaration:
    public: bool = True
    sensitive_parameters: frozenset[str] = frozenset()
    requires_device: bool = False


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


def action(
    fn: Callable[..., Any] | None = None,
    *,
    public: bool = True,
    sensitive_parameters: frozenset[str] = frozenset(),
) -> Any:
    """Declare a documented async extension method; direct calls stay typed."""
    decorate = _decorate(Declaration(public, frozenset(sensitive_parameters)))
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
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError("Actions must be async methods")
        if not inspect.getdoc(fn):
            raise TypeError("Actions require a docstring")
        for name, parameter in inspect.signature(fn).parameters.items():
            if (
                name in declaration.sensitive_parameters
                and parameter.default is not parameter.empty
            ):
                raise TypeError("Sensitive action parameters cannot declare a default")

        signature = inspect.signature(fn)
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

    @classmethod
    def compile(
        cls,
        owner: str,
        name: str,
        fn: Callable[..., Awaitable[Any]],
        declaration: Declaration,
        *,
        description: str | None = None,
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


def _sensitive_values(value: Any) -> Iterator[str]:
    """Register sensitive scalar values, including structured action arguments."""
    if isinstance(value, Enum):
        yield from _sensitive_values(value.value)
    elif isinstance(value, BaseModel):
        yield from _sensitive_values(value.model_dump())
    elif is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            yield from _sensitive_values(getattr(value, item.name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _sensitive_values(key)
            yield from _sensitive_values(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _sensitive_values(item)
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

    def __init__(self, source: Device | Extension) -> None:
        self._source = source

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
    ) -> object:
        entry = self.get(name)
        if audience != "python" and not entry.public:
            raise ExtensionActionError(f"Action {name} is restricted")
        validated = entry.validate(arguments)
        async with asyncio.timeout(timeout):
            return entry.validate_result(await entry._call(**validated))
