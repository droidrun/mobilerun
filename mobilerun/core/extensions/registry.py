"""One root registry and lock, shared by device facades."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractAsyncContextManager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import JsonValue

from mobilerun.utils.redaction import KnownCodes, mask_step_args

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device
    from mobilerun.core.extensions.base import Extension

invoking_facade: ContextVar[Device | None] = ContextVar(
    "extension_invoking_facade", default=None
)


@dataclass
class Binding:
    extension: Extension
    facade: Device
    owned: bool
    uncertain: bool = False


@dataclass
class ActionOutcome:
    """Dispatch outcome shared by observers while the secret scopes are active."""

    error: str | None = None
    arguments: dict[str, JsonValue] | None = None

    def redact_arguments(self) -> None:
        """Redact aliases discovered in a nested secret scope before it exits."""
        if self.arguments is not None:
            safe = mask_step_args(self.arguments)
            self.arguments.clear()
            self.arguments.update(safe)


def redact_action_value(
    value: JsonValue, *, contains_code: Callable[[str], bool] | None = None
) -> JsonValue:
    """Mask a structured value for existing extension evidence consumers."""
    return mask_step_args({"value": value}, contains_code=contains_code).get(
        "value", "***"
    )


ActionObserver = Callable[
    ["Device", str, dict[str, JsonValue], ActionOutcome],
    AbstractAsyncContextManager[None],
]


class ExtensionView(Mapping[str, "Extension"]):
    def __init__(self, registry: Registry, facade: Device | None = None) -> None:
        self._registry = registry
        self._facade = facade

    def __getitem__(self, name: str) -> Extension:
        extension = self._registry.bindings[name].extension
        return (
            self._facade._extension_access(extension)
            if self._facade is not None
            else extension
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self._registry.bindings)

    def __len__(self) -> int:
        return len(self._registry.bindings)


class Registry:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.bindings: dict[str, Binding] = {}
        self.view = ExtensionView(self)
        self.closing = False
        self.observers: dict[Device, ActionObserver] = {}
        self.sensitive_ui = False
        self.known_codes = KnownCodes()
        self.switching = False
        self.force = False

    def remember_verification_codes(self, *codes: str) -> None:
        if codes:
            self.known_codes.remember(*codes)
            self.sensitive_ui = True

    def contains_verification_code(self, text: str) -> bool:
        return self.known_codes.redact(text) != text

    def matches_sensitive_argument(self, text: str) -> bool:
        return (
            text in self.known_codes.sensitive_values
            or self.contains_verification_code(text)
        )

    def contains_sensitive_text(self, text: str) -> bool:
        return (
            self.contains_verification_code(text)
            or self.known_codes.redact_sensitive_text(text) != text
        )
