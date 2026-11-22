"""One root registry and lock, shared by device facades."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device
    from mobilerun.core.extensions.base import Extension


@dataclass
class Binding:
    extension: Extension
    facade: Device
    owned: bool
    uncertain: bool = False


class ExtensionView(Mapping[str, "Extension"]):
    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    def __getitem__(self, name: str) -> Extension:
        return self._registry.bindings[name].extension

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
        self.observers: list[Callable[[str, dict[str, Any]], None]] = []
        self.switching = False
        self.force = False
