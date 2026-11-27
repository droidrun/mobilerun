"""Public, JSON-safe metadata supplied by extensions."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from pydantic import JsonValue, TypeAdapter

JSONValue: TypeAlias = JsonValue


@dataclass(frozen=True)
class ContextItem:
    source: str
    kind: str
    data: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        import json

        copied = TypeAdapter(dict[str, JsonValue]).validate_python(dict(self.data))
        json.dumps(copied, allow_nan=False)
        object.__setattr__(self, "data", copied)
