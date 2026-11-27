"""Value-safe credential data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from typing_extensions import TypedDict


@dataclass
class Credential:
    package: str | None
    name: str
    fields: dict[str, str] = field(repr=False)

    def __repr__(self) -> str:
        return f"Credential(package={self.package!r}, name={self.name!r}, fields={list(self.fields)!r})"


class CredentialLookup(Protocol):
    """Minimal lookup contract consumed by shared credential clients."""

    async def find(
        self,
        package: str | None = None,
        field: str = "password",
        name: str | None = None,
    ) -> Credential | None: ...


class CredentialMetadata(TypedDict):
    package: str | None
    name: str
    fields: list[str]
