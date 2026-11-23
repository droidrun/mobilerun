"""Small, value-safe credential model and provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from mobilerun.core.extensions import Extension, action, device_action
from mobilerun.errors import (
    CredentialFieldNotFound,
    ExtensionActionError,
)
from mobilerun.errors.credentials import CredentialFormatError
from mobilerun.models.context import ContextItem, JSONValue
from mobilerun.models.credentials import Credential, CredentialMetadata

if TYPE_CHECKING:
    from mobilerun.core.devices.base import Device


class CredentialProvider(Extension, ABC):
    default_name = "vault"

    @abstractmethod
    async def _find(
        self,
        package: str | None = None,
        field: str = "password",
        name: str | None = None,
    ) -> Credential | None:
        """Return the first matching credential, or None."""

    @abstractmethod
    async def _list_credentials(self, package: str | None = None) -> list[Credential]:
        """Read credentials internally."""

    @action()
    async def list(self, package: str | None = None) -> list[CredentialMetadata]:
        """List available credential names and fields without their values."""
        return [
            {"package": item.package, "name": item.name, "fields": list(item.fields)}
            for item in await self._list_credentials(package)
        ]

    @device_action()
    async def type_field(
        self, device: Device, package: str | None, field: str, name: str | None = None
    ) -> None:
        """Type a credential field directly into the bound device."""
        credential = await self._find(package, field, name)
        if credential is None:
            raise CredentialFieldNotFound("Credential field not found")
        if not await device._input_secret(credential.fields[field]):
            raise ExtensionActionError("Device rejected credential input")

    async def context(self) -> tuple[ContextItem, ...]:
        return (
            ContextItem(
                self.name,
                "credentials",
                {"credentials": cast(JSONValue, await self.list())},
            ),
        )


def normalize(mapping: Mapping[str, object]) -> list[Credential]:
    """Convert flat names or package → name → fields into one model."""
    result = []
    for key, value in mapping.items():
        if isinstance(value, str):
            result.append(Credential(None, key, {"password": value}))
        elif isinstance(value, dict):
            for name, fields in value.items():
                if not isinstance(fields, dict) or not all(
                    isinstance(v, str) for v in fields.values()
                ):
                    raise CredentialFormatError(
                        "Credential fields must map field types to strings"
                    )
                result.append(Credential(key, name, fields.copy()))
        else:
            raise CredentialFormatError(
                "Credentials must be strings or package mappings"
            )
    return result
