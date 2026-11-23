"""Mobilerun Cloud vault credentials, cached per provider instance."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterable
from contextlib import ExitStack
from typing import Any, cast, get_args

from mobilerun.clients.cloud import secrets
from mobilerun.clients.cloud.client import cloud_client
from mobilerun.config.settings import get_settings
from mobilerun.core.credentials.dict import DictCredentials
from mobilerun.core.extensions import action
from mobilerun.errors.cloud import CloudConflict, CloudResourceNotFound
from mobilerun.models.billing import Billing
from mobilerun.models.cloud_task import CloudCredentialReference, CloudTaskInputs
from mobilerun.models.credentials import Credential
from mobilerun.utils.redaction import _redact_secrets, secret_scope

FIELD_ORDER = get_args(secrets.FieldType)
FIELD_TYPES = frozenset(FIELD_ORDER)


async def _write_secret_values(
    operation: Awaitable[object], values: Iterable[str]
) -> None:
    with ExitStack() as stack:
        for value in values:
            stack.enter_context(secret_scope(value))
        try:
            await operation
            return
        except Exception as exc:
            message = _redact_secrets(str(exc))
            try:
                sanitized = type(exc)(message)
            except Exception:
                sanitized = RuntimeError(message)
        raise sanitized from None


def _validate_field(field: str) -> secrets.FieldType:
    if field not in FIELD_TYPES:
        raise ValueError(
            f"Invalid credential field type {field!r}; expected one of: {', '.join(sorted(FIELD_TYPES))}"
        )
    # The membership check above validates the external string against the SDK union.
    return cast(secrets.FieldType, field)


class CloudCredentials(DictCredentials):
    cloud_native = True

    async def cloud_task_inputs(self) -> CloudTaskInputs:
        return CloudTaskInputs(
            tuple(
                CloudCredentialReference(
                    self._package_names.get(
                        item.package.replace(".", "-"), item.package
                    ),
                    item.name,
                )
                for item in await self._list_credentials()
                if item.package is not None
            )
        )

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        billing: Billing | str | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__({}, name=name)
        self._billing = Billing(
            get_settings().MOBILERUN_BILLING if billing is None else billing
        )
        self._client = cloud_client(api_key, base_url, billing=self._billing)
        self._loaded = False
        self._package_cache = {}
        self._package_names: dict[str, str] = {}
        self._sources: dict[tuple[str, str], str] = {}
        self._lock = asyncio.Lock()

    async def _list_credentials(self, package: str | None = None) -> list[Credential]:
        async with self._lock:
            if package is not None:
                if "." in package:
                    self._package_names[package.replace(".", "-")] = package
                if package not in self._package_cache:
                    response = await self._list_package(package)
                    for item in response:
                        if (
                            getattr(item, "source", None) == "onepassword"
                            or getattr(getattr(item, "link", None), "provider", None)
                            == "onepassword"
                        ):
                            self._sources[
                                (package.replace(".", "-"), item.credential_name)
                            ] = "1password"
                    self._package_cache[package] = [
                        Credential(
                            package,
                            item.credential_name,
                            {entry.field_type: entry.value for entry in item.fields},
                        )
                        for item in response
                    ]
                return list(self._package_cache[package])
            if not self._loaded:
                response = await secrets.list_secrets(
                    self._client, billing=self._billing
                )
                self._sources = {
                    (
                        item.package_name.replace(".", "-"),
                        item.credential_name,
                    ): "1password"
                    for item in response
                    if getattr(item, "source", None) == "onepassword"
                    or getattr(getattr(item, "link", None), "provider", None)
                    == "onepassword"
                }
                items = [
                    Credential(
                        item.package_name,
                        item.credential_name,
                        {entry.field_type: entry.value for entry in item.fields},
                    )
                    for item in response
                ]
                self._credentials = items
                self._loaded = True
        return await super()._list_credentials(package)

    @action()
    async def list(self, package: str | None = None) -> list[dict]:
        """List vault metadata without secret values."""
        items = await self._list_credentials(package)
        result = []
        for item in items:
            slug = (item.package or "").replace(".", "-")
            entry = {
                "package": self._package_names.get(slug, item.package),
                "name": item.name,
                "fields": list(item.fields),
            }
            if (slug, item.name) in self._sources:
                entry["source"] = self._sources[(slug, item.name)]
            result.append(entry)
        return result

    async def save(self, package: str, name: str, fields: dict[str, str]) -> None:
        if "." in package:
            self._package_names[package.replace(".", "-")] = package
        validated_fields: dict[secrets.FieldType, str] = {
            _validate_field(field): value for field, value in fields.items()
        }
        async with self._lock:
            try:
                existing = next(
                    (
                        item
                        for item in await self._list_package(package)
                        if item.credential_name == name
                    ),
                    None,
                )
                if existing is None:
                    try:
                        await secrets.list_package(
                            self._client, package, billing=self._billing
                        )
                    except CloudResourceNotFound:
                        try:
                            await secrets.create_package(
                                self._client, package, billing=self._billing
                            )
                        except CloudConflict:
                            pass
                    await _write_secret_values(
                        secrets.save(
                            self._client,
                            package,
                            name,
                            validated_fields,
                            billing=self._billing,
                        ),
                        validated_fields.values(),
                    )
                else:
                    existing_fields = {item.field_type for item in existing.fields}
                    for field, value in validated_fields.items():
                        operation = (
                            secrets.update
                            if field in existing_fields
                            else secrets.create_field
                        )
                        await _write_secret_values(
                            operation(
                                self._client,
                                package,
                                name,
                                field,
                                value,
                                billing=self._billing,
                            ),
                            (value,),
                        )
            finally:
                self._loaded = False
                self._credentials = []
                self._package_cache.clear()

    async def update(self, package: str, name: str, field: str, value: str) -> None:
        validated_field = _validate_field(field)
        async with self._lock:
            try:
                await _write_secret_values(
                    secrets.update(
                        self._client,
                        package,
                        name,
                        validated_field,
                        value,
                        billing=self._billing,
                    ),
                    (value,),
                )
            finally:
                self._loaded = False
                self._credentials = []
                self._package_cache.clear()

    async def delete(self, package: str, name: str, field: str | None = None) -> None:
        validated_field = _validate_field(field) if field is not None else None
        async with self._lock:
            try:
                await secrets.delete(
                    self._client, package, name, validated_field, billing=self._billing
                )
            finally:
                self._loaded = False
                self._credentials = []
                self._package_cache.clear()

    async def on_close(self) -> None:
        await self._client.close()

    async def _list_package(self, package: str) -> list[Any]:
        try:
            return await secrets.list_package(
                self._client, package, billing=self._billing
            )
        except CloudResourceNotFound:
            return []
