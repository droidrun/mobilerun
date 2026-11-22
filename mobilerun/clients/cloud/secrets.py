"""Functional vault SDK boundaries; package and cache policy belongs to core."""

from typing import Literal

from mobilerun_sdk import APIError, AsyncMobilerun
from mobilerun_sdk.types.credential_list_response import Item
from mobilerun_sdk.types.credentials.package_create_response import (
    PackageCreateResponse,
)
from mobilerun_sdk.types.credentials.package_list_response import Data
from mobilerun_sdk.types.credentials.packages.credential_create_response import (
    CredentialCreateResponse,
)
from mobilerun_sdk.types.credentials.packages.credential_delete_response import (
    CredentialDeleteResponse,
)
from mobilerun_sdk.types.credentials.packages.credentials.field_create_response import (
    FieldCreateResponse,
)
from mobilerun_sdk.types.credentials.packages.credentials.field_delete_response import (
    FieldDeleteResponse,
)
from mobilerun_sdk.types.credentials.packages.credentials.field_update_response import (
    FieldUpdateResponse,
)

from mobilerun.clients.cloud.billing import requires
from mobilerun.clients.cloud.errors import cloud_error
from mobilerun.core.devices.components.identifiers import secret_scope
from mobilerun.models.billing import Billing

FieldType = Literal[
    "email",
    "username",
    "password",
    "api_token",
    "phone_number",
    "two_factor_secret",
    "backup_codes",
]


@requires(Billing.NO_COMMERCE)
async def list_secrets(client: AsyncMobilerun, *, billing: Billing) -> list[Item]:
    """List all credential pages."""
    items, page = [], 1
    try:
        while True:
            response = await client.credentials.list(page=page)
            items.extend(response.items)
            if not response.pagination.has_next:
                return items
            page += 1
    except APIError as exc:
        raise cloud_error(exc, "List vault credentials") from exc


@requires(Billing.NO_COMMERCE)
async def list_package(
    client: AsyncMobilerun, package: str, *, billing: Billing
) -> list[Data]:
    """List credentials in one package."""
    try:
        return (await client.credentials.packages.list(package)).data
    except APIError as exc:
        raise cloud_error(exc, "List vault package") from exc


@requires(Billing.NO_COMMERCE)
async def create_package(
    client: AsyncMobilerun, package: str, *, billing: Billing
) -> PackageCreateResponse:
    """Create a vault package."""
    try:
        return await client.credentials.packages.create(package_name=package)
    except APIError as exc:
        raise cloud_error(exc, "Create vault package") from exc


@requires(Billing.NO_COMMERCE)
async def save(
    client: AsyncMobilerun,
    package: str,
    name: str,
    fields: dict[FieldType, str],
    *,
    billing: Billing,
) -> CredentialCreateResponse:
    """Save a credential in an existing package."""
    with secret_scope(*fields.values()):
        try:
            return await client.credentials.packages.credentials.create(
                package,
                credential_name=name,
                fields=[
                    {"field_type": key, "value": value} for key, value in fields.items()
                ],
            )
        except APIError as exc:
            raise cloud_error(exc, "Save vault credential") from exc


@requires(Billing.NO_COMMERCE)
async def create_field(
    client: AsyncMobilerun,
    package: str,
    name: str,
    field: FieldType,
    value: str,
    *,
    billing: Billing,
) -> FieldCreateResponse:
    """Add a credential field."""
    with secret_scope(value):
        try:
            return await client.credentials.packages.credentials.fields.create(
                name,
                package_name=package,
                field_type=field,
                value=value,
            )
        except APIError as exc:
            raise cloud_error(exc, "Create vault field") from exc


@requires(Billing.NO_COMMERCE)
async def update(
    client: AsyncMobilerun,
    package: str,
    name: str,
    field: FieldType,
    value: str,
    *,
    billing: Billing,
) -> FieldUpdateResponse:
    """Update a credential field."""
    with secret_scope(value):
        try:
            return await client.credentials.packages.credentials.fields.update(
                field,
                package_name=package,
                credential_name=name,
                value=value,
            )
        except APIError as exc:
            raise cloud_error(exc, "Update vault field") from exc


@requires(Billing.NO_COMMERCE)
async def delete(
    client: AsyncMobilerun,
    package: str,
    name: str,
    field: FieldType | None = None,
    *,
    billing: Billing,
) -> CredentialDeleteResponse | FieldDeleteResponse:
    """Delete a credential or field."""
    try:
        if field is None:
            return await client.credentials.packages.credentials.delete(
                name, package_name=package
            )
        return await client.credentials.packages.credentials.fields.delete(
            field,
            package_name=package,
            credential_name=name,
        )
    except APIError as exc:
        raise cloud_error(exc, "Delete vault credential") from exc
