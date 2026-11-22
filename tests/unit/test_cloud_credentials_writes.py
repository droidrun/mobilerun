import asyncio
import logging
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from mobilerun_sdk import BadRequestError, NotFoundError

from mobilerun.core.credentials.cloud import CloudCredentials
from tests.support.devices import MemoryDevice


def test_vault_write_debug_logs_do_not_disclose_values(caplog):
    secret = "vault-write-secret-canary"
    log = logging.getLogger("mobilerun_sdk._base_client")
    caplog.set_level(logging.DEBUG, logger=log.name)

    async def update(client, package, name, field, value, **kwargs):
        log.debug("request %s", {"value": value})

    async def run():
        vault = CloudCredentials(api_key="dummy")
        with patch(
            "mobilerun.core.credentials.cloud.secrets.update",
            AsyncMock(side_effect=update),
        ):
            await vault.update("app", "work", "password", secret)
        await vault.close()

    asyncio.run(run())
    assert secret not in caplog.text


def test_vault_write_failure_redacts_secret():
    secret = "vault-error-secret-canary"

    async def run():
        vault = CloudCredentials(api_key="dummy")
        with patch(
            "mobilerun.core.credentials.cloud.secrets.update",
            AsyncMock(side_effect=RuntimeError(secret)),
        ):
            with pytest.raises(RuntimeError) as error:
                await vault.update("app", "work", "password", secret)
            assert secret not in str(error.value)
            assert error.value.__context__ is None
        await vault.close()

    asyncio.run(run())


class FakeVault:
    def __init__(self):
        self.data = {}
        self.credentials = self
        self.packages = self
        self.fields = Fields(self)

    async def list_all(self):
        return NS(data=[NS(package_name=p) for p in self.data])

    async def list(self, package_name=None, page=None):
        if page is not None:
            return NS(
                items=[
                    self.item(p, n, f)
                    for p, names in self.data.items()
                    for n, f in names.items()
                ],
                pagination=NS(has_next=False),
            )
        package_name = package_name.replace(".", "-")
        if package_name not in self.data:
            raise NotFoundError(
                "Package not found",
                response=httpx.Response(
                    404, request=httpx.Request("GET", "https://fake.invalid")
                ),
                body={"code": "credential_not_found"},
            )
        return NS(
            data=[
                self.item(package_name, n, f)
                for n, f in self.data[package_name].items()
            ]
        )

    def item(self, package, name, fields):
        return NS(
            package_name=package,
            credential_name=name,
            fields=[NS(field_type=k, value=v) for k, v in fields.items()],
        )

    async def create(self, package_name, credential_name=None, fields=None):
        package_name = package_name.replace(".", "-")
        if credential_name is None:
            if package_name in self.data:
                raise BadRequestError(
                    "Package exists",
                    response=httpx.Response(
                        400, request=httpx.Request("POST", "https://fake.invalid")
                    ),
                    body={"code": "package_exists"},
                )
            self.data[package_name] = {}
        else:
            self.data[package_name][credential_name] = {
                f["field_type"]: f["value"] for f in fields
            }

    async def delete(self, credential_name, *, package_name):
        package_name = package_name.replace(".", "-")
        del self.data[package_name][credential_name]


class Fields:
    def __init__(self, vault):
        self.vault = vault

    async def create(self, credential_name, *, package_name, field_type, value):
        package_name = package_name.replace(".", "-")
        fields = self.vault.data[package_name][credential_name]
        assert field_type not in fields
        fields[field_type] = value

    async def update(self, field_type, *, package_name, credential_name, value):
        package_name = package_name.replace(".", "-")
        fields = self.vault.data[package_name][credential_name]
        assert field_type in fields
        fields[field_type] = value

    async def delete(self, field_type, *, package_name, credential_name):
        package_name = package_name.replace(".", "-")
        del self.vault.data[package_name][credential_name][field_type]


def test_writes_create_update_delete_and_invalidate_cache(caplog):
    async def run():
        vault = CloudCredentials(api_key="dummy", base_url="https://example.invalid")
        fake = FakeVault()
        vault._client = NS(credentials=fake)
        await vault.bind(MemoryDevice())
        assert await vault.list() == []
        await vault.save("app.test", "work", {"password": "hidden-value"})
        assert (await vault.list())[0] == {
            "package": "app.test",
            "name": "work",
            "fields": ["password"],
        }
        await vault.save(
            "app.test", "work", {"password": "changed-value", "email": "a@example.com"}
        )
        await vault.update("app.test", "work", "password", "last-value")
        item = (await vault.list())[0]
        assert set(item["fields"]) == {"password", "email"}
        assert fake.data["app-test"]["work"] == {
            "password": "last-value",
            "email": "a@example.com",
        }
        assert "last-value" not in repr(item)
        await vault.delete("app.test", "work", "email")
        assert (await vault.list())[0]["fields"] == ["password"]
        await vault.delete("app.test", "work")
        assert await vault.list() == []
        assert "hidden-value" not in caplog.text and "last-value" not in caplog.text

    asyncio.run(run())


def test_save_reuses_initialized_empty_package():
    async def run():
        vault = CloudCredentials(api_key="dummy", base_url="https://example.invalid")
        fake = FakeVault()
        fake.data["ai-mobilerun-smoke-test"] = {}
        vault._client = NS(credentials=fake)
        await vault.bind(MemoryDevice())
        await vault.save("ai.mobilerun.smoke.test", "work", {"password": "dummy-value"})
        assert [
            item["name"] for item in await vault.list("ai.mobilerun.smoke.test")
        ] == ["work"]
        await vault.delete("ai.mobilerun.smoke.test", "work")
        assert await vault.list("ai.mobilerun.smoke.test") == []

    asyncio.run(run())


def test_invalid_field_rejected_before_any_write():
    async def run():
        vault = CloudCredentials(api_key="dummy", base_url="https://example.invalid")
        fake = FakeVault()
        vault._client = NS(credentials=fake)
        with pytest.raises(ValueError, match="field type"):
            await vault.save("app.test", "work", {"invalid": "secret"})
        with pytest.raises(ValueError, match="field type"):
            await vault.update("app.test", "work", "invalid", "secret")
        with pytest.raises(ValueError, match="field type"):
            await vault.delete("app.test", "work", "invalid")
        assert fake.data == {}

    asyncio.run(run())


def test_package_read_uses_package_endpoint_after_save():
    async def run():
        vault = CloudCredentials(api_key="dummy", base_url="https://example.invalid")
        fake = FakeVault()
        vault._client = NS(credentials=fake)
        await vault.bind(MemoryDevice())
        await vault.list()
        await vault.save("app.test", "work", {"password": "dummy-value"})
        original_list = fake.list

        async def inconsistent_list(package_name=None, page=None):
            if page is not None:
                return NS(items=[], pagination=NS(has_next=False))
            return await original_list(package_name=package_name)

        fake.list = inconsistent_list

        async def hidden_packages():
            return NS(data=[])

        fake.list_all = hidden_packages
        assert [item["name"] for item in await vault.list("app.test")] == ["work"]

    asyncio.run(run())
