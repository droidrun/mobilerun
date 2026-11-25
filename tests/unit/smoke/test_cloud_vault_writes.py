"""Real vault round trip; no device or task credits required."""

import asyncio
import uuid

import pytest


@pytest.mark.smoke_cloud
def test_cloud_vault_write_and_cleanup(request, record_property):
    request.getfixturevalue("real_env")
    from mobilerun.core.credentials import CloudCredentials

    async def run():
        vault = CloudCredentials()
        package = "ai.mobilerun.smoke.test"
        name = f"smoke-{uuid.uuid4().hex}"
        record_property("created_vault_credential", f"{package}/{name}")
        try:
            await vault.save(package, name, {"password": "smoke-only-value"})
            items = await vault.list(package)
            assert any(
                item["name"] == name
                and item["package"] == package
                and set(item["fields"]) == {"password"}
                for item in items
            )
            assert any(item["name"] == name for item in await vault.list())
            await vault.save(
                package,
                name,
                {"password": "smoke-updated-value", "username": "smoke-user"},
            )
            await vault.update(package, name, "password", "smoke-final-value")
            item = await vault._find(package, name=name)
            assert item is not None
            assert set(item.fields) == {"password", "username"}
            assert item.fields["password"] == "smoke-final-value"
            await vault.delete(package, name, "username")
            assert any(
                item["name"] == name and set(item["fields"]) == {"password"}
                for item in await vault.list(package)
            )
        finally:
            try:
                await vault.delete(package, name)
                assert not any(
                    item["name"] == name for item in await vault.list(package)
                )
                record_property("deleted_vault_credential", f"{package}/{name}")
            finally:
                await vault.close()

    asyncio.run(run())
