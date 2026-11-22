import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from mobilerun.core.credentials import (
    CloudCredentials,
    DictCredentials,
    EnvCredentials,
    FileCredentials,
)
from mobilerun.core.extensions import Extension
from mobilerun.errors import (
    CredentialNotFoundError,
    DeviceError,
    ExtensionActionError,
    ExtensionNotBound,
)
from tests.support.devices import MemoryDevice
from tests.support.extensions import check_extension_contract


@pytest.fixture(autouse=True)
def focused_state(monkeypatch):
    from mobilerun.core.state.view import UIStateView

    monkeypatch.setattr(
        UIStateView,
        "refresh",
        AsyncMock(
            return_value=SimpleNamespace(
                phone_state={
                    "focusedElement": {"className": "EditText"},
                    "isEditable": True,
                }
            )
        ),
    )


def test_vault_exposes_only_metadata_and_types_secret():
    asyncio.run(_check_vault())


async def _check_vault():
    secret = "canary-vault-private-value"
    vault = DictCredentials({"app": {"account": {"password": secret}}})
    device = MemoryDevice()
    assert isinstance(vault, Extension)
    await vault.bind(device)
    metadata = await vault.list()
    assert secret not in repr(metadata)
    assert secret not in repr(await vault.context())
    assert await vault.type_field("app", "password") is None
    assert device.inputs == [secret]


def test_dict_vault_extension_contract():
    async def device():
        return MemoryDevice()

    asyncio.run(
        check_extension_contract(lambda name: DictCredentials({}, name=name), device)
    )


def test_file_and_env_vault_extension_contract(tmp_path):
    path = tmp_path / "credentials.yml"
    path.write_text("{}")
    path.chmod(0o600)

    async def device():
        return MemoryDevice()

    for factory in (
        lambda name: FileCredentials(path, name=name),
        lambda name: EnvCredentials("NONEXISTENT_TEST_PREFIX_", name=name),
    ):
        asyncio.run(check_extension_contract(factory, device))


def test_cloud_vault_extension_contract():
    async def device():
        return MemoryDevice()

    with patch(
        "mobilerun.core.credentials.cloud.secrets.list_secrets",
        AsyncMock(return_value=[]),
    ):
        asyncio.run(
            check_extension_contract(
                lambda name: CloudCredentials(api_key="test", name=name), device
            )
        )


def test_type_field_sanitizes_secret_input_failure(caplog):
    async def run():
        secret = "secret-canary-must-not-escape"
        device = MemoryDevice()
        device._input_secret = AsyncMock(side_effect=RuntimeError(secret))
        vault = DictCredentials({"app": {"one": {"password": secret}}})
        await vault.bind(device)
        with pytest.raises(ExtensionActionError) as exc:
            await vault.type_field("app", "password")
        assert secret not in str(exc.value)
        assert secret not in caplog.text
        device._input_secret.assert_awaited_once_with(secret)

    asyncio.run(run())


def test_type_field_reports_unsuccessful_input():
    async def run():
        device = MemoryDevice()
        device._input_secret = AsyncMock(return_value=False)
        vault = DictCredentials({"app": {"one": {"password": "secret"}}})
        await vault.bind(device)
        with pytest.raises(ExtensionActionError) as error:
            await vault.type_field("app", "password")
        assert isinstance(error.value, DeviceError)
        assert "secret" not in str(error.value)

    asyncio.run(run())


def test_type_field_uses_typed_missing_and_unbound_errors():
    async def run():
        vault = DictCredentials({})
        with pytest.raises(ExtensionNotBound):
            await vault.type_field("app", "password")
        await vault.bind(MemoryDevice())
        with pytest.raises(CredentialNotFoundError) as error:
            await vault.type_field("app", "password")
        from mobilerun.errors import CredentialFieldNotFound

        assert isinstance(error.value, CredentialFieldNotFound)
        assert not isinstance(error.value, ExtensionActionError)
        assert "password-value-canary" not in str(error.value)

    asyncio.run(run())


def test_type_field_device_action_rejects_unbound_before_lookup():
    async def run():
        vault = DictCredentials({})
        vault._find = AsyncMock()
        assert vault.actions.get("vault.type_field").requires_device
        with pytest.raises(ExtensionNotBound):
            await vault.type_field("app", "password")
        vault._find.assert_not_awaited()

    asyncio.run(run())
