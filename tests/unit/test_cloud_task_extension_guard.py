import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from mobilerun.core.credentials import CloudCredentials, DictCredentials
from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.core.extensions import Extension
from mobilerun.core.mailbox import CloudMailbox
from mobilerun.core.operator.cloud import CloudTaskOperator
from mobilerun.core.phone_number import CloudPhoneNumber
from mobilerun.errors import OperatorNotSupported
from mobilerun.models import CloudCredentialReference
from mobilerun.models.results import ExecutionResult


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DictCredentials({}),
        lambda: CloudMailbox("m"),
        lambda: Extension(name="custom"),
    ],
)
def test_cloud_task_rejects_local_extensions(factory):
    async def run():
        phone = CloudDevice("phone", api_key="fake")
        extension = factory()
        await extension.bind(phone)
        operator = CloudTaskOperator()
        with pytest.raises(
            OperatorNotSupported, match="local extensions.*mobilerun-agents.*unbind"
        ):
            operator.check(phone)
        with patch("mobilerun.core.operator.cloud.run_task", AsyncMock()) as execute:
            with pytest.raises(OperatorNotSupported):
                await operator.execute(phone, "task")
            execute.assert_not_awaited()

    asyncio.run(run())


def test_cloud_task_allows_cloud_vault_and_phone_number():
    async def run():
        phone = CloudDevice("phone", api_key="fake")
        phone._cloud_telephony = True
        vault = CloudCredentials(api_key="fake")
        number = CloudPhoneNumber("number")
        secret = "cloud-vault-canary"
        items = [
            SimpleNamespace(
                package_name="app",
                credential_name="work",
                fields=[SimpleNamespace(field_type="password", value=secret)],
            )
        ]
        with (
            patch(
                "mobilerun.core.credentials.cloud.secrets.list_secrets",
                AsyncMock(return_value=items),
            ),
            patch("mobilerun.core.phone_number.cloud.devices.bind_number", AsyncMock()),
            patch(
                "mobilerun.core.operator.cloud.run_task",
                AsyncMock(return_value=ExecutionResult(success=True, device="cloud")),
            ) as execute,
        ):
            await vault.bind(phone)
            await number.bind(phone)
            inputs = await vault.cloud_task_inputs()
            assert inputs.credential_references == (
                CloudCredentialReference("app", "work"),
            )
            assert secret not in repr(inputs)
            operator = CloudTaskOperator()
            operator.check(phone)
            await operator.execute(phone, "task")
            assert execute.await_args.kwargs["credentials"] == [
                {"package_name": "app", "credential_names": ["work"]}
            ]
            assert secret not in repr(execute.await_args)

    asyncio.run(run())
