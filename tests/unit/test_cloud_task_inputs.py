"""Cloud task contributions contain references, never credential values."""

import asyncio
import json
from dataclasses import FrozenInstanceError, asdict, dataclass
from unittest.mock import AsyncMock

import pytest

from mobilerun.core.extensions import Extension, action
from mobilerun.core.proxy import CloudProxy
from mobilerun.errors import ExtensionActionError
from tests.support.devices import MemoryDevice


def test_default_cloud_task_contract_and_reference_only_models():
    from mobilerun.models import CloudCredentialReference, CloudTaskInputs

    assert Extension.cloud_native is False
    assert CloudProxy.cloud_native is True
    assert asyncio.run(Extension().cloud_task_inputs()) == CloudTaskInputs()
    inputs = CloudTaskInputs((CloudCredentialReference("app", "login"),))
    assert json.loads(json.dumps(asdict(inputs))) == {
        "credential_references": [{"package": "app", "name": "login"}]
    }
    with pytest.raises(FrozenInstanceError):
        inputs.credential_references = ()
    with pytest.raises(TypeError):
        CloudCredentialReference("app", "login", values={"password": "secret"})
    with pytest.raises(TypeError):
        CloudCredentialReference("app", {"password": "secret"})


def test_cloud_operator_merges_only_opted_in_extensions(monkeypatch):
    from mobilerun.core.devices import CloudDevice
    from mobilerun.core.operator import cloud
    from mobilerun.errors import OperatorNotSupported
    from mobilerun.models import CloudCredentialReference, CloudTaskInputs
    from mobilerun.models.results import ExecutionResult

    class Contribution(Extension):
        cloud_native = True

        async def cloud_task_inputs(self) -> CloudTaskInputs:
            return CloudTaskInputs(
                (
                    CloudCredentialReference("app", self.name),
                    CloudCredentialReference("app", "shared"),
                )
            )

    class LocalOnly(Extension):
        async def cloud_task_inputs(self) -> CloudTaskInputs:
            raise AssertionError("non-native extensions must not contribute")

    run = AsyncMock(return_value=ExecutionResult(success=True, device="cloud"))
    monkeypatch.setattr(cloud, "run_task", run)

    async def scenario():
        phone = CloudDevice("phone", api_key="fake")
        phone._ensure_connected = AsyncMock()
        await Contribution(name="first").bind(phone)
        await Contribution(name="second").bind(phone)
        local = await LocalOnly(name="local").bind(phone)
        with pytest.raises(OperatorNotSupported, match="local extensions"):
            await cloud.CloudTaskOperator().execute(phone, "sign in")
        run.assert_not_awaited()
        await local.unbind()
        await cloud.CloudTaskOperator().execute(phone, "sign in")
        assert run.await_args.kwargs["credentials"] == [
            {"package_name": "app", "credential_names": ["first", "second", "shared"]}
        ]
        await phone.close()

    asyncio.run(scenario())


def test_action_errors_stay_wrapped_without_backend_details():
    class Failing(Extension):
        @action()
        async def fail(self) -> None:
            """Demonstrate safe action error conversion."""
            raise RuntimeError("private backend details")

    async def scenario():
        phone = MemoryDevice()
        extension = Failing()
        await extension.bind(phone)
        for call in (extension.fail, lambda: phone.actions.call("extension.fail", {})):
            with pytest.raises(ExtensionActionError) as raised:
                await call()
            assert "private backend details" not in str(raised.value)
        await phone.close()

    asyncio.run(scenario())


def test_contract_kit_rejects_extra_value_fields_in_task_inputs():
    from mobilerun.models import CloudTaskInputs
    from tests.support.extensions import check_extension_contract

    @dataclass(frozen=True)
    class UnsafeInputs(CloudTaskInputs):
        values: str = "must-not-be-published"

    class Unsafe(Extension):
        async def cloud_task_inputs(self) -> CloudTaskInputs:
            return UnsafeInputs()

    async def device():
        return MemoryDevice()

    with pytest.raises(AssertionError, match="G:.*references only"):
        asyncio.run(check_extension_contract(lambda name: Unsafe(name=name), device))
