"""Cloud task inputs, execution lifecycle, and offline SDK contracts."""

import ast
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError, asdict, dataclass
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from mobilerun_sdk import AsyncMobilerun
from mobilerun_sdk.types import DeviceCreateResponse
from pydantic import BaseModel

import mobilerun.core.devices.cloud as module
import mobilerun.core.operator.cloud as cloud_operator
from mobilerun.clients.cloud import tasks as task_client
from mobilerun.clients.cloud.tasks import run_task
from mobilerun.core.credentials import CloudCredentials, Credential, DictCredentials
from mobilerun.core.devices.cloud import CloudDevice
from mobilerun.core.extensions import Extension, action
from mobilerun.core.mailbox import CloudMailbox
from mobilerun.core.operator.cloud import CloudTaskOperator
from mobilerun.core.phone_number import CloudPhoneNumber
from mobilerun.core.proxy import CloudProxy
from mobilerun.errors import (
    BillingError,
    CloudTaskTimeout,
    ExtensionActionError,
    OperatorNotSupported,
)
from mobilerun.models import CloudCredentialReference
from mobilerun.models.billing import Billing
from mobilerun.models.results import (
    ExecutionResult,
    ExpectationResult,
    ExpectationVerdict,
)
from tests.support.devices import MemoryDevice
from tests.support.paths import REPO_ROOT


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


class Output(BaseModel):
    ok: bool


@pytest.fixture
def task_billing(monkeypatch):
    monkeypatch.setenv("MOBILERUN_BILLING", "included")


class Tasks:
    def __init__(self, *, block=False):
        self.block = block
        self.started = asyncio.Event()
        self.stop = AsyncMock()
        self.run = AsyncMock(return_value=SimpleNamespace(id="task-1"))
        self.get_status = AsyncMock(
            return_value=SimpleNamespace(
                status="completed",
                succeeded=True,
                message="done",
                output={"ok": True},
                steps=2,
            )
        )
        self.get_trajectory = AsyncMock(return_value="trajectory")
        self.screenshots = SimpleNamespace(list=AsyncMock(return_value=["image"]))
        self.with_streaming_response = self

    @asynccontextmanager
    async def attach(self, task_id, **params):
        assert task_id == "task-1"
        try:
            yield self
        finally:
            self.closed = True

    async def iter_lines(self):
        self.started.set()
        yield "event: progress"
        yield 'data: {"secret": "do-not-log"}'
        if self.block:
            await asyncio.Event().wait()


def test_stream_output_and_lazy_artifacts(caplog, monkeypatch, task_billing):
    monkeypatch.setattr(logging.getLogger("mobilerun"), "propagate", True)

    async def run():
        tasks = Tasks()
        with caplog.at_level("INFO", logger="mobilerun.actions"):
            result = await run_task(
                cast(AsyncMobilerun, SimpleNamespace(tasks=tasks)),
                device_id="device",
                billing=Billing.INCLUDED,
                task="go",
                output=Output,
            )
        assert result.output is not None
        assert result.task_id is not None
        assert result.output.ok and tasks.closed
        client = cast(AsyncMobilerun, SimpleNamespace(tasks=tasks))
        assert (
            await task_client.trajectory(
                client, result.task_id, billing=Billing.INCLUDED
            )
            == "trajectory"
        )
        assert await task_client.screenshots(
            client, result.task_id, billing=Billing.INCLUDED
        ) == ["image"]
        tasks.run.assert_awaited_once_with(
            device_id="device", task="go", output_schema=Output.model_json_schema()
        )
        assert not caplog.records
        assert "_cloud_client" not in result.model_dump()

    asyncio.run(run())


@pytest.mark.parametrize("stop_fails", [False, True], ids=["stop-ok", "stop-error"])
@pytest.mark.parametrize("cancel", [False, True])
def test_timeout_and_cancellation_stop_remote_task(cancel, stop_fails, task_billing):
    async def run():
        tasks = Tasks(block=True)
        if stop_fails:
            tasks.stop.side_effect = RuntimeError("stop unavailable")
        pending = asyncio.create_task(
            run_task(
                cast(AsyncMobilerun, SimpleNamespace(tasks=tasks)),
                device_id="device",
                billing=Billing.INCLUDED,
                task="go",
                timeout=10 if cancel else 0.02,
            )
        )
        await tasks.started.wait()
        if cancel:
            pending.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else CloudTaskTimeout):
            await pending
        tasks.stop.assert_awaited_once_with("task-1")
        assert tasks.closed

    asyncio.run(run())


def test_local_credentials_rejected_before_any_api(monkeypatch, task_billing):
    vault = CloudCredentials(api_key="test")
    vault._list_credentials = AsyncMock(
        side_effect=AssertionError("must not read vault")
    )
    device = module.CloudDevice(
        "device",
        api_key="test",
    )
    asyncio.run(vault.bind(device))
    asyncio.run(DictCredentials({"secret": "value"}, name="local").bind(device))
    with pytest.raises(OperatorNotSupported, match="local extensions"):
        asyncio.run(device.do("go"))
    vault._list_credentials.assert_not_awaited()


def test_vault_passes_only_references(monkeypatch, task_billing):
    vault = CloudCredentials(api_key="test")
    vault._list_credentials = AsyncMock(
        return_value=[Credential("com.example", "login", {"password": "never-send"})]
    )
    device = module.CloudDevice("device", api_key="test")
    asyncio.run(vault.bind(device))
    fake_run = AsyncMock(
        return_value=ExecutionResult(success=True, device="CloudDevice")
    )
    monkeypatch.setattr(cloud_operator, "run_task", fake_run)
    asyncio.run(device.do("go"))
    assert fake_run.call_args.kwargs["credentials"] == [
        {"package_name": "com.example", "credential_names": ["login"]}
    ]
    assert "never-send" not in repr(fake_run.call_args)


@pytest.mark.parametrize(
    "name", ["resume", "stop", "terminate", "wait_ready", "reboot", "reset", "info"]
)
def test_device_lifecycle_forwards(name, task_billing):
    device = module.CloudDevice("device", api_key="test")
    response = SimpleNamespace(name="response") if name == "info" else "response"
    method = AsyncMock(return_value=response)
    endpoints = SimpleNamespace(**{"retrieve" if name == "info" else name: method})
    if name == "resume":
        endpoints.retrieve = AsyncMock(
            return_value=SimpleNamespace(billing_strategy="subscription")
        )
    device._client = cast(AsyncMobilerun, SimpleNamespace(devices=endpoints))
    assert asyncio.run(getattr(device, name)()) == response
    method.assert_awaited_once_with("device")


def test_device_resume_refuses_minute_billing_under_included(task_billing):
    device = module.CloudDevice("device", api_key="test", billing=Billing.INCLUDED)
    endpoints = SimpleNamespace(
        retrieve=AsyncMock(return_value=SimpleNamespace(billing_strategy="minute")),
        resume=AsyncMock(),
    )
    device._client = cast(AsyncMobilerun, SimpleNamespace(devices=endpoints))
    with pytest.raises(BillingError, match="paid"):
        asyncio.run(device.resume())
    endpoints.resume.assert_not_awaited()


def test_create_waits_ready_and_resolves_auth(monkeypatch, task_billing):
    devices = SimpleNamespace(
        create=AsyncMock(return_value=SimpleNamespace(id="new", name="new")),
        wait_ready=AsyncMock(),
        retrieve_capabilities=AsyncMock(
            return_value=SimpleNamespace(
                capabilities=SimpleNamespace(telephony=False, proxy=True)
            )
        ),
    )

    monkeypatch.setattr(module, "resolve_cloud_api_key", lambda: "test")
    from mobilerun_core_local.driver import cloud as driver

    client = SimpleNamespace(devices=devices)
    client.with_options = lambda **kwargs: client
    monkeypatch.setattr(driver, "AsyncMobilerun", lambda **params: client)

    async def initialize():
        return await module.CloudDevice(name="example", billing="paid")

    result = asyncio.run(initialize())
    assert result.device_id == "new"
    devices.create.assert_awaited_once_with(
        name="example",
        proxy={"connect": {"country": "us"}, "smart_ip": True},
        billing="auto",
        device_type="android_cloud_phone",
    )
    devices.wait_ready.assert_awaited_once_with("new")


def test_expectation_retains_task_id_without_transport_context(task_billing):
    task = ExecutionResult[ExpectationVerdict](
        success=True, device="CloudDevice", task_id="id"
    )
    result = ExpectationResult(
        passed=True, condition="visible", reason="yes", task=task
    )
    assert result.task.task_id == "id"
    assert not hasattr(result, "trajectory")
    assert not hasattr(task, "_cloud_client") and not hasattr(task, "_billing")


def test_real_sdk_stream_and_wire_credentials(task_billing):
    seen = []

    def handler(request):
        seen.append(request.url.path)
        if request.url.path == "/v1/tasks":
            body = json.loads(request.content)
            assert body["outputSchema"] == Output.model_json_schema()
            assert body["credentials"] == [
                {"packageName": "com.example", "credentialNames": ["login"]}
            ]
            return httpx.Response(
                200, json={"id": "task-1", "status": "queued", "streamUrl": ""}
            )
        if request.url.path.endswith("/attach"):
            return httpx.Response(
                200,
                text="event: progress\ndata: {}\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "succeeded": True,
                "message": "done",
                "output": {"ok": True},
                "steps": 1,
            },
        )

    async def run():
        async with AsyncMobilerun(
            api_key="test",
            base_url="https://sdk.test/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            result = await run_task(
                client,
                device_id="device",
                billing=Billing.INCLUDED,
                task="go",
                output=Output,
                credentials=[
                    {"package_name": "com.example", "credential_names": ["login"]}
                ],
            )
            assert result.output is not None
            assert result.success and result.output.ok

    asyncio.run(run())
    assert seen == ["/v1/tasks", "/v1/tasks/task-1/attach", "/v1/tasks/task-1/status"]


def test_local_results_have_no_cloud_artifacts(task_billing):
    result = ExecutionResult(success=True, device="AndroidDevice")
    assert not hasattr(result, "trajectory")
    assert not hasattr(result, "screenshots")


@pytest.mark.parametrize("explicit_type", [None, "ios_phone"])
def test_real_sdk_create_and_lifecycle_wire_contract(
    monkeypatch, explicit_type, task_billing
):
    # lazy: patch the driver's SDK constructor only in this transport-only test.
    from mobilerun_core_local.driver import cloud as driver_module

    expected_type = explicit_type or "android_cloud_phone"
    payload = {
        "id": "created-device-id",
        "activeTaskId": "",
        "assignedAt": None,
        "createdAt": "2026-11-18T00:00:00Z",
        "name": "example",
        "platform": "ios" if explicit_type == "ios_phone" else "android",
        "state": "ready",
        "stateMessage": "",
        "streamUrl": "",
        "taskCount": 0,
        "terminatesAt": None,
        "type": expected_type,
        "updatedAt": "2026-11-18T00:00:00Z",
        "billingStrategy": "minute",
    }
    assert DeviceCreateResponse.model_validate(payload).id == "created-device-id"
    seen = []
    clients = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path == "/v1/devices":
            assert dict(request.url.params) == {
                "deviceType": expected_type,
                "billing": "auto",
            }
            assert json.loads(request.content) == {
                "name": "example",
                "country": "US",
                "proxy": {"connect": {"country": "us"}, "smartIp": True},
            }
            return httpx.Response(200, json=payload)
        assert request.url.path.startswith("/v1/devices/created-device-id")
        if request.url.path.endswith("/capabilities"):
            return httpx.Response(
                200,
                json={
                    "deviceType": expected_type,
                    "capabilities": {"telephony": False},
                },
            )
        if request.method == "GET":
            return httpx.Response(200, json=payload)
        return httpx.Response(204)

    def factory(**params):
        client = AsyncMobilerun(
            **params,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(driver_module, "AsyncMobilerun", factory)

    async def run():
        try:
            device = await module.CloudDevice(
                api_key="test",
                base_url="https://sdk.test/v1",
                name="example",
                billing="paid",
                country="US",
                device_type=explicit_type,
            )
            assert device.device_id == "created-device-id"
            assert (await device.info()).id == device.device_id
            await device.stop()
            await device.resume()
            await device.wait_ready()
            await device.reboot()
            await device.reset(mode="identity")
            await device.terminate()
        finally:
            for client in clients:
                await client.close()

    asyncio.run(run())
    prefix = "/v1/devices/created-device-id"
    assert seen == [
        ("POST", "/v1/devices"),
        ("GET", prefix + "/wait"),
        ("GET", prefix + "/capabilities"),
        ("GET", prefix),
        ("POST", prefix + "/stop"),
        ("POST", prefix + "/resume"),
        ("GET", prefix + "/wait"),
        ("POST", prefix + "/reboot"),
        ("POST", prefix + "/reset"),
        ("DELETE", prefix),
    ]


def test_task_boundary_does_not_probe_sdk_error_body():
    path = REPO_ROOT / "mobilerun/clients/cloud/tasks.py"
    tree = ast.parse(path.read_text())
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "body"
        for node in ast.walk(tree)
    )
