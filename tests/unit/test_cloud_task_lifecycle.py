"""Cloud lifecycle and task safety with a fake SDK (no paid calls)."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

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
from mobilerun.errors import BillingError, CloudTaskTimeout, OperatorNotSupported
from mobilerun.models.billing import Billing
from mobilerun.models.results import (
    ExecutionResult,
    ExpectationResult,
    ExpectationVerdict,
)


class Output(BaseModel):
    ok: bool


@pytest.fixture(autouse=True)
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


def test_stream_output_and_lazy_artifacts(caplog, monkeypatch):
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


@pytest.mark.parametrize("cancel", [False, True])
def test_timeout_and_cancellation_stop_remote_task(cancel):
    async def run():
        tasks = Tasks(block=True)
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


def test_local_credentials_rejected_before_any_api(monkeypatch):
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


def test_vault_passes_only_references(monkeypatch):
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
def test_device_lifecycle_forwards(name):
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


def test_device_resume_refuses_minute_billing_under_included():
    device = module.CloudDevice("device", api_key="test", billing=Billing.INCLUDED)
    endpoints = SimpleNamespace(
        retrieve=AsyncMock(return_value=SimpleNamespace(billing_strategy="minute")),
        resume=AsyncMock(),
    )
    device._client = cast(AsyncMobilerun, SimpleNamespace(devices=endpoints))
    with pytest.raises(BillingError, match="paid"):
        asyncio.run(device.resume())
    endpoints.resume.assert_not_awaited()


def test_create_waits_ready_and_resolves_auth(monkeypatch):
    devices = SimpleNamespace(
        create=AsyncMock(return_value=SimpleNamespace(id="new", name="new")),
        wait_ready=AsyncMock(),
        retrieve_capabilities=AsyncMock(
            return_value=SimpleNamespace(capabilities=SimpleNamespace(telephony=False))
        ),
    )

    class Client:
        async def __aenter__(self):
            return SimpleNamespace(devices=devices)

        async def __aexit__(self, *args):
            pass

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


def test_expectation_retains_task_id_without_transport_context():
    task = ExecutionResult[ExpectationVerdict](
        success=True, device="CloudDevice", task_id="id"
    )
    result = ExpectationResult(
        passed=True, condition="visible", reason="yes", task=task
    )
    assert result.task.task_id == "id"
    assert not hasattr(result, "trajectory")
    assert not hasattr(task, "_cloud_client") and not hasattr(task, "_billing")


def test_real_sdk_stream_and_wire_credentials():
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


@pytest.mark.parametrize("cancel", [False, True])
def test_stop_failure_does_not_mask_original_error(cancel):
    async def run():
        tasks = Tasks(block=True)
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

    asyncio.run(run())


def test_local_results_have_no_cloud_artifacts():
    result = ExecutionResult(success=True, device="AndroidDevice")
    assert not hasattr(result, "trajectory")
    assert not hasattr(result, "screenshots")


@pytest.mark.parametrize("explicit_type", [None, "ios_phone"])
def test_real_sdk_create_and_lifecycle_wire_contract(monkeypatch, explicit_type):
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
