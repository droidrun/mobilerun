"""Operator selection, cloud task execution and structured expectations."""

import ast
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from mobilerun_sdk import NotFoundError, UnprocessableEntityError
from pydantic import BaseModel

import mobilerun.core.operator.cloud as cloud_module
from mobilerun.clients.cloud import tasks as cloud_tasks
from mobilerun.clients.cloud.tasks import run_task
from mobilerun.core.devices import (
    AndroidDevice,
    CloudDevice,
    Device,
    DevicePool,
    IOSDevice,
    RecordingDevice,
    RemoteDevice,
    StealthDevice,
)
from mobilerun.core.operator import Operator
from mobilerun.core.operator.cloud import CloudTaskOperator
from mobilerun.errors import (
    CloudDeviceNotFound,
    CloudResourceNotFound,
    CloudTaskTimeout,
    CloudValidationError,
    ExpectationFailed,
    ExpectationInconclusive,
    InvalidCloudDeviceId,
    OperatorNotAvailable,
    OperatorNotSupported,
)
from mobilerun.models.results import (
    ExecutionResult,
    ExpectationResult,
    ExpectationVerdict,
)
from mobilerun.utils.prompts import PromptLoader, load_prompt_text
from tests.support.paths import REPO_ROOT


def test_operator_default_check_is_noop():
    assert Operator().check(Device()) is None


def test_device_initializes_extension_registry():

    device = Device()
    assert not device.extensions
    assert not hasattr(device, "_mailbox")
    assert not hasattr(device, "_placeholder_resources")


def test_unwrap_device_stops_before_non_device_driver():
    from mobilerun.core.devices.base import _unwrap_device
    from mobilerun.core.devices.components.wrapper import DeviceWrapper

    class WithInner(Device):
        inner: object

    inner = WithInner()
    inner._inner = object()
    outer = DeviceWrapper(inner)
    assert _unwrap_device(outer) is inner


def test_device_constructor_checks_operator():

    class RejectingOperator(Operator):
        def check(self, device):
            raise ValueError("invalid pairing")

    with pytest.raises(ValueError, match="invalid pairing"):
        Device(operator=RejectingOperator())


def test_device_rejects_operator_without_check():

    with pytest.raises(AttributeError, match="check"):
        Device(operator=cast(Operator, object()))


def test_android_requires_operator():

    device = AndroidDevice(serial="test", portal_mode="disabled")
    with pytest.raises(OperatorNotAvailable):
        asyncio.run(device.do("hello"))


def test_operator_precedence():

    class Op(Operator):
        def __init__(self, value):
            self.value = value

        async def execute(
            self, device, instruction, *, output=None, max_steps=None, timeout=None
        ):
            return self.value

    class DeviceWithDefault(Device):
        def _default_operator(self):
            return Op("default")

    device = DeviceWithDefault(operator=Op("attached"))
    assert asyncio.run(device.do("hello", operator=Op("call"))) == "call"
    assert asyncio.run(device.do("hello")) == "attached"
    assert asyncio.run(DeviceWithDefault().do("hello")) == "default"


def test_operator_base_raises():

    with pytest.raises(
        OperatorNotAvailable, match="from mobilerun_agents import AgentOperator"
    ):
        asyncio.run(Operator().execute(cast(Device, object()), "hello"))


def test_android_expect_uses_attached_operator():

    class FakeOperator(Operator):
        async def execute(
            self, device, instruction, *, output=None, max_steps=None, timeout=None
        ):
            assert output is ExpectationVerdict
            return ExecutionResult[ExpectationVerdict](
                success=True,
                output=ExpectationVerdict(passed=True, reason="yes"),
                device="AndroidDevice",
            )

    device = AndroidDevice(
        serial="test", portal_mode="disabled", operator=FakeOperator()
    )
    assert asyncio.run(device.expect("visible")).passed


def test_cloud_default_operator():

    assert isinstance(
        CloudDevice("id", api_key="test")._default_operator(), CloudTaskOperator
    )


def test_core_has_no_agent_imports():

    for path in (REPO_ROOT / "mobilerun" / "core").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    ("mobilerun.agents", "mobilerun_agents")
                ), path
            elif isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith(("mobilerun.agents", "mobilerun_agents"))
                    for alias in node.names
                ), path


def test_constructor_and_attachment_reject_local():

    with pytest.raises(OperatorNotSupported, match="is not a cloud device"):
        AndroidDevice("emulator-5554", operator=CloudTaskOperator())
    assert "attach_operator" not in dir(AndroidDevice)


def test_per_call_and_pool_validate():

    device = AndroidDevice("emulator-5554")
    with pytest.raises(OperatorNotSupported):
        asyncio.run(device.do("noop", operator=CloudTaskOperator()))
    with pytest.raises(OperatorNotSupported):
        asyncio.run(device.expect("noop", operator=CloudTaskOperator()))
    device._operator = CloudTaskOperator()
    with pytest.raises(OperatorNotSupported):
        DevicePool([device])


def test_cloud_pairing_works():
    device = CloudDevice("id", api_key="test", operator=CloudTaskOperator())
    assert device._cloud_client is not None
    DevicePool([device])


@pytest.mark.parametrize("wrapper", [RecordingDevice, StealthDevice])
def test_cloud_capability_delegates_through_wrappers(wrapper):
    local = AndroidDevice("serial")
    cloud = CloudDevice("id", api_key="test")
    assert local._is_cloud is False
    assert cloud._is_cloud is True
    assert wrapper(local)._is_cloud is False
    assert wrapper(cloud)._is_cloud is True
    CloudTaskOperator().check(wrapper(cloud))


def test_wrappers_validate_inner_device():

    for wrapper in (RecordingDevice, StealthDevice):
        with pytest.raises(OperatorNotSupported):
            wrapper(AndroidDevice("emulator-5554"), operator=CloudTaskOperator())
        device = wrapper(AndroidDevice("emulator-5554"))
        expected = rf"{wrapper.__name__}\(AndroidDevice\)"
        with pytest.raises(OperatorNotSupported, match=expected):
            asyncio.run(device.do("noop", operator=CloudTaskOperator()))
        device._inner._operator = CloudTaskOperator()
        device._operator = device._inner._operator
        with pytest.raises(OperatorNotSupported, match=expected):
            DevicePool([device])


@pytest.mark.parametrize(
    "status,body,expected",
    [
        pytest.param(
            404,
            {"detail": "Device 123 not found"},
            CloudDeviceNotFound,
            id="missing-device",
        ),
        pytest.param(
            404,
            {"detail": "Model not found"},
            CloudResourceNotFound,
            id="missing-model",
        ),
        pytest.param(
            422,
            {"detail": [{"type": "uuid_parsing", "loc": ["body", "deviceId"]}]},
            InvalidCloudDeviceId,
            id="invalid-device-id",
        ),
        pytest.param(
            422,
            {"detail": [{"type": "other", "loc": ["body", "task"]}]},
            CloudValidationError,
            id="invalid-task",
        ),
    ],
)
def test_cloud_task_classifies_sdk_errors(status, body, expected):
    sdk_error = NotFoundError if status == 404 else UnprocessableEntityError

    async def run(**kwargs):
        raise sdk_error(
            "failure",
            response=httpx.Response(
                status, request=httpx.Request("POST", "https://example.invalid")
            ),
            body=body,
        )

    with pytest.raises(expected, match=f"HTTP {status}") as caught:
        asyncio.run(
            run_task(
                SimpleNamespace(tasks=SimpleNamespace(run=run)),
                device_id="123",
                task="noop",
                billing="included",
            )
        )
    # The device-specific errors subclass the generic errors for the same status.
    assert type(caught.value) is expected


class ExampleOutput(BaseModel):
    ok: bool


class StreamedTasks:
    @property
    def with_streaming_response(self):
        return self

    @asynccontextmanager
    async def attach(self, task_id, **kwargs):
        yield self

    async def iter_lines(self):
        yield "event: progress"

    async def stop(self, task_id):
        pass


@pytest.mark.parametrize(
    "device",
    [
        IOSDevice("http://localhost:8080"),
        RemoteDevice(url="http://localhost:8090", device_id="test"),
    ],
)
@pytest.mark.parametrize("method", ["do", "expect"])
def test_unsupported_devices_raise(device, method):
    with pytest.raises(OperatorNotAvailable, match="No operator"):
        asyncio.run(getattr(device, method)("test"))


def test_cloud_do_maps_client_result(monkeypatch):
    device = CloudDevice("device", api_key="test-key")
    outcome = ExecutionResult[ExampleOutput](
        success=True,
        task_id="task-x",
        output=ExampleOutput(ok=True),
        device="CloudDevice",
    )

    async def fake_run(client, **kwargs):
        assert client is device._client
        assert kwargs == {
            "billing": "paid",
            "device_id": "device",
            "task": "instruction",
            "output": ExampleOutput,
            "max_steps": 6,
        }
        return outcome

    monkeypatch.setattr(cloud_module, "run_task", fake_run)
    result = asyncio.run(device.do("instruction", output=ExampleOutput, max_steps=6))
    assert isinstance(result, ExecutionResult)
    assert (result.task_id, result.success, result.output.ok, result.device) == (
        "task-x",
        True,
        True,
        "CloudDevice",
    )


def test_client_run_task_polls_and_validates_output(monkeypatch):
    calls = []
    statuses = iter(
        [
            SimpleNamespace(status="running"),
            SimpleNamespace(
                status="completed",
                succeeded=True,
                message="done",
                output={"ok": True},
                steps=4,
            ),
        ]
    )

    class Tasks(StreamedTasks):
        async def run(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(id="task-z")

        async def get_status(self, task_id):
            assert task_id == "task-z"
            return next(statuses)

    async def no_sleep(_):
        return None

    monkeypatch.setattr(cloud_tasks.asyncio, "sleep", no_sleep)
    result = asyncio.run(
        run_task(
            SimpleNamespace(tasks=Tasks()),
            device_id="d",
            billing="included",
            task="t",
            output=ExampleOutput,
            max_steps=3,
            timeout=4,
            poll_interval=0,
        )
    )
    assert calls == [
        {
            "device_id": "d",
            "task": "t",
            "output_schema": ExampleOutput.model_json_schema(),
            "max_steps": 3,
        }
    ]
    assert isinstance(result, ExecutionResult)
    assert (result.task_id, result.success, result.output.ok, result.steps) == (
        "task-z",
        True,
        True,
        4,
    )


def test_client_run_task_timeout_is_typed():
    class Tasks(StreamedTasks):
        async def run(self, **kwargs):
            return SimpleNamespace(id="task-timeout")

        async def get_status(self, task_id):
            return SimpleNamespace(status="running")

    with pytest.raises(CloudTaskTimeout, match="task-timeout.*did not finish"):
        asyncio.run(
            run_task(
                SimpleNamespace(tasks=Tasks()),
                device_id="d",
                billing="included",
                task="t",
                timeout=0,
                poll_interval=0,
            )
        )


@pytest.mark.parametrize("slow_phase", ["submit", "status"])
def test_client_run_task_timeout_bounds_submit_and_status(slow_phase):
    class Tasks(StreamedTasks):
        async def run(self, **kwargs):
            if slow_phase == "submit":
                await asyncio.sleep(0.05)
            return SimpleNamespace(id="slow-task")

        async def get_status(self, task_id):
            if slow_phase == "status":
                await asyncio.sleep(0.05)
            return SimpleNamespace(
                status="completed", succeeded=True, message="done", steps=1
            )

    with pytest.raises(CloudTaskTimeout):
        asyncio.run(
            run_task(
                SimpleNamespace(tasks=Tasks()),
                device_id="d",
                billing="included",
                task="t",
                timeout=0.01,
                poll_interval=0,
            )
        )


def test_cloud_invalid_output_returns_failed_typed_result(monkeypatch):
    class Tasks(StreamedTasks):
        async def run(self, **kwargs):
            return SimpleNamespace(id="task-invalid")

        async def get_status(self, task_id):
            return SimpleNamespace(
                status="completed",
                succeeded=True,
                message="done",
                output={"ok": "not-a-bool"},
                steps=1,
            )

    result = asyncio.run(
        run_task(
            SimpleNamespace(tasks=Tasks()),
            device_id="d",
            billing="included",
            task="t",
            output=ExampleOutput,
            timeout=1,
            poll_interval=0,
        )
    )
    assert isinstance(result, ExecutionResult)
    assert result.success is False and result.output is None
    assert "validation error" in result.answer


@pytest.mark.parametrize("passed", [True, False])
def test_expect_renders_prompt_and_returns_or_raises(monkeypatch, passed):
    assert "a blue button exists" in PromptLoader.render_template(
        load_prompt_text("devices/expect.jinja2"), {"condition": "a blue button exists"}
    )
    device = object.__new__(AndroidDevice)
    task = ExecutionResult[ExpectationVerdict](
        success=True,
        output=ExpectationVerdict(passed=passed, reason="not visible"),
        device="AndroidDevice",
    )

    async def fake_do(instruction, *, output=None, max_steps=None):
        assert "a blue button exists" in instruction
        assert output is ExpectationVerdict
        return task

    monkeypatch.setattr(device, "do", fake_do, raising=False)
    if passed:
        result = asyncio.run(device.expect("a blue button exists"))
        assert isinstance(result, ExpectationResult) and result.passed
    else:
        with pytest.raises(ExpectationFailed, match="not visible") as error:
            asyncio.run(device.expect("a blue button exists"))
        assert error.value.result.reason == "not visible"


@pytest.mark.parametrize(
    ("success", "answer"), [(True, "passed=true text"), (False, "task failed")]
)
def test_expect_without_verdict_is_inconclusive(monkeypatch, success, answer):
    device = object.__new__(AndroidDevice)
    task = ExecutionResult[ExpectationVerdict](
        success=success, answer=answer, output=None, device="AndroidDevice"
    )

    async def fake_do(*args, **kwargs):
        return task

    monkeypatch.setattr(device, "do", fake_do, raising=False)
    with pytest.raises(ExpectationInconclusive) as error:
        asyncio.run(device.expect("Settings is open"))
    assert error.value.condition == "Settings is open"
    assert error.value.task is task
    assert "no structured verdict" in str(error.value)
    assert answer not in str(error.value)
