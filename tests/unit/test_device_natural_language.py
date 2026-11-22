import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import mobilerun.core.operator.cloud as cloud_module
from mobilerun.clients.cloud import tasks as cloud_tasks
from mobilerun.clients.cloud.tasks import run_task
from mobilerun.core.devices import (
    AndroidDevice,
    CloudDevice,
    IOSDevice,
    RemoteDevice,
)
from mobilerun.errors import (
    CloudTaskTimeout,
    ExpectationFailed,
    ExpectationInconclusive,
    OperatorNotAvailable,
)
from mobilerun.models.results import (
    ExecutionResult,
    ExpectationResult,
    ExpectationVerdict,
)
from mobilerun.config.prompts import PromptLoader, load_prompt_text


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
