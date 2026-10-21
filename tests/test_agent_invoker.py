import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from mobilerun.config_manager.config_manager import MobileConfig
from mobilerun.orchestration.invoker import MobileAgentInvoker
from mobilerun.orchestration.models import TaskRequest


class FakeHandler:
    def __init__(self, result=None, *, raise_on_await: Exception | None = None):
        self._result = result or SimpleNamespace(
            success=True,
            reason="done",
            steps=3,
            structured_output=None,
        )
        self._raise_on_await = raise_on_await

    async def stream_events(self):
        yield SimpleNamespace(type="tap")
        yield SimpleNamespace(type="swipe")

    def __await__(self):
        async def done():
            if self._raise_on_await is not None:
                raise self._raise_on_await
            return self._result

        return done().__await__()


class FakeAgent:
    instances: list[dict] = []

    def __init__(self, **kwargs):
        FakeAgent.instances.append(kwargs)

    def run(self):
        return FakeHandler()


def test_run_task_maps_result_event_to_task_result():
    FakeAgent.instances = []
    request = TaskRequest(goal="open settings", id="task-1")
    config = MobileConfig()

    with (
        patch(
            "mobilerun.orchestration.invoker.ConfigLoader.load",
            return_value=config,
        ),
        patch("mobilerun.orchestration.invoker.MobileAgent", FakeAgent),
    ):
        result = asyncio.run(MobileAgentInvoker().run_task(request))

    assert result.task_id == "task-1"
    assert result.success is True
    assert result.reason == "done"
    assert result.steps == 3
    assert result.error is None
    assert result.started_at is not None
    assert result.finished_at is not None

    agent_kwargs = FakeAgent.instances[0]
    assert agent_kwargs["goal"] == "open settings"
    assert agent_kwargs["config"] is config
    assert agent_kwargs["timeout"] == 1000
    assert agent_kwargs["llms"] is None


def test_run_task_forwards_streamed_events_to_callback():
    FakeAgent.instances = []
    request = TaskRequest(goal="tap home")
    received: list[object] = []

    with (
        patch(
            "mobilerun.orchestration.invoker.ConfigLoader.load",
            return_value=MobileConfig(),
        ),
        patch("mobilerun.orchestration.invoker.MobileAgent", FakeAgent),
    ):
        asyncio.run(
            MobileAgentInvoker().run_task(request, event_callback=received.append)
        )

    assert len(received) == 2
    assert received[0].type == "tap"
    assert received[1].type == "swipe"


def test_run_task_without_callback_does_not_crash():
    FakeAgent.instances = []
    request = TaskRequest(goal="tap home")

    with (
        patch(
            "mobilerun.orchestration.invoker.ConfigLoader.load",
            return_value=MobileConfig(),
        ),
        patch("mobilerun.orchestration.invoker.MobileAgent", FakeAgent),
    ):
        result = asyncio.run(MobileAgentInvoker().run_task(request, event_callback=None))

    assert result.success is True


def test_run_task_applies_device_serial_override():
    FakeAgent.instances = []
    request = TaskRequest(goal="check wifi", device_serial="emulator-5554")
    config = MobileConfig()

    with (
        patch(
            "mobilerun.orchestration.invoker.ConfigLoader.load",
            return_value=config,
        ),
        patch("mobilerun.orchestration.invoker.MobileAgent", FakeAgent),
    ):
        asyncio.run(MobileAgentInvoker().run_task(request))

    assert config.device.serial == "emulator-5554"


def test_run_task_uses_injected_config_without_loading():
    FakeAgent.instances = []
    request = TaskRequest(goal="check wifi", config_path="/ignored/config.yaml")
    config = MobileConfig()
    config.device.serial = "preset-device"

    with (
        patch("mobilerun.orchestration.invoker.ConfigLoader.load") as load_config,
        patch("mobilerun.orchestration.invoker.MobileAgent", FakeAgent),
    ):
        asyncio.run(MobileAgentInvoker(config=config).run_task(request))

    load_config.assert_not_called()
    assert FakeAgent.instances[0]["config"] is config


def test_run_task_loads_llm_when_provider_set():
    FakeAgent.instances = []
    request = TaskRequest(
        goal="summarize screen",
        llm_provider="OpenAI",
        llm_model="gpt-5.1",
    )
    fake_llm = object()

    with (
        patch(
            "mobilerun.orchestration.invoker.ConfigLoader.load",
            return_value=MobileConfig(),
        ),
        patch("mobilerun.orchestration.invoker.load_llm", return_value=fake_llm) as load_llm,
        patch("mobilerun.orchestration.invoker.MobileAgent", FakeAgent),
    ):
        asyncio.run(MobileAgentInvoker().run_task(request))

    load_llm.assert_called_once_with("OpenAI", model="gpt-5.1")
    assert FakeAgent.instances[0]["llms"] is fake_llm


def test_run_task_returns_failed_result_when_agent_raises():
    FakeAgent.instances = []

    class RaisingFakeAgent:
        def __init__(self, **kwargs):
            FakeAgent.instances.append(kwargs)

        def run(self):
            return FakeHandler(raise_on_await=RuntimeError("boom"))

    request = TaskRequest(goal="fail task", id="task-fail")

    with (
        patch(
            "mobilerun.orchestration.invoker.ConfigLoader.load",
            return_value=MobileConfig(),
        ),
        patch("mobilerun.orchestration.invoker.MobileAgent", RaisingFakeAgent),
    ):
        result = asyncio.run(MobileAgentInvoker().run_task(request))

    assert result.task_id == "task-fail"
    assert result.success is False
    assert result.reason == "agent raised"
    assert result.steps == 0
    assert result.error == "boom"
    assert result.started_at is not None
    assert result.finished_at is not None


def test_run_task_returns_failed_result_when_config_load_raises():
    request = TaskRequest(goal="bad config", id="task-config")

    with patch(
        "mobilerun.orchestration.invoker.ConfigLoader.load",
        side_effect=FileNotFoundError("missing config"),
    ):
        result = asyncio.run(MobileAgentInvoker().run_task(request))

    assert result.success is False
    assert result.error == "missing config"
