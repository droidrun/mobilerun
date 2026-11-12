"""Exercise finalization through the real workflow timer and event stream."""

import asyncio
from types import SimpleNamespace

import pytest
from llama_index.core.workflow import Context, StartEvent, Workflow, step
from workflows.errors import WorkflowCancelledByUser, WorkflowTimeoutError

import mobilerun.agent.droid.droid_agent as module
from mobilerun.agent.droid.droid_agent import MobileAgent
from mobilerun.agent.droid.events import (
    FastAgentExecuteEvent,
    FinalizeEvent,
    ManagerInputEvent,
    ResultEvent,
)
from mobilerun.agent.trajectory import TrajectoryWriter
from mobilerun.tools.driver.secret_safe import SecretRedactor


class FinishingAgent(MobileAgent):
    """Replace device/LLM setup, retaining MobileAgent.run and finalize."""

    def __init__(self, timeout, num_concurrent_runs=None):
        Workflow.__init__(
            self, timeout=timeout, num_concurrent_runs=num_concurrent_runs
        )
        self.timeout = timeout
        self._redactor = SecretRedactor()

    @step
    async def start_handler(
        self, ctx: Context, ev: StartEvent
    ) -> FastAgentExecuteEvent | ManagerInputEvent | FinalizeEvent:
        await self._initialize_workflow_deadline(ctx)
        await self._initialize_run_trajectory()
        started = ev.get("started", default=None)
        if started is not None:
            started.set()
        await asyncio.sleep(ev.delay)
        return FinalizeEvent(
            success=ev.success,
            reason=ev.get("reason", default="decided"),
        )


def _agent(monkeypatch, timeout, screenshot, *, num_concurrent_runs=None):
    async def flush(**kwargs):
        pass

    async def state():
        return SimpleNamespace(elements=[])

    monkeypatch.setattr(module, "capture", lambda *a, **kw: None)
    monkeypatch.setattr(module, "flush", flush)
    monkeypatch.setattr(module, "_FINALIZE_SCHEDULING_RESERVE_SECONDS", 0.05)
    agent = FinishingAgent(timeout, num_concurrent_runs=num_concurrent_runs)
    agent.shared_state = SimpleNamespace(
        instruction="test task",
        workflow_completed=False,
        step_number=1,
        visited_packages=set(),
        visited_activities=set(),
        telemetry_config_enabled=False,
    )
    agent.user_id = None
    agent.output_model = None
    agent._using_external_agent = False
    agent.config = SimpleNamespace(
        agent=SimpleNamespace(
            manager=SimpleNamespace(vision=True),
            executor=SimpleNamespace(vision=False),
            fast_agent=SimpleNamespace(vision=False),
        ),
        logging=SimpleNamespace(
            save_trajectory="none",
            debug=False,
            trajectory_path="trajectories",
            trajectory_gifs=False,
        ),
        tracing=SimpleNamespace(langfuse_screenshots=False),
    )
    agent._stream_screenshots = False
    agent.structured_output_llm = object()
    agent.driver = None
    agent.action_ctx = SimpleNamespace(driver=SimpleNamespace(screenshot=screenshot))
    agent.state_provider = SimpleNamespace(get_state=state)
    agent.mcp_manager = None
    return agent


@pytest.mark.parametrize("success", [True, False])
def test_near_deadline_preserves_decision_and_terminates_stream(monkeypatch, success):
    async def run():
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def screenshot():
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        agent = _agent(monkeypatch, 0.6, screenshot)
        handler = agent.run(delay=0.3, success=success)
        events = [event async for event in handler.stream_events()]
        result = await handler
        assert entered.is_set(), "the hung epilog must actually run"
        await asyncio.wait_for(cancelled.wait(), 0.5)
        assert isinstance(result, ResultEvent)
        assert (result.success, result.reason) == (success, "decided")
        assert isinstance(events[-1], ResultEvent)
        assert sum(isinstance(e, FinalizeEvent) for e in events) == 1
        assert sum(isinstance(e, ResultEvent) for e in events) == 1

    asyncio.run(run())


def test_timeout_before_decision_still_fails(monkeypatch):
    async def run():
        async def screenshot():
            raise AssertionError("finalization must not start")

        agent = _agent(monkeypatch, 0.05, screenshot)
        with pytest.raises(WorkflowTimeoutError):
            await agent.run(delay=1, success=True)
        assert agent.shared_state.workflow_completed is False

    asyncio.run(run())


def test_epilog_ceiling_applies_with_plenty_of_workflow_time(monkeypatch):
    async def run():
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def screenshot():
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        monkeypatch.setattr(module, "_FINALIZE_BUDGET_SECONDS", 0.15)
        agent = _agent(monkeypatch, 60, screenshot)
        handler = agent.run(delay=0, success=True)

        async def collect():
            return [event async for event in handler.stream_events()], await handler

        events, result = await asyncio.wait_for(collect(), 1)
        assert entered.is_set()
        await asyncio.wait_for(cancelled.wait(), 0.5)
        assert result.success is True
        assert isinstance(events[-1], ResultEvent)

    asyncio.run(run())


def test_user_cancellation_during_epilog_still_cancels_workflow(monkeypatch):
    async def run():
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def screenshot():
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        agent = _agent(monkeypatch, 5, screenshot)
        handler = agent.run(delay=0, success=True)
        await asyncio.wait_for(entered.wait(), 1)
        await handler.cancel_run()
        with pytest.raises(WorkflowCancelledByUser):
            await handler
        await asyncio.wait_for(cancelled.wait(), 1)

    asyncio.run(run())


def test_serialized_runs_each_receive_a_runtime_scoped_finalization_budget(monkeypatch):
    async def run():
        extracted = []

        class Structured:
            def __init__(self, *, answer_text, **kwargs):
                self.answer_text = answer_text

            async def extract_structured_output(self, ctx, event):
                await asyncio.sleep(0.2)
                extracted.append(self.answer_text)
                return SimpleNamespace(
                    result={
                        "success": True,
                        "structured_output": {"run": self.answer_text},
                    }
                )

        monkeypatch.setattr(module, "StructuredOutputAgent", Structured)
        agent = _agent(
            monkeypatch,
            timeout=0.6,
            screenshot=lambda: None,
            num_concurrent_runs=1,
        )
        agent.output_model = object()
        agent.config.agent.manager.vision = False
        first_started = asyncio.Event()
        first = agent.run(
            delay=0.3,
            success=True,
            reason="first",
            started=first_started,
        )
        await asyncio.wait_for(first_started.wait(), timeout=1)
        await asyncio.sleep(0.05)
        second = agent.run(delay=0, success=True, reason="second")

        first_result = await first
        second_result = await second

        assert first_result.structured_output == {"run": "first"}
        assert second_result.structured_output == {"run": "second"}
        assert extracted == ["first", "second"]

    asyncio.run(run())


def test_second_run_cannot_extend_active_run_deadline(monkeypatch):
    async def run():
        screenshot_calls = 0

        async def screenshot():
            nonlocal screenshot_calls
            screenshot_calls += 1
            if screenshot_calls == 1:
                await asyncio.Future()

        agent = _agent(
            monkeypatch,
            timeout=0.5,
            screenshot=screenshot,
            num_concurrent_runs=1,
        )
        first_started = asyncio.Event()
        first = agent.run(delay=0.2, success=True, started=first_started)
        await asyncio.wait_for(first_started.wait(), timeout=1)
        await asyncio.sleep(0.15)
        second = agent.run(delay=0, success=True)

        first_result = await first
        second_result = await second

        assert first_result.success is True
        assert second_result.success is True
        assert screenshot_calls == 2

    asyncio.run(run())


def test_serialized_runs_use_distinct_trajectory_writers(monkeypatch, tmp_path):
    async def run():
        first_draining = asyncio.Event()
        release_first = asyncio.Event()
        second_written = asyncio.Event()
        writers = []

        class RecordingWriter(TrajectoryWriter):
            def __init__(self, queue_size=300):
                super().__init__(queue_size=queue_size)
                self.run_index = len(writers)
                writers.append(self)

            def write(self, trajectory, stage):
                if stage != "final":
                    return

                run_index = self.run_index

                class Job:
                    async def execute(self):
                        if run_index == 0:
                            first_draining.set()
                            await release_first.wait()
                        else:
                            second_written.set()

                self.worker.submit(Job())

        monkeypatch.setattr(module, "TrajectoryWriter", RecordingWriter)
        monkeypatch.setattr(module, "_FINALIZE_BUDGET_SECONDS", 0.05)
        monkeypatch.setattr(module, "_FINALIZE_TRAJECTORY_SECONDS", 0.5)
        agent = _agent(
            monkeypatch,
            timeout=2,
            screenshot=_no_screenshot,
            num_concurrent_runs=1,
        )
        agent.config.agent.manager.vision = False
        agent.config.logging.save_trajectory = "all"
        agent.config.logging.trajectory_path = str(tmp_path)

        first_result = await agent.run(delay=0, success=True, reason="first")
        await asyncio.wait_for(first_draining.wait(), timeout=1)
        second_result = await agent.run(delay=0, success=True, reason="second")

        assert first_result.success is True
        assert second_result.success is True
        assert len(writers) == 2
        assert writers[0] is not writers[1]
        assert second_written.is_set()
        assert writers[1].worker.queue.empty()

        release_first.set()
        background_cleanup = list(module._ABANDONED_FINALIZE_TASKS)
        if background_cleanup:
            await asyncio.wait_for(
                asyncio.gather(*background_cleanup, return_exceptions=True),
                timeout=1,
            )
        await asyncio.sleep(0)
        assert not writers[0].worker.running
        assert not writers[1].worker.running

    async def _no_screenshot():
        return None

    asyncio.run(run())
