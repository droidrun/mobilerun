import asyncio
import os
import sys
import time
from types import SimpleNamespace

import pytest
from llama_index.core.workflow import StopEvent
from llama_index_instrumentation.span import active_span_id
from llama_index_instrumentation.span.base import BaseSpan
from llama_index_instrumentation.span_handlers.base import BaseSpanHandler
from mobilerun_core_local.driver.android import AndroidDriver
from pydantic import BaseModel

import mobilerun.agent.droid.droid_agent as module
import mobilerun.agent.oneflows.structured_output_agent as structured_module
import mobilerun.telemetry.phoenix as phoenix_module
import mobilerun.tools.ui.provider as provider_module
from mobilerun.agent.common.events import ScreenshotEvent
from mobilerun.agent.droid.droid_agent import MobileAgent
from mobilerun.agent.droid.events import FinalizeEvent
from mobilerun.agent.oneflows.structured_output_agent import StructuredOutputAgent
from mobilerun.agent.trajectory import TrajectoryWriter
from mobilerun.tools.ui.provider import AndroidStateProvider


async def _empty_state():
    return SimpleNamespace(elements=[])


async def _nothing():
    return None


class _Store:
    def __init__(self, deadline):
        self.deadline = deadline

    async def get(self, key, default=None):
        return self.deadline

    async def set(self, key, value):
        self.deadline = value


def _agent(monkeypatch, screenshot=_nothing, state=_empty_state, *, deadline=None):
    monkeypatch.setattr(module, "capture", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module, "record_langfuse_screenshot", lambda *args, **kwargs: None
    )

    async def flush(*args, **kwargs):
        return None

    monkeypatch.setattr(module, "flush", flush)
    agent = object.__new__(MobileAgent)
    agent.shared_state = SimpleNamespace(
        workflow_completed=False,
        step_number=2,
        visited_packages=set(),
        visited_activities=set(),
        telemetry_config_enabled=False,
    )
    agent.user_id = None
    agent.output_model = None
    agent.config = SimpleNamespace(
        agent=SimpleNamespace(
            manager=SimpleNamespace(vision=True),
            executor=SimpleNamespace(vision=False),
            fast_agent=SimpleNamespace(vision=False),
        ),
        logging=SimpleNamespace(
            save_trajectory="none", debug=False, trajectory_gifs=False
        ),
        tracing=SimpleNamespace(langfuse_screenshots=False),
    )
    agent._stream_screenshots = False
    agent.action_ctx = SimpleNamespace(driver=SimpleNamespace(screenshot=screenshot))
    agent.state_provider = SimpleNamespace(get_state=state)
    agent.macro_recorder = None
    agent.mcp_manager = None
    agent._test_workflow_deadline = deadline
    agent.timeout = 60
    agent.structured_output_llm = object()
    return agent


async def _finalize(agent, *, success=True, reason="answer", events=None):
    events = [] if events is None else events
    result = await agent.finalize(
        SimpleNamespace(
            write_event_to_stream=events.append,
            store=_Store(agent._test_workflow_deadline),
        ),
        FinalizeEvent(success=success, reason=reason),
    )
    return result, events


def _enable_trajectory(agent, stop=_nothing, write_final=lambda *args: None):
    agent.config.logging.save_trajectory = "all"
    agent.trajectory = SimpleNamespace(trajectory_folder="unused", macro=[])
    agent.macro_recorder = None
    agent.driver = None
    agent.trajectory_writer = SimpleNamespace(write_final=write_final, stop=stop)


def _structured(result=None):
    class Structured:
        def __init__(self, **kwargs):
            pass

        async def extract_structured_output(self, ctx, event):
            return SimpleNamespace(
                result={"success": True, "structured_output": result}
            )

    return Structured


def _failed_structured(error_message):
    class Structured:
        def __init__(self, **kwargs):
            pass

        async def extract_structured_output(self, ctx, event):
            return SimpleNamespace(
                result={
                    "success": False,
                    "structured_output": None,
                    "error_message": error_message,
                }
            )

    return Structured


class _RecordedSpan(BaseSpan):
    arguments: dict
    result: object = None
    error: BaseException | None = None


class _RecordingSpanHandler(BaseSpanHandler[_RecordedSpan]):
    def new_span(self, id_, bound_args, instance=None, parent_span_id=None, tags=None):
        return _RecordedSpan(
            id_=id_,
            parent_id=parent_span_id,
            tags=tags or {},
            arguments=dict(bound_args.arguments),
        )

    def prepare_to_exit_span(self, id_, bound_args, instance=None, result=None):
        span = self.open_spans[id_]
        span.result = result
        self.completed_spans.append(span)
        return span

    def prepare_to_drop_span(self, id_, bound_args, instance=None, err=None):
        span = self.open_spans[id_]
        span.error = err
        self.dropped_spans.append(span)
        return span


@pytest.fixture
def span_recorder():
    handler = _RecordingSpanHandler()
    phoenix_module.dispatcher.add_span_handler(handler)
    try:
        yield handler
    finally:
        phoenix_module.dispatcher.span_handlers.remove(handler)


def test_direct_structured_extraction_restores_observations_and_parentage(
    monkeypatch, span_recorder
):
    async def run():
        agent = _agent(monkeypatch)
        agent.output_model = object()
        agent.config.agent.manager.vision = False
        monkeypatch.setattr(
            module, "StructuredOutputAgent", _structured({"parsed": True})
        )

        token = active_span_id.set("MobileAgent.finalize-parent")
        try:
            result, _ = await _finalize(agent)
        finally:
            active_span_id.reset(token)

        assert result.structured_output == {"parsed": True}
        assert not span_recorder.open_spans
        assert len(span_recorder.completed_spans) == 2
        run_span = next(
            span
            for span in span_recorder.completed_spans
            if span.id_.startswith("StructuredOutputAgent.run-")
        )
        extract_span = next(
            span
            for span in span_recorder.completed_spans
            if span.id_.startswith("StructuredOutputAgent.extract_structured_output-")
        )
        assert run_span.parent_id == "MobileAgent.finalize-parent"
        assert extract_span.parent_id == run_span.id_
        assert run_span.arguments == extract_span.arguments == {}

        for span in span_recorder.completed_spans:
            assert span.result.result == {
                "success": True,
                "structured_output": {"parsed": True},
            }

    asyncio.run(run())


def test_structured_extraction_timeout_drops_observations(monkeypatch, span_recorder):
    async def run():
        entered = asyncio.Event()

        class Structured:
            def __init__(self, **kwargs):
                pass

            async def extract_structured_output(self, ctx, event):
                entered.set()
                await asyncio.Future()

        monkeypatch.setattr(module, "_FINALIZE_BUDGET_SECONDS", 0.03)
        monkeypatch.setattr(module, "StructuredOutputAgent", Structured)
        agent = _agent(monkeypatch)
        agent.output_model = object()
        agent.config.agent.manager.vision = False

        token = active_span_id.set("MobileAgent.finalize-parent")
        try:
            result, _ = await _finalize(agent)
            assert entered.is_set()
            abandoned = list(module._ABANDONED_FINALIZE_TASKS)
            if abandoned:
                await asyncio.wait_for(
                    asyncio.gather(*abandoned, return_exceptions=True), timeout=1
                )
            await asyncio.sleep(0)
        finally:
            active_span_id.reset(token)

        assert result.structured_output is None
        assert not module._ABANDONED_FINALIZE_TASKS
        assert not span_recorder.open_spans
        assert not span_recorder.completed_spans
        assert len(span_recorder.dropped_spans) == 2
        assert all(
            isinstance(span.error, asyncio.CancelledError)
            for span in span_recorder.dropped_spans
        )
        assert any(
            span.id_.startswith("StructuredOutputAgent.run-")
            for span in span_recorder.dropped_spans
        )
        assert any(
            span.id_.startswith("StructuredOutputAgent.extract_structured_output-")
            for span in span_recorder.dropped_spans
        )

    asyncio.run(run())


def test_direct_structured_extraction_records_error_result(monkeypatch, span_recorder):
    async def run():
        agent = _agent(monkeypatch)
        agent.output_model = object()
        agent.config.agent.manager.vision = False
        monkeypatch.setattr(
            module, "StructuredOutputAgent", _failed_structured("invalid response")
        )

        result, _ = await _finalize(agent)

        assert result.structured_output is None
        assert not span_recorder.open_spans
        assert len(span_recorder.completed_spans) == 2
        for span in span_recorder.completed_spans:
            assert span.result.result == {
                "success": False,
                "structured_output": None,
                "error_message": "invalid response",
            }

    asyncio.run(run())


def test_resistant_final_observation_returns_before_late_task_and_publishes_nothing(
    monkeypatch,
):
    async def run():
        entered, released, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def screenshot():
            entered.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                await released.wait()
                completed.set()
                return b"late"

        monkeypatch.setattr(module, "_FINALIZE_BUDGET_SECONDS", 0.03)
        events = []
        task = asyncio.create_task(
            _finalize(_agent(monkeypatch, screenshot), events=events)
        )
        await asyncio.wait_for(entered.wait(), timeout=1)
        result, _ = await asyncio.wait_for(task, timeout=0.5)
        assert result.success is True
        assert not completed.is_set()
        assert not any(isinstance(event, ScreenshotEvent) for event in events)
        before_release = list(events)
        released.set()
        await asyncio.wait_for(completed.wait(), timeout=1)
        await asyncio.sleep(0)
        assert events == before_release

    asyncio.run(run())


def test_expired_workflow_deadline_preserves_false_result_without_optional_work(
    monkeypatch,
):
    async def run():
        called = False

        async def screenshot():
            nonlocal called
            called = True
            return b"png"

        agent = _agent(monkeypatch, screenshot, deadline=time.monotonic())
        result, _ = await _finalize(agent, success=False, reason="no")
        assert (result.success, result.reason, called) == (False, "no", False)

    asyncio.run(run())


def test_external_cancellation_cancels_owned_stage(monkeypatch):
    async def run():
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def screenshot():
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(_finalize(_agent(monkeypatch, screenshot)))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(cancelled.wait(), timeout=1)

    asyncio.run(run())


@pytest.mark.parametrize("success", [True, False])
def test_finalize_keeps_normal_structured_output_and_artifact_events(
    monkeypatch, success
):
    async def screenshot():
        return b"png"

    async def state():
        return SimpleNamespace(elements=[{"text": "element"}])

    agent = _agent(monkeypatch, screenshot, state)
    agent.output_model = object()
    monkeypatch.setattr(module, "StructuredOutputAgent", _structured({"parsed": True}))
    result, events = asyncio.run(_finalize(agent, success=success))
    assert (result.success, result.reason) == (success, "answer")
    assert result.structured_output == {"parsed": True}
    assert isinstance(events[0], FinalizeEvent)
    assert isinstance(events[1], ScreenshotEvent)
    assert events[1].screenshot == b"png"
    assert events[2].ui_state == [{"text": "element"}]


@pytest.mark.parametrize("stage", ["capture", "flush", "screenshot", "state"])
def test_epilog_errors_are_isolated_and_preserve_false_result(monkeypatch, stage):
    async def run():
        entered = []

        async def screenshot():
            entered.append("screenshot")
            if stage == "screenshot":
                raise RuntimeError(stage)

        async def state():
            entered.append("state")
            if stage == "state":
                raise RuntimeError(stage)
            return SimpleNamespace(elements=[])

        agent = _agent(monkeypatch, screenshot, state)
        if stage == "capture":
            monkeypatch.setattr(
                module,
                "capture",
                lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(stage)),
            )
        if stage == "flush":

            async def flush(**kwargs):
                entered.append("flush")
                raise RuntimeError(stage)

            monkeypatch.setattr(module, "flush", flush)
        result, _ = await _finalize(agent, success=False, reason="no")
        assert (result.success, result.reason) == (False, "no")
        assert stage in entered or stage == "capture"

    asyncio.run(run())


def test_trajectory_write_error_still_stops_and_disconnects_without_saved_log(
    monkeypatch,
):
    async def run():
        stopped, disconnected, messages = False, False, []

        async def stop():
            nonlocal stopped
            stopped = True

        async def disconnect():
            nonlocal disconnected
            disconnected = True

        def write_final(*args):
            raise RuntimeError("disk")

        agent = _agent(monkeypatch)
        _enable_trajectory(agent, stop, write_final)
        agent.mcp_manager = SimpleNamespace(disconnect_all=disconnect)
        monkeypatch.setattr(module.logger, "info", messages.append)
        await _finalize(agent)
        assert stopped and disconnected
        assert not any("Trajectory saved" in str(message) for message in messages)

    asyncio.run(run())


def test_android_final_observation_is_single_read_and_normal_read_retries(monkeypatch):
    async def parse(value):
        return value

    async def run():
        final_provider = object.__new__(AndroidStateProvider)
        final_calls = []

        class Driver:
            async def get_ui_tree(self):
                final_calls.append("read")
                return {"raw": True}

        final_provider.driver = Driver()
        final_provider._state_from_data_async = parse

        async def no_retry(*args, **kwargs):
            raise AssertionError("final observation must not retry")

        monkeypatch.setattr(provider_module, "fetch_state_with_retry", no_retry)
        assert await final_provider.get_final_state() == {"raw": True}
        assert final_calls == ["read"]

        normal_provider = object.__new__(AndroidStateProvider)
        normal_provider.driver = object.__new__(AndroidDriver)
        normal_provider.driver.device = object()
        normal_provider.driver.get_ui_tree = object()
        normal_provider._state_from_data_async = parse
        retried = False

        async def retry(*, fetch, recovery):
            nonlocal retried
            retried = True
            assert fetch is normal_provider.driver.get_ui_tree
            assert recovery == normal_provider._recover_portal
            return {"retried": True}

        monkeypatch.setattr(provider_module, "fetch_state_with_retry", retry)
        assert await normal_provider.get_state() == {"retried": True}
        assert retried

    asyncio.run(run())


class _StructuredOutput(BaseModel):
    value: str


@pytest.mark.parametrize("raises", [False, True])
def test_direct_structured_extraction_matches_real_workflow(monkeypatch, raises):
    async def inference(*args, **kwargs):
        if raises:
            raise RuntimeError("inference failed")
        return _StructuredOutput(value="parsed")

    monkeypatch.setattr(
        structured_module, "astructured_predict_with_retries", inference
    )

    async def run():
        reference = StructuredOutputAgent(
            llm=object(), pydantic_model=_StructuredOutput, answer_text="answer"
        )
        handler = reference.run()
        reference_events = [
            event
            async for event in handler.stream_events()
            if not isinstance(event, StopEvent)
        ]
        reference_result = await handler
        agent = _agent(monkeypatch)
        agent.output_model = _StructuredOutput
        agent.config.agent.manager.vision = False
        result, events = await _finalize(agent)
        assert events[1:] == reference_events
        assert result.structured_output == reference_result["structured_output"]
        assert (result.success, result.reason) == (True, "answer")

    asyncio.run(run())


@pytest.mark.parametrize(
    "leading_stage", ["structured", "flush", "screenshot", "state"]
)
def test_hung_epilog_stages_share_one_budget_and_keep_cleanup(
    monkeypatch, leading_stage
):
    async def run():
        entered = []

        async def stall(name):
            entered.append(name)
            await asyncio.Future()

        class Structured:
            def __init__(self, **kwargs):
                pass

            async def extract_structured_output(self, ctx, event):
                if leading_stage == "structured":
                    await stall("structured")
                return SimpleNamespace(
                    result={"success": True, "structured_output": None}
                )

        async def screenshot():
            if leading_stage == "screenshot":
                await stall("screenshot")

        async def state():
            if leading_stage == "state":
                await stall("state")
            return SimpleNamespace(elements=[])

        async def flush(**kwargs):
            if leading_stage == "flush":
                await stall("flush")

        monkeypatch.setattr(module, "_FINALIZE_BUDGET_SECONDS", 0.12)
        monkeypatch.setattr(module, "_FINALIZE_TRAJECTORY_SECONDS", 0.04)
        monkeypatch.setattr(module, "_FINALIZE_MCP_SECONDS", 0.02)
        monkeypatch.setattr(module, "StructuredOutputAgent", Structured)
        agent = _agent(monkeypatch, screenshot, state)
        monkeypatch.setattr(module, "flush", flush)
        agent.output_model = object()
        agent.config.agent.manager.vision = False
        _enable_trajectory(agent, stop=lambda: stall("trajectory"))
        agent.mcp_manager = SimpleNamespace(disconnect_all=lambda: stall("mcp"))
        started = time.monotonic()
        result, _ = await _finalize(agent)
        assert result.success is True
        assert time.monotonic() - started < 0.3
        assert {leading_stage, "trajectory", "mcp"} <= set(entered)

    asyncio.run(run())


def test_finalize_stage_policy_uses_named_caps_and_no_phantom_cleanup(monkeypatch):
    async def run():
        calls = {}
        original = module._run_finalize_stage

        async def record(name, awaitable, deadline, cap, **kwargs):
            calls[name] = (deadline, cap, kwargs.get("continue_in_background", False))
            return await original(name, awaitable, deadline, cap, **kwargs)

        monkeypatch.setattr(module, "_run_finalize_stage", record)
        monkeypatch.setattr(module, "StructuredOutputAgent", _structured())
        agent = _agent(monkeypatch)
        agent.output_model = object()
        agent.config.agent.manager.vision = False
        _enable_trajectory(agent)
        agent.mcp_manager = SimpleNamespace(disconnect_all=_nothing)
        await _finalize(agent)
        assert calls["structured-output"][1] == module._FINALIZE_BUDGET_SECONDS
        assert calls["telemetry-flush"][1] == module._FINALIZE_TELEMETRY_SECONDS
        assert calls["trajectory-stop"][1] == module._FINALIZE_TRAJECTORY_SECONDS
        assert calls["mcp-disconnect"][1] == module._FINALIZE_MCP_SECONDS
        assert calls["trajectory-stop"][2] is True
        assert calls["mcp-disconnect"][2] is True

        calls.clear()
        monkeypatch.setattr(module, "_FINALIZE_BUDGET_SECONDS", 0.2)
        started = time.monotonic()
        bare = _agent(monkeypatch)
        bare.output_model = object()
        bare.config.agent.manager.vision = False
        await _finalize(bare)
        assert calls["structured-output"][0] - started > 0.18

    asyncio.run(run())


@pytest.mark.parametrize("available", [32.9, 33.1])
def test_cleanup_reserve_is_continuous_at_demand_threshold(monkeypatch, available):
    async def run():
        structured_deadline = None
        original = module._run_finalize_stage

        async def record(name, awaitable, deadline, cap, **kwargs):
            nonlocal structured_deadline
            if name == "structured-output":
                structured_deadline = deadline
            return await original(name, awaitable, deadline, cap, **kwargs)

        monkeypatch.setattr(module, "_run_finalize_stage", record)
        monkeypatch.setattr(module, "StructuredOutputAgent", _structured())
        started = time.monotonic()
        agent = _agent(
            monkeypatch,
            deadline=started + available + module._FINALIZE_SCHEDULING_RESERVE_SECONDS,
        )
        agent.output_model = object()
        agent.config.agent.manager.vision = False
        _enable_trajectory(agent)
        agent.mcp_manager = SimpleNamespace(disconnect_all=_nothing)
        await _finalize(agent)
        assert structured_deadline is not None
        assert structured_deadline - started > available * 0.45

    asyncio.run(run())


def _process_exists(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_expired_budget_still_closes_real_writer_and_mcp(monkeypatch, tmp_path):
    server_script = tmp_path / "finalization_mcp_server.py"
    server_script.write_text("""
import os

from mcp.server.fastmcp import FastMCP

server = FastMCP("finalization-cleanup-test")


@server.tool()
def pid() -> str:
    return str(os.getpid())


if __name__ == "__main__":
    server.run(transport="stdio")
""".lstrip())

    async def run():
        writer = TrajectoryWriter()
        await writer.start()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(server_script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        server_pid = process.pid

        async def disconnect_all():
            assert process.stdin is not None
            process.stdin.close()
            await process.wait()

        manager = SimpleNamespace(
            connected_servers=["fixture"], disconnect_all=disconnect_all
        )
        assert writer.worker.running
        assert manager.connected_servers == ["fixture"]
        assert _process_exists(server_pid)

        agent = _agent(monkeypatch, deadline=time.monotonic() - 1)
        agent.config.agent.manager.vision = False
        agent.config.logging.save_trajectory = "all"
        agent.trajectory = SimpleNamespace(trajectory_folder="unused", macro=[])
        agent.driver = None
        agent.trajectory_writer = writer
        monkeypatch.setattr(writer, "write_final", lambda *args: None)
        agent.mcp_manager = manager

        try:
            result, _ = await _finalize(agent)
            background_cleanup = list(module._ABANDONED_FINALIZE_TASKS)
            if background_cleanup:
                await asyncio.wait_for(
                    asyncio.gather(*background_cleanup, return_exceptions=True),
                    timeout=5,
                )
            await asyncio.sleep(0)

            process_deadline = time.monotonic() + 2
            while _process_exists(server_pid) and time.monotonic() < process_deadline:
                await asyncio.sleep(0.01)

            assert result.success is True
            assert not writer.worker.running
            assert not _process_exists(server_pid)
        finally:
            await writer.stop(timeout=0.1)
            if process.returncode is None:
                await disconnect_all()

    asyncio.run(run())
