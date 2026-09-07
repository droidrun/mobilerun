"""Final screenshot / UI-state capture is best-effort telemetry that must never
turn an already-produced agent result into a workflow timeout.

Regression for the 2026-09-07 Dev incident: the agent reached
``complete(success=true)`` but the workflow failed after 120 s with
``Currently active steps: finalize`` because the final screenshot kept
retrying/hanging inside the device SDK.
"""

import asyncio
import time
from types import SimpleNamespace

import mobilerun.agent.droid.droid_agent as droid_agent_module
from mobilerun.agent.common.events import RecordUIStateEvent, ScreenshotEvent
from mobilerun.agent.droid.droid_agent import MobileAgent
from mobilerun.agent.droid.events import FinalizeEvent, ResultEvent


def _run(coro):
    return asyncio.run(coro)


async def _hang_forever():
    await asyncio.sleep(3600)


def _finalizing_agent(monkeypatch, *, screenshot, get_state, budget: float):
    """A MobileAgent stub carrying exactly the state ``finalize`` touches."""
    monkeypatch.setattr(droid_agent_module, "capture", lambda *a, **k: None)

    async def _flush(*a, **k):
        return None

    monkeypatch.setattr(droid_agent_module, "flush", _flush)
    monkeypatch.setattr(droid_agent_module, "record_langfuse_screenshot", lambda *a, **k: None)

    agent = object.__new__(MobileAgent)
    agent.shared_state = SimpleNamespace(
        workflow_completed=False,
        step_number=1,
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
        logging=SimpleNamespace(save_trajectory="none", debug=False),
        tracing=SimpleNamespace(langfuse_screenshots=False),
    )
    agent._stream_screenshots = False
    agent.action_ctx = SimpleNamespace(driver=SimpleNamespace(screenshot=screenshot))
    agent.state_provider = SimpleNamespace(get_state=get_state)
    agent.macro_recorder = None
    agent.mcp_manager = None
    agent.final_telemetry_budget_seconds = budget
    return agent


def _finalize(agent):
    events = []
    ctx = SimpleNamespace(write_event_to_stream=events.append)
    started = time.monotonic()
    result = _run(agent.finalize(ctx, FinalizeEvent(success=True, reason="done")))
    return result, events, time.monotonic() - started


def test_hung_final_screenshot_cannot_overturn_success(monkeypatch):
    async def get_state():
        return SimpleNamespace(elements=[])

    agent = _finalizing_agent(
        monkeypatch, screenshot=_hang_forever, get_state=get_state, budget=0.2
    )

    result, events, elapsed = _finalize(agent)

    assert isinstance(result, ResultEvent)
    assert result.success is True
    assert result.reason == "done"
    assert elapsed < 2.0, "finalize must return within its local telemetry budget"
    # The hung screenshot AND the (never reached) UI-state read are both dropped.
    assert not any(isinstance(e, (ScreenshotEvent, RecordUIStateEvent)) for e in events)


def test_hung_final_ui_state_cannot_overturn_success(monkeypatch):
    async def screenshot():
        return b"png"

    agent = _finalizing_agent(
        monkeypatch, screenshot=screenshot, get_state=_hang_forever, budget=0.2
    )

    result, events, elapsed = _finalize(agent)

    assert isinstance(result, ResultEvent)
    assert result.success is True
    assert elapsed < 2.0
    # The screenshot that completed inside the budget is still emitted.
    assert any(isinstance(e, ScreenshotEvent) for e in events)
    assert not any(isinstance(e, RecordUIStateEvent) for e in events)


def test_failing_final_screenshot_still_captures_ui_state(monkeypatch):
    async def screenshot():
        raise RuntimeError("HTTP 500 context deadline exceeded")

    async def get_state():
        return SimpleNamespace(elements=[{"index": 1, "text": "e1"}])

    agent = _finalizing_agent(monkeypatch, screenshot=screenshot, get_state=get_state, budget=5.0)

    result, events, _ = _finalize(agent)

    assert result.success is True
    assert not any(isinstance(e, ScreenshotEvent) for e in events)
    ui = [e for e in events if isinstance(e, RecordUIStateEvent)]
    assert len(ui) == 1 and ui[0].ui_state[0]["text"] == "e1"


def test_normal_run_emits_final_screenshot_and_ui_state(monkeypatch):
    async def screenshot():
        return b"png"

    async def get_state():
        return SimpleNamespace(elements=[{"index": 1, "text": "e1"}])

    agent = _finalizing_agent(monkeypatch, screenshot=screenshot, get_state=get_state, budget=5.0)

    result, events, _ = _finalize(agent)

    assert result.success is True
    kinds = [type(e) for e in events]
    assert kinds.count(ScreenshotEvent) == 1
    assert kinds.count(RecordUIStateEvent) == 1


def test_default_budget_is_well_below_typical_task_timeouts():
    assert 0 < MobileAgent.final_telemetry_budget_seconds <= 30
