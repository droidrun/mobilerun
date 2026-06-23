import asyncio
import logging
from types import SimpleNamespace

import pytest
from llama_index.core.base.llms.types import ChatMessage

from mobilerun.agent.manager.events import ManagerContextEvent
from mobilerun.agent.manager.manager_agent import ManagerAgent
from mobilerun.agent.manager.prompts import (
    ManagerResponseValidationError,
    parse_manager_response,
)


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(message=SimpleNamespace(content=content))


def _run(coro):
    return asyncio.run(coro)


class FakeContext:
    def __init__(self) -> None:
        self.events = []

    def write_event_to_stream(self, event) -> None:
        self.events.append(event)


def _manager() -> ManagerAgent:
    agent = object.__new__(ManagerAgent)
    agent.llm = SimpleNamespace(class_name=lambda: "FakeLLM")
    agent.agent_config = SimpleNamespace(streaming=False)
    agent.shared_state = SimpleNamespace(
        screenshot=None,
        message_history=[],
        previous_plan="",
        plan="",
        current_subgoal="",
        last_thought="",
        answer="",
        progress_summary="",
        append_memory=lambda text: None,
    )

    async def build_system_prompt() -> str:
        return "system"

    agent._build_system_prompt = build_system_prompt
    agent._build_messages_with_context = lambda **kwargs: [
        ChatMessage(role="user", content="prompt")
    ]
    return agent


@pytest.fixture
def mobilerun_caplog(caplog):
    logger = logging.getLogger("mobilerun")
    previous = logger.propagate
    logger.propagate = True
    caplog.set_level(logging.WARNING, logger="mobilerun")
    yield caplog
    logger.propagate = previous


def test_manager_boundary_normalizes_missing_success_after_retry_exhaustion(
    monkeypatch,
    mobilerun_caplog,
):
    calls = []

    async def fake_acall(llm, messages, stream=False):
        calls.append((llm, messages, stream))
        return _response("<answer>Done</answer>")

    monkeypatch.setattr(
        "mobilerun.agent.manager.manager_agent.acall_with_retries", fake_acall
    )

    manager = _manager()
    ctx = FakeContext()

    response_event = _run(manager.get_response(ctx, ManagerContextEvent()))
    details_event = _run(manager.process_response(ctx, response_event))
    parsed = parse_manager_response(response_event.response)

    assert len(calls) == 4
    assert parsed["answer"] == "Done"
    assert parsed["success"] is True
    assert details_event.answer == "Done"
    assert details_event.success is True
    assert any(
        "omitted final success after retries" in record.message
        for record in mobilerun_caplog.records
    )


def test_manager_boundary_rejects_malformed_success_after_retry_exhaustion(
    monkeypatch,
):
    calls = []

    async def fake_acall(llm, messages, stream=False):
        calls.append((llm, messages, stream))
        return _response("<answer success='maybe'>Done</answer>")

    monkeypatch.setattr(
        "mobilerun.agent.manager.manager_agent.acall_with_retries", fake_acall
    )

    with pytest.raises(ManagerResponseValidationError):
        _run(_manager().get_response(FakeContext(), ManagerContextEvent()))

    assert len(calls) == 4


def test_manager_boundary_rejects_empty_final_body_after_retry_exhaustion(
    monkeypatch,
):
    calls = []

    async def fake_acall(llm, messages, stream=False):
        calls.append((llm, messages, stream))
        return _response("<answer success='true'></answer>")

    monkeypatch.setattr(
        "mobilerun.agent.manager.manager_agent.acall_with_retries", fake_acall
    )

    with pytest.raises(ManagerResponseValidationError):
        _run(_manager().get_response(FakeContext(), ManagerContextEvent()))

    assert len(calls) == 4


def test_manager_boundary_plan_final_conflict_does_not_fallback_to_success(
    monkeypatch,
    mobilerun_caplog,
):
    calls = []
    invalid_response = (
        "<plan>1. Continue checking the screen.</plan>"
        "<answer success='true'>Done</answer>"
    )

    async def fake_acall(llm, messages, stream=False):
        calls.append((llm, messages, stream))
        return _response(invalid_response)

    monkeypatch.setattr(
        "mobilerun.agent.manager.manager_agent.acall_with_retries", fake_acall
    )

    manager = _manager()
    ctx = FakeContext()

    response_event = _run(manager.get_response(ctx, ManagerContextEvent()))
    details_event = _run(manager.process_response(ctx, response_event))
    parsed = parse_manager_response(response_event.response)

    assert len(calls) == 4
    assert parsed["plan"] == "1. Continue checking the screen."
    assert parsed["answer"] == ""
    assert parsed["success"] is None
    assert details_event.success is None
    assert any(
        "continuing with the plan" in record.message
        for record in mobilerun_caplog.records
    )
    assert not any(
        "omitted final success after retries" in record.message
        for record in mobilerun_caplog.records
    )
