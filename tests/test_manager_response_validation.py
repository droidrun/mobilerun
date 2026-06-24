import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from llama_index.core.base.llms.types import ChatMessage

from mobilerun.agent.droid.droid_agent import MobileAgent
from mobilerun.agent.droid.events import (
    FinalizeEvent,
    ManagerInputEvent,
    ManagerPlanEvent,
)
from mobilerun.agent.manager.manager_agent import ManagerAgent
from mobilerun.agent.manager.prompts import (
    ManagerResponseValidationError,
    parse_manager_response,
    strip_manager_final_tags,
    validate_manager_response,
)
from mobilerun.agent.manager.stateless_manager_agent import StatelessManagerAgent


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(message=SimpleNamespace(content=content))


def _stateful_manager() -> ManagerAgent:
    agent = object.__new__(ManagerAgent)
    agent.llm = SimpleNamespace()
    agent.agent_config = SimpleNamespace(streaming=False)
    return agent


def _stateless_manager() -> StatelessManagerAgent:
    agent = object.__new__(StatelessManagerAgent)
    agent.llm = SimpleNamespace()
    return agent


def _run(coro):
    return asyncio.run(coro)


def test_parse_final_success_attribute_is_bound_to_matching_final_tag():
    parsed = parse_manager_response(
        """
        <thought>done</thought>
        <request_accomplished reason='checked' success='TRUE'>
        Android 16 is installed.
        </request_accomplished>
        """
    )

    assert parsed["answer"] == "Android 16 is installed."
    assert parsed["success"] is True
    assert parsed["final_tag"] == "request_accomplished"


def test_parse_answer_alias_and_missing_success_stays_none():
    parsed = parse_manager_response("<answer>Done</answer>")

    assert parsed["answer"] == "Done"
    assert parsed["success"] is None
    assert parsed["success_attr_present"] is False
    assert parsed["final_tag"] == "answer"


def test_validate_rejects_plan_plus_final_answer_but_can_continue_with_plan():
    parsed = parse_manager_response(
        """
        <thought>the version is visible</thought>
        <plan>1. Read the Android version from Settings.</plan>
        <request_accomplished success="true">Android 16.</request_accomplished>
        """
    )

    validation = validate_manager_response(parsed)

    assert not validation.is_valid
    assert validation.can_continue_with_plan
    assert "exactly one" in validation.error_message


def test_validate_missing_success_final_answer_can_fallback_to_success():
    validation = validate_manager_response(
        parse_manager_response("<answer>Done</answer>")
    )

    assert not validation.is_valid
    assert validation.can_accept_final_without_success
    assert "success" in validation.error_message


@pytest.mark.parametrize(
    "response",
    [
        "<plan>1. Read version</plan>"
        "<request_accomplished success='true'></request_accomplished>",
        "<plan>1. Read version</plan>"
        "<answer success='true'>Done</answer>"
        "<request_accomplished success='false'>Failed</request_accomplished>",
    ],
)
def test_validate_rejects_stray_final_tags_but_can_continue_with_valid_plan(
    response,
):
    validation = validate_manager_response(parse_manager_response(response))

    assert not validation.is_valid
    assert validation.can_continue_with_plan


def test_strip_manager_final_tags_keeps_plan_for_safe_continuation():
    output = """
    <thought>the version is visible</thought>
    <plan>1. Read the Android version from Settings.</plan>
    <request_accomplished success="true">Android 16.</request_accomplished>
    """

    sanitized = strip_manager_final_tags(output)
    parsed = parse_manager_response(sanitized)

    assert parsed["plan"] == "1. Read the Android version from Settings."
    assert parsed["answer"] == ""
    assert parsed["success"] is None


@pytest.mark.parametrize(
    "response, expected_error",
    [
        ("", "provide exactly one"),
        ("<answer>Done</answer>", "success"),
        (
            "<request_accomplished success='maybe'>Done</request_accomplished>",
            "success",
        ),
        (
            "<request_accomplished success='true'></request_accomplished>",
            "must not be empty",
        ),
        (
            "<plan>1. Open Settings</plan><plan>2. Read version</plan>",
            "multiple <plan>",
        ),
        (
            "<plan></plan><answer success='true'>Done</answer>",
            "<plan> tag must not be empty",
        ),
        (
            "<answer success='true'>Done</answer><answer success='false'>No</answer>",
            "multiple final",
        ),
    ],
)
def test_validate_rejects_malformed_responses(response, expected_error):
    validation = validate_manager_response(parse_manager_response(response))

    assert not validation.is_valid
    assert expected_error in validation.error_message


@pytest.mark.parametrize(
    "response",
    [
        "<plan>1. Open Settings</plan>",
        "<request_accomplished success='true'>Done</request_accomplished>",
        "<answer success='false'>Could not complete.</answer>",
    ],
)
def test_validate_accepts_single_valid_manager_output(response):
    validation = validate_manager_response(parse_manager_response(response))

    assert validation.is_valid
    assert validation.error_message is None


def test_stateful_manager_retries_invalid_response_then_returns_valid(monkeypatch):
    calls = []

    async def fake_acall(llm, messages, stream=False):
        calls.append((llm, messages, stream))
        return _response(
            "<request_accomplished success='true'>Android 16.</request_accomplished>"
        )

    monkeypatch.setattr(
        "mobilerun.agent.manager.manager_agent.acall_with_retries", fake_acall
    )

    output = _run(
        _stateful_manager()._validate_and_retry(
            [ChatMessage(role="user", content="prompt")],
            "<plan>1. Read version</plan>"
            "<request_accomplished success='true'>Android 16.</request_accomplished>",
        )
    )

    assert output == (
        "<request_accomplished success='true'>Android 16.</request_accomplished>"
    )
    assert len(calls) == 1
    assert calls[0][2] is False


def test_stateful_manager_strips_final_answer_after_retry_exhaustion(monkeypatch):
    async def fake_acall(llm, messages, stream=False):
        return _response(
            "<plan>1. Read version</plan>"
            "<request_accomplished success='true'>Android 16.</request_accomplished>"
        )

    monkeypatch.setattr(
        "mobilerun.agent.manager.manager_agent.acall_with_retries", fake_acall
    )

    output = _run(
        _stateful_manager()._validate_and_retry(
            [ChatMessage(role="user", content="prompt")],
            "<plan>1. Read version</plan>"
            "<request_accomplished success='true'>Android 16.</request_accomplished>",
        )
    )

    parsed = parse_manager_response(output)
    assert parsed["plan"] == "1. Read version"
    assert parsed["answer"] == ""


def test_stateful_manager_raises_validation_error_after_retry_exception(monkeypatch):
    async def fake_acall(llm, messages, stream=False):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "mobilerun.agent.manager.manager_agent.acall_with_retries", fake_acall
    )

    with pytest.raises(ManagerResponseValidationError):
        _run(
            _stateful_manager()._validate_and_retry(
                [ChatMessage(role="user", content="prompt")],
                "<answer success='maybe'>Done</answer>",
            )
        )


def test_stateful_manager_normalizes_missing_success_after_retry_exhaustion(
    monkeypatch,
):
    async def fake_acall(llm, messages, stream=False):
        return _response("<answer>Done</answer>")

    monkeypatch.setattr(
        "mobilerun.agent.manager.manager_agent.acall_with_retries", fake_acall
    )

    output = _run(
        _stateful_manager()._validate_and_retry(
            [ChatMessage(role="user", content="prompt")],
            "<answer>Done</answer>",
        )
    )

    parsed = parse_manager_response(output)
    assert parsed["answer"] == "Done"
    assert parsed["success"] is True


def test_stateful_manager_normalizes_missing_success_after_retry_exception(
    monkeypatch,
):
    async def fake_acall(llm, messages, stream=False):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "mobilerun.agent.manager.manager_agent.acall_with_retries", fake_acall
    )

    output = _run(
        _stateful_manager()._validate_and_retry(
            [ChatMessage(role="user", content="prompt")],
            "<answer>Done</answer>",
        )
    )

    parsed = parse_manager_response(output)
    assert parsed["answer"] == "Done"
    assert parsed["success"] is True


def test_stateful_manager_strips_final_answer_after_retry_exception(monkeypatch):
    async def fake_acall(llm, messages, stream=False):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "mobilerun.agent.manager.manager_agent.acall_with_retries", fake_acall
    )

    output = _run(
        _stateful_manager()._validate_and_retry(
            [ChatMessage(role="user", content="prompt")],
            "<plan>1. Read version</plan>"
            "<request_accomplished success='true'>Android 16.</request_accomplished>",
        )
    )

    parsed = parse_manager_response(output)
    assert parsed["plan"] == "1. Read version"
    assert parsed["answer"] == ""


def test_stateless_manager_strips_final_answer_after_retry_exhaustion(monkeypatch):
    async def fake_acall(llm, messages):
        return _response(
            "<plan>1. Read version</plan>" "<answer success='true'>Android 16.</answer>"
        )

    monkeypatch.setattr(
        "mobilerun.agent.manager.stateless_manager_agent.acall_with_retries", fake_acall
    )

    output = _run(
        _stateless_manager()._validate_and_retry(
            [{"role": "user", "content": [{"text": "prompt"}]}],
            "<plan>1. Read version</plan>"
            "<answer success='true'>Android 16.</answer>",
        )
    )

    parsed = parse_manager_response(output)
    assert parsed["plan"] == "1. Read version"
    assert parsed["answer"] == ""


def test_stateless_manager_raises_validation_error_after_retry_exhaustion(monkeypatch):
    async def fake_acall(llm, messages):
        return _response("<answer success='maybe'>Done</answer>")

    monkeypatch.setattr(
        "mobilerun.agent.manager.stateless_manager_agent.acall_with_retries", fake_acall
    )

    with pytest.raises(ManagerResponseValidationError):
        _run(
            _stateless_manager()._validate_and_retry(
                [{"role": "user", "content": [{"text": "prompt"}]}],
                "<answer success='maybe'>Done</answer>",
            )
        )


def test_droid_agent_run_manager_turns_validation_error_into_failed_finalize():
    class FakeHandler:
        async def stream_events(self):
            if False:
                yield None

        def __await__(self):
            async def fail():
                raise ManagerResponseValidationError("bad manager response")

            return fail().__await__()

    events = []
    agent = object.__new__(MobileAgent)
    agent.shared_state = SimpleNamespace(
        step_number=0,
        drain_user_messages=lambda: [],
    )
    agent.config = SimpleNamespace(agent=SimpleNamespace(max_steps=5))
    agent.manager_agent = SimpleNamespace(run=lambda: FakeHandler())
    agent.handle_stream_event = lambda nested_ev, ctx: None
    ctx = SimpleNamespace(write_event_to_stream=events.append)

    result = _run(agent.run_manager(ctx, ManagerInputEvent()))

    assert isinstance(result, FinalizeEvent)
    assert result.success is False
    assert result.reason == "bad manager response"
    assert events == []


def test_droid_agent_does_not_default_missing_success_to_true():
    agent = object.__new__(MobileAgent)
    agent.shared_state = SimpleNamespace(
        pending_user_messages=[],
        progress_summary="",
    )

    event = ManagerPlanEvent(
        plan="",
        current_subgoal="",
        thought="done",
        answer="Done, but success is missing.",
        success=None,
    )

    result = _run(agent.handle_manager_plan(SimpleNamespace(), event))

    assert isinstance(result, FinalizeEvent)
    assert result.success is False
    assert result.reason == "Done, but success is missing."


def test_manager_prompt_contracts_require_exactly_one_terminal_form():
    prompt_dir = Path("mobilerun/config/prompts/manager")
    for prompt_name in ("system.jinja2", "rev1.jinja2", "stateless.jinja2"):
        prompt = (prompt_dir / prompt_name).read_text()
        assert "exactly one" in prompt.lower()
        assert (
            "both <plan>" in prompt.lower() or "do not include <plan>" in prompt.lower()
        )

    trained = (prompt_dir / "trained.jinja2").read_text()
    assert "both <plan>" not in trained.lower()
