import asyncio
from types import SimpleNamespace

import pytest
from llama_index.core.base.llms.types import ChatMessage

from mobilerun.agent.droid.droid_agent import MobileAgent
from mobilerun.agent.droid.events import (
    ExecutorInputEvent,
    FinalizeEvent,
    ManagerInputEvent,
    ManagerPlanEvent,
)
from mobilerun.agent.manager.manager_agent import ManagerAgent
from mobilerun.agent.manager.prompts import (
    ManagerResponseValidationError,
    parse_manager_response,
    validate_manager_response,
)
from mobilerun.agent.manager.stateless_manager_agent import StatelessManagerAgent


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(message=SimpleNamespace(content=content))


def _run(coro):
    return asyncio.run(coro)


def _stateful_manager() -> ManagerAgent:
    agent = object.__new__(ManagerAgent)
    agent.llm = SimpleNamespace()
    agent.agent_config = SimpleNamespace(streaming=False)
    return agent


def _stateless_manager() -> StatelessManagerAgent:
    agent = object.__new__(StatelessManagerAgent)
    agent.llm = SimpleNamespace()
    return agent


@pytest.mark.parametrize(
    "response",
    [
        "<plan>1. Open Settings</plan>",
        "<thought>Need to inspect Settings.</thought><plan>1. Open Settings</plan>",
        '<request_accomplished success="true">Done</request_accomplished>',
        '<request_accomplished success="false">Blocked</request_accomplished>',
        '<answer success="true">Done</answer>',
        '<answer success="false">Blocked</answer>',
    ],
)
def test_validate_accepts_one_nonempty_result_with_optional_metadata(response):
    validation = validate_manager_response(parse_manager_response(response))

    assert validation.is_valid


@pytest.mark.parametrize(
    ("response", "field", "expected"),
    [
        (
            "<plan>1. Inspect the `<script>` element</plan>",
            "plan",
            "1. Inspect the `<script>` element",
        ),
        (
            '<answer success="true">Found `<script>` markup.</answer>',
            "answer",
            "Found `<script>` markup.",
        ),
        (
            "<plan>1. Inspect <em>Android 16</em> text</plan>",
            "plan",
            "1. Inspect <em>Android 16</em> text",
        ),
        (
            '<answer success="true">Found <em>Android 16</em>.</answer>',
            "answer",
            "Found <em>Android 16</em>.",
        ),
    ],
)
def test_validate_accepts_markup_as_top_level_result_content(response, field, expected):
    parsed = parse_manager_response(response)

    assert validate_manager_response(parsed).is_valid
    assert parsed[field] == expected


def test_validate_preserves_paired_script_plan_content():
    parsed = parse_manager_response("<plan><script>tap Settings</script></plan>")

    assert validate_manager_response(parsed).is_valid
    assert parsed["plan"] == "<script>tap Settings</script>"
    assert parsed["current_subgoal"] == "<script>tap Settings</script>"


@pytest.mark.parametrize(
    "response",
    [
        "",
        "<plan></plan>",
        '<answer success="true"></answer>',
        "<plan>1. Read version</plan><answer success='true'>Done</answer>",
        "<plan>1. Read version</plan><plan>2. Finish</plan>",
        "<answer success='true'>Done</answer><answer success='false'>Blocked</answer>",
        "<answer>Done</answer>",
        "<answer success='maybe'>Done</answer>",
        "<answer success=true>Done</answer>",
        "<answer success='true' success='false'>Done</answer>",
        "<plan>1. Read <answer success='true'>Done</answer></plan>",
        '<request_accomplished success="true">Done',
    ],
)
def test_validate_rejects_unsafe_or_ambiguous_result_forms(response):
    validation = validate_manager_response(parse_manager_response(response))

    assert not validation.is_valid


@pytest.mark.parametrize(
    "response",
    [
        '<plan>1. Open Settings</plan><answer success="true"',
        '<plan>1. Open Settings</plan><request_accomplished success="true"',
        '<answer success="true">Done</answer><plan',
        '<request_accomplished success="true">Done</request_accomplished></answer',
    ],
)
def test_validate_rejects_trailing_unterminated_control_tags(response):
    parsed = parse_manager_response(response)

    assert not validate_manager_response(parsed).is_valid
    assert any("unterminated" in error for error in parsed["result_tag_errors"])


@pytest.mark.parametrize(
    "response",
    [
        "<plan>1. Inspect the `<script` token</plan>",
        "<plan>1. Open Settings</plan><plans",
        "<plan>1. Open Settings</plan><answerable",
    ],
)
def test_validate_does_not_treat_non_control_markup_as_truncated_result(response):
    assert validate_manager_response(parse_manager_response(response)).is_valid


@pytest.mark.parametrize(
    "response",
    [
        "<plan>1. Inspect <script><answer success='true'>Done</answer></plan>",
        "<plan>1. Inspect <thought>details</plan>",
        "<thought><plan>1. Open Settings</plan></thought>",
        '<add_memory><answer success="true">Done</answer></add_memory>',
        "<progress_summary><plan>1. Open Settings</plan></progress_summary>",
        "<div><plan>1. Open Settings</plan></div>",
    ],
)
def test_validate_rejects_results_nested_in_envelope_or_outer_markup(response):
    assert not validate_manager_response(parse_manager_response(response)).is_valid


def test_success_is_bound_to_the_matching_final_tag():
    parsed = parse_manager_response(
        "<request_accomplished reason='checked' success='TRUE'>"
        "Android 16 is installed.</request_accomplished>"
    )

    assert parsed["answer"] == "Android 16 is installed."
    assert parsed["success"] is True


def test_stateful_manager_repairs_a_mixed_response(monkeypatch):
    calls = []

    async def fake_acall(llm, messages, stream=False):
        calls.append((llm, messages, stream))
        return _response(
            '<request_accomplished success="true">Android 16.</request_accomplished>'
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

    assert output.startswith("<request_accomplished")
    assert len(calls) == 1


def test_stateless_manager_repairs_a_missing_success_attribute(monkeypatch):
    calls = []

    async def fake_acall(llm, messages):
        calls.append((llm, messages))
        return _response('<answer success="false">Blocked</answer>')

    monkeypatch.setattr(
        "mobilerun.agent.manager.stateless_manager_agent.acall_with_retries", fake_acall
    )

    output = _run(
        _stateless_manager()._validate_and_retry(
            [{"role": "user", "content": [{"text": "prompt"}]}],
            "<answer>Done</answer>",
        )
    )

    assert output == '<answer success="false">Blocked</answer>'
    assert len(calls) == 1


def test_stateful_manager_fails_closed_after_exhausted_repairs(monkeypatch):
    calls = []
    invalid_response = (
        "<plan>1. Read version</plan><answer success='true'>Done</answer>"
    )

    async def fake_acall(llm, messages, stream=False):
        calls.append((llm, messages, stream))
        return _response(invalid_response)

    monkeypatch.setattr(
        "mobilerun.agent.manager.manager_agent.acall_with_retries", fake_acall
    )

    with pytest.raises(ManagerResponseValidationError):
        _run(
            _stateful_manager()._validate_and_retry(
                [ChatMessage(role="user", content="prompt")], invalid_response
            )
        )

    assert len(calls) == 3


def test_stateless_manager_fails_closed_after_exhausted_repairs(monkeypatch):
    calls = []
    invalid_response = (
        "<plan>1. Read version</plan><answer success='true'>Done</answer>"
    )

    async def fake_acall(llm, messages):
        calls.append((llm, messages))
        return _response(invalid_response)

    monkeypatch.setattr(
        "mobilerun.agent.manager.stateless_manager_agent.acall_with_retries", fake_acall
    )

    with pytest.raises(ManagerResponseValidationError):
        _run(
            _stateless_manager()._validate_and_retry(
                [{"role": "user", "content": [{"text": "prompt"}]}],
                invalid_response,
            )
        )

    assert len(calls) == 3


def test_droid_agent_validation_failure_never_reaches_the_executor():
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
        terminal_abort=None,
        drain_user_messages=lambda: [],
    )
    agent.config = SimpleNamespace(agent=SimpleNamespace(max_steps=5))
    agent.manager_agent = SimpleNamespace(run=lambda: FakeHandler())
    agent.handle_stream_event = lambda *_: pytest.fail("executor path must not run")
    ctx = SimpleNamespace(write_event_to_stream=events.append)

    result = _run(agent.run_manager(ctx, ManagerInputEvent()))

    assert isinstance(result, FinalizeEvent)
    assert result.success is False
    assert events == []


@pytest.mark.parametrize("success", [None, False])
def test_unknown_or_false_final_success_never_becomes_successful(success):
    agent = object.__new__(MobileAgent)
    agent.shared_state = SimpleNamespace(
        pending_user_messages=[],
        progress_summary="",
    )

    result = _run(
        agent.handle_manager_plan(
            SimpleNamespace(),
            ManagerPlanEvent(
                plan="",
                current_subgoal="",
                thought="done",
                answer="Could not complete the task.",
                success=success,
            ),
        )
    )

    assert isinstance(result, FinalizeEvent)
    assert result.success is False


def test_valid_plan_routes_to_the_executor():
    parsed = parse_manager_response(
        "<thought>Need the About screen next.</thought><plan>1. Open Settings</plan>"
    )
    agent = object.__new__(MobileAgent)
    agent.shared_state = SimpleNamespace(
        pending_user_messages=[],
        progress_summary="",
    )

    result = _run(
        agent.handle_manager_plan(
            SimpleNamespace(),
            ManagerPlanEvent(
                plan=parsed["plan"],
                current_subgoal=parsed["current_subgoal"],
                thought=parsed["thought"],
                answer=parsed["answer"],
                success=parsed["success"],
            ),
        )
    )

    assert isinstance(result, ExecutorInputEvent)
    assert result.current_subgoal == "Open Settings"
