import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, ImageBlock, TextBlock
from llama_index.core.llms.mock import MockLLM
from selector_helpers import StubSelector

from mobilerun.agent.action_context import ActionContext
from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.executor.executor_agent import ExecutorAgent
from mobilerun.agent.fast_agent.fast_agent import FastAgent
from mobilerun.agent.fast_agent.xml_parser import ToolCall
from mobilerun.agent.selectors import (
    ActionCandidate,
    ActionSelection,
    Decision,
    DecisionSelector,
    DelegatingSelector,
    LlamaIndexLLMSelector,
    SelectionRequest,
)
from mobilerun.agent.tool_registry import ToolRegistry
from mobilerun.agent.usage import UsageResult
from mobilerun.agent.utils.actions import complete
from mobilerun.config_manager import AgentConfig
from mobilerun.tools.ui.provider import StateProvider
from mobilerun.tools.ui.state import UIState


def _request(protocol="xml", **kwargs):
    return SelectionRequest(
        "Select Bob", protocol=protocol,
        tools={"click": {"parameters": {"index": {"type": "integer"}}}},
        options=(ActionCandidate(ToolCall("click", {"index": 2}), "Click Bob"),),
        messages=[
            ChatMessage(role="system", content="Use the existing action format."),
            ChatMessage(role="user", blocks=[TextBlock(text="Select Bob"), ImageBlock(image=b"image")]),
        ],
        param_types={"index": "number"}, **kwargs,
    )


def _selector(result):
    return StubSelector(select=AsyncMock(return_value=result))


def _response(protocol, action):
    if protocol == "executor":
        import json

        text = (
            f"### Thought\nChoose action\n### Action\n"
            f"{json.dumps({'action': action.name, **action.parameters})}\n"
            "### Description\nChoose action"
        )
    else:
        from mobilerun.agent.fast_agent.xml_parser import format_tool_calls

        text = f"Choose action\n{format_tool_calls([action])}"
    return ChatResponse(message=ChatMessage(role="assistant", content=text))


@pytest.mark.parametrize("protocol", ["xml", "executor"])
def test_decision_model_can_explicitly_delegate_to_llm_without_mutating_request(protocol):
    async def run():
        async def decide(request):
            assert request.options[-1].action == ToolCall("delegate")
            assert request.tools["delegate"]["parameters"] == {}
            assert "delegate instead of declaring the goal impossible" in request.instructions
            return Decision(len(request.options) - 1)

        model = SimpleNamespace(decide=AsyncMock(side_effect=decide))
        request = _request(protocol, memory="Remember Bob", instructions="Keep my custom instructions.")
        original_messages = [message.model_copy(deep=True) for message in request.messages]
        selector = DelegatingSelector(DecisionSelector(model), LlamaIndexLLMSelector(MockLLM()))
        expected = ToolCall("click", {"index": 2})
        with patch("mobilerun.agent.selectors.llama_index_llm.acall_with_retries", new_callable=AsyncMock) as call:
            call.return_value = _response(protocol, expected)
            selection = await selector.select(request)

        assert selection.actions == [expected]
        call.assert_awaited_once_with(selector.delegate.llm, request.messages, stream=False)
        primary_request = model.decide.call_args.args[0]
        assert primary_request.instructions.startswith(request.instructions)
        assert primary_request.memory == request.memory
        assert "delegate" not in request.tools
        assert len(request.options) == 1
        assert request.messages == original_messages
        assert primary_request.messages[-1].blocks == request.messages[-1].blocks

    asyncio.run(run())


@pytest.mark.parametrize("protocol", ["xml", "executor"])
def test_llm_can_explicitly_delegate_to_decision_model(protocol):
    async def run():
        model = SimpleNamespace(decide=AsyncMock(return_value=Decision(0)))
        selector = DelegatingSelector(LlamaIndexLLMSelector(MockLLM()), DecisionSelector(model))
        request = _request(protocol)
        with patch("mobilerun.agent.selectors.llama_index_llm.acall_with_retries", new_callable=AsyncMock) as call:
            call.return_value = _response(protocol, ToolCall("delegate"))
            selection = await selector.select(request)

        assert selection.actions == [ToolCall("click", {"index": 2})]
        primary_messages = call.call_args.args[1]
        assert "Additional action: delegate()" in primary_messages[0].content
        assert "delegate" not in request.messages[0].content
        supplied = model.decide.call_args.args[0]
        assert supplied.options == request.options
        assert "delegate" not in supplied.tools
        assert "Additional action: delegate()" not in supplied.instructions

    asyncio.run(run())


def test_delegation_is_one_step_and_returns_to_primary_on_next_selection():
    async def run():
        direct = ActionSelection([ToolCall("click", {"index": 2})])
        primary = _selector(None)
        primary.select.side_effect = [ActionSelection([ToolCall("delegate")]), direct]
        delegated = ActionSelection([ToolCall("click", {"index": 1})])
        delegate = _selector(delegated)
        selector = DelegatingSelector(primary, delegate)
        request = _request()

        assert (await selector.select(request)).actions == delegated.actions
        assert await selector.select(request) is direct
        assert primary.select.await_count == 2
        delegate.select.assert_awaited_once_with(request)

    asyncio.run(run())


@pytest.mark.parametrize("result", [
    None,
    ActionSelection(),
    ActionSelection([ToolCall("click", {"index": 2})]),
    ActionSelection(action_json='{"action":"click","index":2}'),
    ActionSelection(action_json="not JSON"),
    ActionSelection(action_json='["delegate"]'),
])
def test_no_implicit_fallback_when_primary_does_not_delegate(result):
    primary = _selector(result)
    delegate = _selector(None)
    assert asyncio.run(DelegatingSelector(primary, delegate).select(_request())) is result
    delegate.select.assert_not_called()


@pytest.mark.parametrize("error", [ValueError("provider failed"), asyncio.CancelledError()])
def test_primary_errors_and_cancellation_propagate_without_delegating(error):
    primary = _selector(None)
    primary.select.side_effect = error
    delegate = _selector(None)
    with pytest.raises(type(error)):
        asyncio.run(DelegatingSelector(primary, delegate).select(_request()))
    delegate.select.assert_not_called()


@pytest.mark.parametrize("actions", [
    [ToolCall("click", {"index": 2}), ToolCall("delegate")],
    [ToolCall("delegate"), ToolCall("delegate")],
    [ToolCall("delegate", {"unexpected": "argument"})],
])
def test_delegation_cannot_be_batched_with_other_actions_or_take_parameters(actions):
    delegate = _selector(None)
    with pytest.raises(ValueError, match="only selected action"):
        asyncio.run(DelegatingSelector(_selector(ActionSelection(actions)), delegate).select(_request()))
    delegate.select.assert_not_called()


@pytest.mark.parametrize("collision", ["tools", "options"])
def test_reserved_delegate_name_does_not_overwrite_user_tools(collision):
    request = _request()
    if collision == "tools":
        request.tools["delegate"] = {"description": "An existing user tool"}
    else:
        request.options += (ActionCandidate(ToolCall("delegate"), "Existing choice"),)
    primary = _selector(None)
    with pytest.raises(ValueError, match="reserves the tool name"):
        asyncio.run(DelegatingSelector(primary, _selector(None)).select(request))
    primary.select.assert_not_called()


@pytest.mark.parametrize("delegated", [
    ActionSelection([ToolCall("delegate")]),
    ActionSelection(action_json='{"action":"delegate"}'),
])
def test_delegate_cannot_leak_virtual_action_to_execution(delegated):
    selector = DelegatingSelector(_selector(ActionSelection([ToolCall("delegate")])), _selector(delegated))
    with pytest.raises(ValueError, match="must select a real action"):
        asyncio.run(selector.select(_request()))


def test_json_only_custom_selector_can_delegate():
    expected = ActionSelection([ToolCall("click", {"index": 2})])
    selector = DelegatingSelector(
        _selector(ActionSelection(action_json='{"action":"delegate"}')), _selector(expected),
    )
    assert asyncio.run(selector.select(_request("executor"))).actions == expected.actions


def test_json_only_delegation_also_rejects_unexpected_parameters():
    delegate = _selector(None)
    selector = DelegatingSelector(
        _selector(ActionSelection(action_json='{"action":"delegate","unexpected":true}')), delegate,
    )
    with pytest.raises(ValueError, match="takes no parameters"):
        asyncio.run(selector.select(_request("executor")))
    delegate.select.assert_not_called()


def test_decision_model_can_delegate_even_without_device_candidates():
    model = SimpleNamespace(decide=AsyncMock(return_value=Decision(0)))
    expected = ActionSelection([ToolCall("click", {"index": 2})])
    selector = DelegatingSelector(DecisionSelector(model), _selector(expected))
    request = _request()
    request.options = None
    assert asyncio.run(selector.select(request)).actions == expected.actions
    assert model.decide.call_args.args[0].options == (
        ActionCandidate(ToolCall("delegate"), selector.description),
    )


@pytest.mark.parametrize("result", [None, ActionSelection()])
def test_delegate_abstention_or_empty_selection_is_returned(result):
    selector = DelegatingSelector(_selector(ActionSelection([ToolCall("delegate")])), _selector(result))
    selection = asyncio.run(selector.select(_request()))
    assert selection is None if result is None else selection.actions == []


def test_delegate_error_propagates_without_retry():
    delegate = _selector(None)
    delegate.select.side_effect = ValueError("delegate failed")
    selector = DelegatingSelector(_selector(ActionSelection([ToolCall("delegate")])), delegate)
    with pytest.raises(ValueError, match="delegate failed"):
        asyncio.run(selector.select(_request()))
    assert delegate.select.await_count == 1


@pytest.mark.parametrize("primary_usage", [False, True])
@pytest.mark.parametrize("delegate_usage", [False, True])
def test_delegation_preserves_usage_and_memory_without_mutating_selections(primary_usage, delegate_usage):
    usage = UsageResult(request_tokens=10, response_tokens=2, total_tokens=12, requests=1)
    primary = ActionSelection(
        [ToolCall("delegate")], thought="<add_memory>Bob is visible</add_memory>",
        memory_update="Need help", usage=usage if primary_usage else None,
    )
    delegated = ActionSelection(
        [ToolCall("click", {"index": 2})], memory_update="Selected Bob",
        usage=usage if delegate_usage else None,
    )
    result = asyncio.run(DelegatingSelector(_selector(primary), _selector(delegated)).select(_request()))
    count = int(primary_usage) + int(delegate_usage)
    assert result.usage == UsageResult(
        request_tokens=10 * count, response_tokens=2 * count, total_tokens=12 * count, requests=count,
    ) if count else result.usage is None
    assert result.memory_update == "Bob is visible\nNeed help\nSelected Bob"
    assert delegated.memory_update == "Selected Bob"
    assert usage.requests == 1


def test_delegation_exposes_both_selectors_vision_metadata():
    llms = [MockLLM(), MockLLM()]
    selector = DelegatingSelector(LlamaIndexLLMSelector(llms[0]), LlamaIndexLLMSelector(llms[1]))
    assert all(actual is expected for actual, expected in zip(selector.llms, llms, strict=True))
    assert not selector.requires_action_candidates
    assert selector.screenshot_max_side is None
    selector.primary = DecisionSelector(SimpleNamespace(decide=AsyncMock(), screenshot_max_side=1568))
    assert selector.requires_action_candidates
    assert selector.screenshot_max_side == 1568


@pytest.mark.parametrize("agent_kind", ["fast_agent", "executor"])
@pytest.mark.parametrize("direction", ["decision_to_llm", "llm_to_decision"])
def test_workflows_execute_only_delegated_device_actions(agent_kind, direction):
    async def run():
        shared = MobileAgentState(instruction="Select Bob")
        config = AgentConfig(after_sleep_action=0, streaming=False)
        registry = ToolRegistry()
        click = AsyncMock(return_value=ActionResult(success=True, summary="Selected Bob"))
        registry.register("click", click, {"index": {"type": "number"}}, "Click")
        registry.register("complete", complete, {
            "success": {"type": "boolean"}, "message": {"type": "string"},
        }, "Complete")
        state = UIState(
            elements=[{"index": 2, "text": "Bob", "bounds": "0,0,40,40"}],
            formatted_text="Bob", focused_text="", phone_state={}, screen_width=100, screen_height=100,
        )
        driver = SimpleNamespace(supported=set(), supported_buttons=set())
        provider = StateProvider(driver)
        provider.get_state = AsyncMock(return_value=state)
        action_ctx = ActionContext(driver, state, shared, provider)

        async def decide(request):
            if direction == "llm_to_decision":
                return Decision(0, thought="Pick Bob")
            if any("Selected Bob" in str(item) for item in request.history):
                return Decision(success=True, message="Finished")
            return Decision(next(i for i, option in enumerate(request.options) if option.action.name == "delegate"))

        model = SimpleNamespace(decide=AsyncMock(side_effect=decide))
        selector = DelegatingSelector(DecisionSelector(model), LlamaIndexLLMSelector(MockLLM()))
        if direction == "llm_to_decision":
            selector = DelegatingSelector(selector.delegate, selector.primary)
        kwargs = dict(
            action_selector=selector, agent_config=config, registry=registry,
            action_ctx=action_ctx, shared_state=shared,
        )
        protocol = "xml" if agent_kind == "fast_agent" else "executor"
        with patch("mobilerun.agent.selectors.llama_index_llm.acall_with_retries", new_callable=AsyncMock) as call:
            if direction == "decision_to_llm":
                call.return_value = _response(protocol, ToolCall("click", {"index": 2}))
            else:
                call.side_effect = [
                    _response(protocol, ToolCall("delegate")),
                    _response(protocol, ToolCall("complete", {"success": True, "message": "Finished"})),
                ]
            if agent_kind == "fast_agent":
                result = await FastAgent(**kwargs, state_provider=provider).run(input="Select Bob")
                assert result["success"]
                assert result["tool_call_count"] == 2  # click and complete; delegation is not a tool call.
                assert model.decide.await_count == (2 if direction == "decision_to_llm" else 1)
                assert not any('name="delegate"' in (message.content or "") for message in shared.message_history)
            else:
                result = await ExecutorAgent(**kwargs).run(subgoal="Select Bob")
                assert result["outcome"]
                assert result["action"] == {"action": "click", "index": 2}
            assert call.await_count == (2 if direction == "llm_to_decision" and agent_kind == "fast_agent" else 1)
        click.assert_awaited_once_with(index=2, ctx=action_ctx)
        assert "delegate" not in registry.tools

    asyncio.run(run())
