import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, TextBlock, ThinkingBlock
from llama_index.core.llms.mock import MockLLM
from llama_index.core.workflow import Context, StartEvent
from selector_helpers import StubSelector

from mobilerun import MobileAgent, MobileConfig
from mobilerun.agent.action_context import ActionContext
from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.common.constants import LLM_HISTORY_LIMIT
from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.executor.executor_agent import ExecutorAgent
from mobilerun.agent.fast_agent.events import FastAgentEndEvent, FastAgentInputEvent
from mobilerun.agent.fast_agent.fast_agent import FastAgent
from mobilerun.agent.fast_agent.xml_parser import ToolCall, ToolCallParseStatus, format_tool_calls
from mobilerun.agent.selectors import (
    ActionCandidate,
    ActionSelection,
    Decision,
    DecisionSelector,
    FallbackSelector,
    LlamaIndexLLMSelector,
    SelectionRequest,
)
from mobilerun.agent.tool_registry import ToolRegistry
from mobilerun.agent.utils.actions import complete
from mobilerun.agent.utils.prompt_resolver import PromptResolver
from mobilerun.cli.event_handler import EventHandler
from mobilerun.config_manager import AgentConfig
from mobilerun.tools.formatters import IndexedFormatter
from mobilerun.tools.ui.provider import StateProvider
from mobilerun.tools.ui.state import UIState
from mobilerun.tools.ui.stealth_state import StealthUIState


def _state(elements=None, cls=UIState):
    return cls(
        elements=elements if elements is not None else [
            {"index": 1, "text": "Alice", "bounds": "0,0,40,40"},
            {"index": 2, "text": "Bob", "bounds": "0,50,40,90"},
        ],
        formatted_text="Contacts", focused_text="", phone_state={},
        screen_width=100, screen_height=100,
    )


class _Provider(StateProvider):
    def __init__(self, state):
        self.state = state

    async def get_state(self):
        return self.state


def _dependencies():
    shared = MobileAgentState(instruction="Select the remembered contact")
    config = AgentConfig(max_steps=150, after_sleep_action=0, streaming=False)
    registry = ToolRegistry()
    click = AsyncMock(return_value=ActionResult(success=True, summary="Selected"))
    # Match the production schema so LLM XML responses exercise numeric coercion.
    registry.register("click", click, {"index": {"type": "number"}}, "Click")
    registry.register("complete", complete, {
        "success": {"type": "boolean"}, "message": {"type": "string"},
    }, "Complete")
    provider = _Provider(_state())
    action_ctx = ActionContext(SimpleNamespace(), provider.state, shared, provider)
    return config, registry, action_ctx, shared, provider, click


def _model(result):
    return SimpleNamespace(decide=AsyncMock(return_value=result))


def _request(**kwargs):
    values = dict(tools={"click": {}}, options=(
        ActionCandidate(ToolCall("click", {"index": 2}), "Click Bob"),
    ))
    values.update(kwargs)
    return SelectionRequest("Select", **values)


def test_llm_selector_preserves_native_llm_messages_and_response():
    async def run():
        llm = MockLLM()
        selector = LlamaIndexLLMSelector(llm)
        message = ChatMessage(role="assistant", blocks=[
            ThinkingBlock(content="private"),
            TextBlock(text='Tap Bob\n<function_calls><invoke name="click">'
                      '<parameter name="index">2</parameter></invoke></function_calls>'),
        ])
        response = ChatResponse(message=message)
        request = SelectionRequest("Select", messages=[ChatMessage(role="user", content="Goal")],
                                   param_types={"index": "number"}, stream=True)
        with patch("mobilerun.agent.selectors.llama_index_llm.acall_with_retries", new_callable=AsyncMock) as call:
            call.return_value = response
            selection = await selector.select(request)
        assert selector.llm is llm
        assert selection.response is response
        assert selection.response.message is message
        assert selection.actions == [ToolCall("click", {"index": 2})]
        assert selection.parse_result.status == ToolCallParseStatus.VALID
        call.assert_awaited_once_with(llm, request.messages, stream=True)

    asyncio.run(run())


@pytest.mark.parametrize("raw", ['{"action":"click","index":2}', 'not JSON'])
def test_executor_llm_selector_keeps_original_action_json(raw):
    async def run():
        response = ChatResponse(message=ChatMessage(role="assistant", content=(
            f"### Thought\nChoose Bob\n### Action\n{raw}\n### Description\nTap contact"
        )))
        with patch("mobilerun.agent.selectors.llama_index_llm.acall_with_retries", new_callable=AsyncMock) as call:
            call.return_value = response
            selection = await LlamaIndexLLMSelector(MockLLM()).select(SelectionRequest(
                "Select", protocol="executor", messages=[ChatMessage(role="user", content="Select")]
            ))
        assert selection.action_json == raw
        assert selection.thought == "Choose Bob"
        assert bool(selection.actions) == raw.startswith("{")

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["success", "empty", "parse_error"])
def test_executor_preserves_llm_logging_levels_messages_and_order(caplog, monkeypatch, outcome):
    async def run():
        config, registry, action_ctx, shared, _, _ = _dependencies()
        shared.last_thought = "Previous thought"
        agent = ExecutorAgent(
            LlamaIndexLLMSelector(SimpleNamespace(class_name=lambda: "MockLLM")),
            registry, action_ctx, shared, config,
        )
        ctx = Context(agent)
        await ctx.store.set("executor_messages", [ChatMessage(role="user", content="Select")])
        text = '### Thought\nChoose Bob\n### Action\n{"action":"click","index":2}\n### Description\nTap Bob'

        async def infer(*args, **kwargs):
            if outcome == "empty":
                raise ValueError("Empty response content")
            logging.getLogger("mobilerun").info(text)
            return ChatResponse(message=ChatMessage(role="assistant", content=text))

        from mobilerun.agent.executor.events import ExecutorContextEvent

        caplog.clear()
        with (
            patch.object(Context, "write_event_to_stream"),
            patch("mobilerun.agent.selectors.llama_index_llm.acall_with_retries", new=infer),
            patch("mobilerun.agent.selectors.llama_index_llm.get_usage_from_response",
                  side_effect=ValueError("No usage")),
        ):
            response = await agent.get_response(ctx, ExecutorContextEvent(subgoal="Select"))
            action = await agent.process_response(ctx, response)

        expected = [
            ("DEBUG", "Executor getting LLM response..."),
            ("INFO", "Executor response:"),
        ]
        if outcome == "empty":
            expected.append(("WARNING", "Executor LLM returned empty response: Empty response content"))
        else:
            expected.extend([("INFO", text), ("WARNING", "Could not get usage: No usage")])
        expected.append(("DEBUG", "⚙️ Processing executor response..."))
        if outcome == "parse_error":
            expected.append(("ERROR", "❌ Failed to parse executor response: bad format"))
            assert shared.last_thought == "Previous thought"
            assert action.description == "Invalid response format from LLM"
        records = [record for record in caplog.records if record.name == "mobilerun"]
        assert [(record.levelname, record.getMessage()) for record in records] == expected
        assert records[1].color == "green"

    monkeypatch.setattr(logging.getLogger("mobilerun"), "propagate", True)
    caplog.set_level(logging.DEBUG, logger="mobilerun")
    if outcome == "parse_error":
        monkeypatch.setattr(
            "mobilerun.agent.executor.prompts.parse_executor_response",
            Mock(side_effect=ValueError("bad format")),
        )
    asyncio.run(run())


@pytest.mark.parametrize("external", [False, True])
def test_mobile_agent_preserves_initialization_logging(caplog, monkeypatch, external):
    from mobilerun.agent.droid import droid_agent

    config = MobileConfig()
    if external:
        config.agent.name = "external-agent"
    monkeypatch.setattr(logging.getLogger("mobilerun"), "propagate", True)
    caplog.set_level(logging.DEBUG, logger="mobilerun")
    monkeypatch.setattr(droid_agent, "load_agent_llms", Mock(return_value={
        "fast_agent": MockLLM(), "app_opener": MockLLM(),
    }))
    MobileAgent("Select", config=config)
    expected = (
        "🔄 Using external agent: external-agent" if external
        else "🔄 Loading LLMs from config (llms not provided)..."
    )
    assert expected in caplog.messages
    assert caplog.messages.index(expected) < caplog.messages.index("✅ MobileAgent initialized successfully.")


def test_decision_filters_disabled_tools_and_preserves_request_context():
    async def run():
        request = SelectionRequest(
            "Select", state=_state(), memory="Bob", screenshot=b"image",
            history=[ChatMessage(role="user", content="Earlier result")],
            tools={"click": {}}, options=(
                ActionCandidate(ToolCall("long_press", {"index": 1}), "Hold Alice"),
                ActionCandidate(ToolCall("click", {"index": 2}), "Click Bob"),
            ),
        )
        model = _model(Decision(0, thought="Remembered Bob"))
        selection = await DecisionSelector(model).select(request)
        supplied = model.decide.call_args.args[0]
        assert supplied.options == request.options[1:]
        assert supplied.memory == "Bob"
        assert supplied.history is request.history
        assert supplied.screenshot == b"image"
        assert selection.actions == [ToolCall("click", {"index": 2})]
        assert selection.description == "Click Bob"
        assert len(request.options) == 2

    asyncio.run(run())


@pytest.mark.parametrize("index", [-1, 1, True, "0", 0.5])
def test_decision_rejects_invalid_indices(index):
    request = SelectionRequest("Select", tools={"click": {}}, options=(
        ActionCandidate(ToolCall("click", {"index": 2}), "Click Bob"),
    ))
    with pytest.raises(ValueError, match="option index"):
        asyncio.run(DecisionSelector(_model(Decision(index))).select(request))


@pytest.mark.parametrize("decision", [None, Decision()])
def test_decision_can_abstain(decision):
    assert asyncio.run(DecisionSelector(_model(decision)).select(_request())) is None


@pytest.mark.parametrize("success", [True, False])
def test_decision_only_fast_agent_executes_and_completes(success):
    async def run():
        config, registry, action_ctx, shared, provider, click = _dependencies()
        model = _model(None)
        model.decide.side_effect = [Decision(1, thought="Choose Bob"), Decision(
            success=success, thought="Checked the result", message="Finished",
        )]
        agent = FastAgent(
            action_selector=DecisionSelector(model), agent_config=config,
            registry=registry, action_ctx=action_ctx, state_provider=provider, shared_state=shared,
        )
        result = await agent.run(input="Select Bob")
        assert result == {"success": success, "reason": "Finished", "tool_call_count": 2}
        click.assert_awaited_once_with(index=2, ctx=action_ctx)
        assert model.decide.call_args.args[0].goal == "Select Bob"
        assert "Selected" in str(model.decide.call_args.args[0].history)

    asyncio.run(run())


@pytest.mark.parametrize("exit_path", [
    "complete_success", "complete_failure", "max_steps", "malformed", "abstain",
])
def test_fast_agent_streams_and_logs_terminal_result_once(monkeypatch, caplog, exit_path):
    async def run():
        config, registry, action_ctx, shared, provider, _ = _dependencies()
        if exit_path.startswith("complete_"):
            selector = DecisionSelector(_model(Decision(
                success=exit_path == "complete_success", message="Finished",
            )))
        elif exit_path == "malformed":
            selector = LlamaIndexLLMSelector(MockLLM())
        else:
            selector = StubSelector(select=AsyncMock(return_value=None))
            if exit_path == "max_steps":
                config.max_steps = 0
        agent = FastAgent(
            action_selector=selector, agent_config=config, registry=registry,
            action_ctx=action_ctx, state_provider=provider, shared_state=shared,
        )
        events = []
        event_handler = EventHandler()
        malformed = ChatResponse(message=ChatMessage(
            role="assistant", content='<function_calls><invoke name="click">',
        ))
        with patch(
            "mobilerun.agent.selectors.llama_index_llm.acall_with_retries",
            new=AsyncMock(return_value=malformed),
        ):
            handler = agent.run(input="Select Bob")
            async for event in handler.stream_events():
                events.append(event)
                event_handler.handle(event)
            result = await handler

        terminal_events = [event for event in events if isinstance(event, FastAgentEndEvent)]
        assert len(terminal_events) == 1
        event = terminal_events[0]
        assert event.success is (exit_path == "complete_success")
        assert event.reason == result["reason"]
        assert event.tool_call_count == result["tool_call_count"]

    monkeypatch.setattr(logging.getLogger("mobilerun"), "propagate", True)
    with caplog.at_level(logging.DEBUG, logger="mobilerun"):
        asyncio.run(run())
    assert len([record for record in caplog.records if record.message.startswith("■ ")]) == 1


def test_executor_selects_without_llm_and_receives_memory():
    async def run():
        config, registry, action_ctx, shared, _, click = _dependencies()
        shared.agent_memory = "Bob"
        model = _model(Decision(1, thought="Remembered contact"))
        agent = ExecutorAgent(
            action_selector=DecisionSelector(model), agent_config=config,
            registry=registry, action_ctx=action_ctx, shared_state=shared,
        )
        result = await agent.run(subgoal="Select the remembered contact")
        assert result["outcome"] is True
        assert result["action"] == {"action": "click", "index": 2}
        request = model.decide.call_args.args[0]
        assert request.memory == "Bob"
        assert "complete" not in request.tools
        click.assert_awaited_once_with(index=2, ctx=action_ctx)

    asyncio.run(run())


@pytest.mark.parametrize("agent_kind", ["fast_agent", "executor"])
@pytest.mark.parametrize("prompt_source", ["default", "file", "inline"])
@pytest.mark.parametrize("fallback", [False, True])
def test_workflows_forward_decision_prompts(tmp_path, agent_kind, prompt_source, fallback):
    async def run():
        config, registry, action_ctx, shared, provider, _ = _dependencies()
        model = _model(Decision(0))
        selector = DecisionSelector(model)
        if fallback:
            selector = FallbackSelector(
                StubSelector(select=AsyncMock(return_value=None)), [selector],
            )
        prompt = tmp_path / "decision.jinja2"
        prompts = {}
        if prompt_source != "default":
            config.decision.system_prompt = str(prompt)
            template = "{{ goal }}; typing={{ can_type }}; protocol={{ protocol }}"
            if prompt_source == "file":
                prompt.write_text(template, encoding="utf-8")
            else:
                prompts["decision_system"] = template
        resolver = PromptResolver(custom_prompts=prompts)
        kwargs = dict(
            action_selector=selector, agent_config=config, registry=registry,
            action_ctx=action_ctx, shared_state=shared, prompt_resolver=resolver,
        )
        if agent_kind == "fast_agent":
            agent = FastAgent(**kwargs, state_provider=provider)
            agent.system_prompt = await agent._build_system_prompt()
            shared.message_history = [ChatMessage(role="user", content="Select Bob")]
            ctx = Context(agent)
            with patch.object(Context, "write_event_to_stream"):
                await ctx.store.set("selection_goal", "Select Bob")
                await agent.handle_llm_input(ctx, FastAgentInputEvent())
        else:
            agent = ExecutorAgent(**kwargs)
            ctx = Context(agent)
            with patch.object(Context, "write_event_to_stream"):
                event = await agent.prepare_context(ctx, StartEvent(subgoal="Select Bob"))
                await agent.get_response(ctx, event)

        request = model.decide.call_args.args[0]
        if prompt_source == "default":
            assert "You cannot write arbitrary text" in request.instructions
        else:
            protocol = "xml" if agent_kind == "fast_agent" else "executor"
            assert request.instructions == f"Select Bob; typing=False; protocol={protocol}"

    asyncio.run(run())


@pytest.mark.parametrize("agent_kind", ["fast_agent", "executor"])
@pytest.mark.parametrize("prompt_source", ["file", "inline"])
def test_llm_fallback_uses_its_own_prompt_and_executes_once(tmp_path, agent_kind, prompt_source):
    async def run():
        config, registry, action_ctx, shared, provider, click = _dependencies()
        model = _model(None)
        selector = FallbackSelector(DecisionSelector(model), [LlamaIndexLLMSelector(MockLLM())])
        prompt_path = tmp_path / "llm.jinja2"
        getattr(config, agent_kind).system_prompt = str(prompt_path)
        template = "LLM ACTION PROMPT: {{ variables.contact }}"
        shared.custom_variables = {"contact": "Bob"}
        prompts = {"decision_system": "DECISION ONLY: {{ goal }}"}
        if prompt_source == "file":
            prompt_path.write_text(template, encoding="utf-8")
        else:
            # Inline prompts must work even when the configured file is absent.
            prompts[f"{agent_kind}_system"] = template
        actions = [ToolCall("click", {"index": 2})]
        if agent_kind == "fast_agent":
            actions.append(ToolCall("complete", {"success": True, "message": "Finished"}))
        responses = []
        for action in actions:
            text = format_tool_calls([action]) if agent_kind == "fast_agent" else (
                '### Thought\nPick Bob\n### Action\n{"action":"click","index":2}'
                '\n### Description\nTap Bob'
            )
            responses.append(ChatResponse(message=ChatMessage(role="assistant", content=text)))
        kwargs = dict(
            action_selector=selector, agent_config=config, registry=registry,
            action_ctx=action_ctx, shared_state=shared,
            prompt_resolver=PromptResolver(custom_prompts=prompts),
        )

        with patch(
            "mobilerun.agent.selectors.llama_index_llm.acall_with_retries",
            new_callable=AsyncMock,
        ) as infer:
            infer.side_effect = responses
            if agent_kind == "fast_agent":
                result = await FastAgent(**kwargs, state_provider=provider).run(input="Select Bob")
                assert result["success"]
            else:
                result = await ExecutorAgent(**kwargs).run(subgoal="Select Bob")
                assert result["outcome"]

        click.assert_awaited_once_with(index=2, ctx=action_ctx)
        assert infer.await_count == len(actions)
        assert model.decide.await_count == len(actions)
        for call in infer.call_args_list:
            messages = call.args[1]
            assert messages[0].content == "LLM ACTION PROMPT: Bob"
            assert "DECISION ONLY" not in str(messages)
        for call in model.decide.call_args_list:
            assert call.args[0].instructions == "DECISION ONLY: Select Bob"

    asyncio.run(run())


def test_decision_memory_and_goal_survive_history_truncation():
    async def run():
        config, registry, action_ctx, shared, provider, _ = _dependencies()
        shared.agent_memory = "Bob"
        shared.step_number = LLM_HISTORY_LIMIT + 1
        shared.message_history = [
            ChatMessage(role="user", content="Select the remembered contact"),
            ChatMessage(role="assistant", content="<add_memory>Bob</add_memory>"),
        ] + [ChatMessage(role="user", content=f"Result {i}") for i in range(LLM_HISTORY_LIMIT * 2)]

        async def decide(request):
            assert request.memory == "Bob"
            assert "Bob" not in str(request.history)
            assert len(request.history) == LLM_HISTORY_LIMIT * 2
            assert request.goal == "Select the remembered contact"
            return Decision(next(i for i, option in enumerate(request.options)
                                 if option.description == f"Click {request.memory}"))

        agent = FastAgent(
            action_selector=DecisionSelector(SimpleNamespace(decide=decide)),
            agent_config=config, registry=registry, action_ctx=action_ctx,
            state_provider=provider, shared_state=shared,
        )
        agent.system_prompt = await agent._build_system_prompt()
        ctx = Context(agent)
        with patch.object(Context, "write_event_to_stream"):
            await agent.handle_llm_input(ctx, FastAgentInputEvent())
        assert await ctx.store.get("pending_tool_calls") == [ToolCall("click", {"index": 2})]

    asyncio.run(run())


@pytest.mark.parametrize("primary_result", [None, ActionSelection(), ValueError("bad")])
def test_fallback_composes_selection_without_execution(primary_result):
    async def run():
        primary = StubSelector(select=AsyncMock())
        if isinstance(primary_result, Exception):
            primary.select.side_effect = primary_result
        else:
            primary.select.return_value = primary_result
        expected = ActionSelection([ToolCall("click", {"index": 2})])
        fallback = StubSelector(select=AsyncMock(return_value=expected))
        request = SelectionRequest("Select")
        assert await FallbackSelector(primary, [fallback]).select(request) is expected
        fallback.select.assert_awaited_once_with(request)

    asyncio.run(run())


def test_successful_selection_does_not_call_fallback():
    result = ActionSelection([ToolCall("click", {"index": 2})])
    primary = StubSelector(select=AsyncMock(return_value=result))
    fallback = StubSelector(select=AsyncMock())
    assert asyncio.run(FallbackSelector(primary, [fallback]).select(SelectionRequest("Select"))) is result
    fallback.select.assert_not_called()


def test_cancellation_is_not_swallowed_by_fallback():
    primary = StubSelector(select=AsyncMock(side_effect=asyncio.CancelledError()))
    fallback = StubSelector(select=AsyncMock())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(FallbackSelector(primary, [fallback]).select(SelectionRequest("Select")))
    fallback.select.assert_not_called()


def test_multiple_fallbacks_run_in_order_and_stop_after_selection():
    async def run():
        visited = []
        result = ActionSelection([ToolCall("click", {"index": 2})])

        def selector(name, outcome):
            async def select(request):
                visited.append(name)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

            return StubSelector(select=AsyncMock(side_effect=select))

        primary = selector("primary", None)
        fallbacks = [
            selector("empty", ActionSelection()),
            selector("error", ValueError("bad selection")),
            selector("chosen", result),
            selector("unused", result),
        ]
        request = _request()
        assert await FallbackSelector(primary, fallbacks).select(request) is result
        assert visited == ["primary", "empty", "error", "chosen"]
        for fallback in fallbacks[:3]:
            fallback.select.assert_awaited_once_with(request)
        fallbacks[-1].select.assert_not_called()

    asyncio.run(run())


@pytest.mark.parametrize("result", [None, ActionSelection()])
def test_exhausted_fallback_chain_returns_last_result(result):
    primary = StubSelector(select=AsyncMock(return_value=None))
    fallbacks = [
        StubSelector(select=AsyncMock(return_value=ActionSelection())),
        StubSelector(select=AsyncMock(return_value=result)),
    ]
    assert asyncio.run(FallbackSelector(primary, fallbacks).select(_request())) is result


def test_final_fallback_exception_propagates():
    error = ValueError("No selector succeeded")
    primary = StubSelector(select=AsyncMock(return_value=None))
    fallbacks = [
        StubSelector(select=AsyncMock(side_effect=ValueError("Try next"))),
        StubSelector(select=AsyncMock(side_effect=error)),
    ]
    with pytest.raises(ValueError) as raised:
        asyncio.run(FallbackSelector(primary, fallbacks).select(_request()))
    assert raised.value is error


def test_unconfigured_intermediate_exception_propagates():
    primary = StubSelector(select=AsyncMock(return_value=None))
    failing = StubSelector(select=AsyncMock(side_effect=RuntimeError("Stop")))
    unused = StubSelector(select=AsyncMock())
    with pytest.raises(RuntimeError, match="Stop"):
        asyncio.run(FallbackSelector(
            primary, [failing, unused], errors=(ValueError,)
        ).select(_request()))
    unused.select.assert_not_called()


@pytest.mark.parametrize("result", [None, ActionSelection([ToolCall("click", {"index": 2})])])
def test_fallback_chain_allows_no_fallbacks(result):
    primary = StubSelector(select=AsyncMock(return_value=result))
    assert asyncio.run(FallbackSelector(primary, []).select(_request())) is result
    primary.select.assert_awaited_once()


def test_fallback_chain_collects_all_native_llms_in_order():
    llms = [MockLLM() for _ in range(3)]
    chain = FallbackSelector(
        LlamaIndexLLMSelector(llms[0]),
        [DecisionSelector(_model(None)), FallbackSelector(
            LlamaIndexLLMSelector(llms[1]), [LlamaIndexLLMSelector(llms[2])]
        )],
    )
    assert chain.llms == tuple(llms)
    assert all(actual is expected for actual, expected in zip(chain.llms, llms, strict=True))


def test_completion_cannot_be_selected_by_executor():
    with pytest.raises(ValueError, match="Completion is not available"):
        asyncio.run(DecisionSelector(_model(Decision(success=True))).select(
            _request()
        ))


def test_provider_candidates_are_deterministic_and_respect_element_flags():
    state = _state([
        {"index": 1, "text": "Open", "bounds": "0,0,40,40", "isLongClickable": True},
        {"index": 2, "bounds": "0,50,40,90", "isEnabled": False},
        {"index": 3, "bounds": "0,50,40,90", "isVisibleToUser": False},
        {"index": 4, "bounds": "0,50,40,90", "isClickable": False},
        {"index": 5, "bounds": "bad"},
        {"index": 6, "bounds": "200,200,240,240"},
        {"index": 7, "bounds": "0,0,0,0"},
    ], cls=StealthUIState)
    provider = _Provider(state)
    with patch.object(StealthUIState, "get_element_coords", side_effect=AssertionError("Random geometry")):
        expected = provider.get_action_candidates(state)
        assert provider.get_action_candidates(state) == expected
    assert [option.action for option in expected] == [
        ToolCall("click", {"index": 1}), ToolCall("long_press", {"index": 1}),
    ]


def test_formatter_preserves_interaction_flags():
    node = {"bounds": "0,0,40,40", "isClickable": False, "isLongClickable": True,
            "isEnabled": False, "isVisibleToUser": False}
    element = IndexedFormatter().format(node, {})[2][0]
    for key in ("isClickable", "isLongClickable", "isEnabled", "isVisibleToUser"):
        assert element[key] is node[key]


@pytest.mark.parametrize("agent_class", [ExecutorAgent, FastAgent])
def test_agent_requires_a_selection_dependency(agent_class):
    with pytest.raises(ValueError, match="action_selector or llm is required"):
        agent_class()


@pytest.mark.parametrize("agent_class", [ExecutorAgent, FastAgent])
@pytest.mark.parametrize("explicit_selector", [False, True])
def test_agents_accept_legacy_llm_from_kwargs(monkeypatch, caplog, agent_class, explicit_selector):
    config, registry, action_ctx, shared, provider, _ = _dependencies()
    llm = MockLLM()
    selector = DecisionSelector(_model(Decision(0))) if explicit_selector else None
    kwargs = dict(
        llm=llm, agent_config=config, registry=registry, action_ctx=action_ctx,
        shared_state=shared, verbose=True,
    )
    if explicit_selector:
        kwargs["action_selector"] = selector
    if agent_class is FastAgent:
        kwargs["state_provider"] = provider
    monkeypatch.setattr(logging.getLogger("mobilerun"), "propagate", True)

    with caplog.at_level(logging.DEBUG, logger="mobilerun"):
        agent = agent_class(**kwargs)

    if explicit_selector:
        assert agent.action_selector is selector
    else:
        assert isinstance(agent.action_selector, LlamaIndexLLMSelector)
        assert agent.action_selector.llm is llm
    deprecations = [record for record in caplog.records if "deprecated" in record.message]
    assert len(deprecations) == 1
    assert deprecations[0].levelno == logging.DEBUG
    assert f"{agent_class.__name__}(llm=...) is deprecated" in deprecations[0].message


@pytest.mark.parametrize("agent_class", [ExecutorAgent, FastAgent])
def test_agents_require_non_null_legacy_llm(agent_class):
    with pytest.raises(ValueError, match="action_selector or llm is required"):
        agent_class(llm=None)


@pytest.mark.parametrize("options", [None, ()])
def test_decision_requires_options_before_calling_model(options):
    model = _model(Decision(0))
    with pytest.raises(ValueError, match="non-empty options"):
        asyncio.run(DecisionSelector(model).select(_request(options=options)))
    model.decide.assert_not_called()


def test_decision_rejects_options_for_disabled_tools():
    model = _model(Decision(0))
    with pytest.raises(ValueError, match="available tools"):
        asyncio.run(DecisionSelector(model).select(_request(tools={})))
    model.decide.assert_not_called()


def test_executor_does_not_hide_missing_decision_options():
    async def run():
        config, registry, action_ctx, shared, _, _ = _dependencies()
        action_ctx.ui = _state([])
        model = _model(Decision(0))
        agent = ExecutorAgent(
            DecisionSelector(model), registry, action_ctx, shared, config,
        )
        ctx = Context(agent)
        await ctx.store.set("executor_messages", [ChatMessage(role="user", content="Select")])
        from mobilerun.agent.executor.events import ExecutorContextEvent
        with pytest.raises(ValueError, match="non-empty options"):
            await agent.get_response(ctx, ExecutorContextEvent(subgoal="Select"))
        model.decide.assert_not_called()

    asyncio.run(run())


def test_decision_timeout_can_fall_back():
    async def run():
        cancelled = asyncio.Event()

        async def decide(request):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        primary = DecisionSelector(SimpleNamespace(decide=decide), timeout=0.05)
        result = ActionSelection([ToolCall("click", {"index": 2})])
        fallback = StubSelector(select=AsyncMock(return_value=result))
        assert await FallbackSelector(primary, [fallback]).select(_request()) is result
        assert cancelled.is_set(), "A timed-out provider must not keep running behind the fallback"
        fallback.select.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize("result", [{}, 0, "choice"])
def test_decision_rejects_invalid_result_types(result):
    with pytest.raises(TypeError, match="must return Decision"):
        asyncio.run(DecisionSelector(_model(result)).select(_request()))


@pytest.mark.parametrize("decision,error", [
    (Decision(success=1), "must be a boolean"),
    (Decision(success="true"), "must be a boolean"),
    (Decision(option_index=0, success=True), "Completion is not available"),
])
def test_decision_rejects_ambiguous_or_non_boolean_completion(decision, error):
    # Completion and option selection are mutually exclusive, even when both
    # are individually available. Python's bool/int overlap must not bypass it.
    request = _request(tools={"click": {}, "complete": {}})
    with pytest.raises(ValueError, match=error):
        asyncio.run(DecisionSelector(_model(decision)).select(request))
