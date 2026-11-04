import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    TextBlock,
    ToolCallBlock,
)
from llama_index.llms.openai.utils import to_openai_message_dicts

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.fast_agent.fast_agent import FastAgent
from mobilerun.agent.selectors import LlamaIndexLLMSelector
from mobilerun.agent.tool_registry import ToolRegistry
from mobilerun.agent.utils.chat_utils import limit_history
from mobilerun.agent.utils.inference import acall_with_retries

FAILING_INDEX = 99
CLICK_PARAMETERS = {"type": "object", "properties": {"index": {"type": "number"}}}
CLICK_SPEC = {
    "type": "function",
    "name": "click",
    "description": "Tap an element.",
    "parameters": CLICK_PARAMETERS,
}


def _call(call_id, name, arguments):
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return ToolCallBlock(tool_call_id=call_id, tool_name=name, tool_kwargs=arguments)


def _reply(text, *calls):
    blocks = [TextBlock(text=text)] if text else []
    return ChatMessage(role="assistant", blocks=[*blocks, *calls])


def _complete(call_id="call_done"):
    return _reply(
        "Done.", _call(call_id, "complete", {"success": True, "message": "finished"})
    )


class _Responder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def __call__(self, _llm, messages, *, stream, **chat_kwargs):
        self.requests.append((list(messages), chat_kwargs))
        if not self.responses:
            raise AssertionError("FastAgent made an unexpected extra LLM request")
        return ChatResponse(message=self.responses.pop(0))


class _PromptResolver:
    def get_prompt(self, name):
        if name == "fast_agent_system":
            return "{% if native_tools %}native{% else %}xml{% endif %}"
        if name == "fast_agent_user":
            return "Goal: {{ goal }}"
        return None


class _StateProvider:
    requires_coordinate_tools = False

    def get_action_candidates(self, state):
        return ()

    async def get_state(self):
        return SimpleNamespace(
            formatted_text="Settings home",
            focused_text="",
            elements=[],
            phone_state={"packageName": "com.android.settings"},
        )


class _Registry:
    def __init__(self, shared_state):
        self.shared_state = shared_state
        self.executed = []
        self.tools = {"click": object(), "complete": object()}

    def get_tool_descriptions_xml(self):
        return '<tool name="click"/><tool name="complete"/>'

    def get_param_types(self):
        return {"index": "number", "success": "boolean"}

    def get_signatures(self):
        return {"click": {"parameters": {"index": {"type": "number"}}}, "complete": {}}

    def get_function_tool_specs(self):
        return [CLICK_SPEC]

    async def execute(self, name, parameters, _action_ctx, workflow_ctx=None):
        self.executed.append((name, parameters))
        if name == "click" and parameters.get("index") == FAILING_INDEX:
            return ActionResult(success=False, summary="No element at index 99")
        if name == "complete":
            await self.shared_state.complete(
                parameters.get("success", False),
                message=parameters.get("message", ""),
            )
        return ActionResult(success=True, summary=f"{name} executed")


def _run(
    responses,
    *,
    native=True,
    tool_calling="auto",
    parallel_tools=True,
    native_tool_api="responses",
    use_selector=False,
):
    async def _go():
        shared_state = MobileAgentState()
        registry = _Registry(shared_state)
        responder = _Responder(responses)
        llm = SimpleNamespace(
            supports_native_tool_calls=native, native_tool_api=native_tool_api
        )
        selection_kwargs = (
            {"action_selector": LlamaIndexLLMSelector(llm)}
            if use_selector else {"llm": llm}
        )
        agent = FastAgent(
            **selection_kwargs,
            agent_config=SimpleNamespace(
                fast_agent=SimpleNamespace(
                    vision=False,
                    parallel_tools=parallel_tools,
                    tool_calling=tool_calling,
                ),
                max_steps=20,
                streaming=False,
                after_sleep_action=0,
            ),
            registry=registry,
            action_ctx=SimpleNamespace(
                driver=SimpleNamespace(), credential_manager=None, ui=None
            ),
            state_provider=_StateProvider(),
            shared_state=shared_state,
            prompt_resolver=_PromptResolver(),
        )
        with (
            patch(
                "mobilerun.agent.selectors.llama_index_llm.acall_with_retries",
                new=responder,
            ),
            patch(
                "mobilerun.agent.selectors.llama_index_llm.get_usage_from_response",
                return_value=None,
            ),
        ):
            result = await agent.run(input="Exercise native tools")
        return result, responder, registry, shared_state

    return asyncio.run(_go())


@pytest.mark.parametrize("use_selector", [False, True])
def test_native_tools_are_sent_and_answered_with_tool_messages(use_selector):
    result, responder, registry, shared_state = _run(
        [_reply("I'll tap it.", _call("call_1", "click", {"index": 3})), _complete()],
        parallel_tools=False,
        use_selector=use_selector,
    )

    assert result["success"] is True
    assert registry.executed[0] == ("click", {"index": 3})

    first_messages, first_kwargs = responder.requests[0]
    assert first_messages[0].content == "native"
    assert first_kwargs == {
        "tools": [CLICK_SPEC],
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }

    second_messages = responder.requests[1][0]
    assistant, tool, user = second_messages[-3:]
    assert any(isinstance(b, ToolCallBlock) for b in assistant.blocks)
    assert tool.role == "tool"
    assert tool.additional_kwargs == {"tool_call_id": "call_1"}
    assert tool.content == "click executed"
    assert user.role == "user"
    assert "<device_state>\nSettings home\n</device_state>" in user.content
    assert "<function_results>" not in user.content


def test_tool_call_without_text_runs_without_a_reasoning_nudge():
    _, responder, registry, _ = _run(
        [_reply("", _call("call_1", "click", {"index": 3})), _complete()]
    )

    assert registry.executed[0] == ("click", {"index": 3})
    second_messages = responder.requests[1][0]
    assert [m.role for m in second_messages[-3:]] == ["assistant", "tool", "user"]
    assert "explaining your reasoning" not in second_messages[-1].content


def test_every_native_call_gets_a_result_when_one_fails():
    _, responder, registry, _ = _run(
        [
            _reply(
                "Two taps.",
                _call("call_1", "click", {"index": FAILING_INDEX}),
                _call("call_2", "click", {"index": 4}),
            ),
            _complete(),
        ]
    )

    assert registry.executed[0] == ("click", {"index": FAILING_INDEX})
    assert ("click", {"index": 4}) not in registry.executed
    tools = [m for m in responder.requests[1][0] if m.role == "tool"]
    assert [m.additional_kwargs["tool_call_id"] for m in tools] == [
        "call_1",
        "call_2",
    ]
    assert tools[0].content == "Error: No element at index 99"
    assert tools[1].content == "Error: Not executed because click failed."


def test_invalid_and_string_arguments():
    _, responder, registry, _ = _run(
        [
            _reply("Bad JSON.", _call("call_1", "click", "{index: ")),
            _reply("Index as text.", _call("call_2", "click", {"index": "7"})),
            _complete(),
        ]
    )

    assert registry.executed[0] == ("click", {"index": 7})
    first_result = [m for m in responder.requests[1][0] if m.role == "tool"][0]
    assert first_result.content == (
        "Error: Invalid arguments for click: arguments are not valid JSON"
    )


def test_text_only_reply_asks_for_a_tool_call():
    _, responder, _, _ = _run([_reply("I'll open Settings."), _complete()])

    nudge = responder.requests[1][0][-1]
    assert nudge.role == "user"
    assert "No tool was called" in nudge.content
    assert "<function_calls>" not in nudge.content


def test_xml_mode_is_kept_for_unsupported_llms_and_when_configured():
    xml_reply = ChatMessage(
        role="assistant",
        content=(
            'Done.\n<function_calls>\n<invoke name="complete">'
            '<parameter name="success">true</parameter>'
            '<parameter name="message">finished</parameter></invoke>\n'
            "</function_calls>"
        ),
    )
    for kwargs in ({"native": False}, {"native": True, "tool_calling": "xml"}):
        result, responder, _, _ = _run([xml_reply.model_copy(deep=True)], **kwargs)

        assert result["success"] is True
        messages, chat_kwargs = responder.requests[0]
        assert messages[0].content == "xml"
        assert chat_kwargs == {}


def test_unknown_tool_calling_mode_is_rejected():
    with pytest.raises(ValueError, match="tool_calling"):
        _run([], tool_calling="json")


def test_history_serializes_to_responses_input():
    _, responder, _, _ = _run(
        [_reply("I'll tap it.", _call("call_1", "click", {"index": 3})), _complete()]
    )

    items = to_openai_message_dicts(responder.requests[1][0][1:], is_responses_api=True)
    types = [item.get("type", item.get("role")) for item in items]
    assert types == [
        "user",
        "assistant",
        "function_call",
        "function_call_output",
        "user",
    ]
    assert items[2]["call_id"] == items[3]["call_id"] == "call_1"


def test_chat_completions_llms_get_chat_tools_and_history():
    _, responder, _, _ = _run(
        [_reply("I'll tap it.", _call("call_1", "click", {"index": 3})), _complete()],
        native_tool_api="chat_completions",
    )

    first_kwargs = responder.requests[0][1]
    assert first_kwargs == {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "click",
                    "description": "Tap an element.",
                    "parameters": CLICK_PARAMETERS,
                },
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }

    items = to_openai_message_dicts(responder.requests[1][0][1:])
    assert [item["role"] for item in items] == ["user", "assistant", "tool", "user"]
    assert items[1]["tool_calls"][0]["id"] == items[2]["tool_call_id"] == "call_1"


def test_registry_builds_function_tool_specs():
    registry = ToolRegistry()
    registry.register(
        "swipe",
        lambda **_: None,
        {
            "coordinate": {"type": "list", "required": True},
            "duration": {"type": "number", "required": False, "default": 1.0},
            "payload": {"type": "any", "required": False, "description": "Any value"},
        },
        "Swipe between points.",
    )

    assert registry.get_function_tool_specs() == [
        {
            "type": "function",
            "name": "swipe",
            "description": "Swipe between points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coordinate": {"type": "array", "items": {}},
                    "duration": {"type": "number"},
                    "payload": {"description": "Any value"},
                },
                "required": ["coordinate"],
            },
            "strict": False,
        }
    ]


def test_trimmed_history_never_starts_with_tool_results():
    history = [
        ChatMessage(role="user", content="goal"),
        _reply("tap", _call("call_1", "click", {"index": 1})),
        ChatMessage(
            role="tool", content="ok", additional_kwargs={"tool_call_id": "call_1"}
        ),
        ChatMessage(role="user", content="next"),
    ]

    assert [m.role for m in limit_history(history, 3)] == ["user", "user"]
    assert [m.role for m in limit_history(history, 2, preserve_first=False)] == ["user"]


def test_retries_accept_tool_call_only_responses_and_forward_kwargs():
    calls = []

    class _LLM:
        async def achat(self, messages, **kwargs):
            calls.append(kwargs)
            return ChatResponse(message=_reply("", _call("call_1", "click", {})))

    response = asyncio.run(
        acall_with_retries(_LLM(), [], retries=1, tools=[{"name": "click"}])
    )

    assert response.message.blocks[0].tool_call_id == "call_1"
    assert calls == [{"tools": [{"name": "click"}]}]


def test_openai_xai_and_deepseek_llms_use_native_tools(monkeypatch):
    from mobilerun.agent.utils.llm_picker import load_llm
    from mobilerun.agent.utils.oauth.grok_oauth_llm import GrokOAuth
    from mobilerun.agent.utils.oauth.openai_oauth_llm import OpenAIOAuth

    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    monkeypatch.setenv("XAI_API_KEY", "stub")

    assert load_llm("OpenAIResponses", model="gpt-6-luna").supports_native_tool_calls
    assert load_llm("XAI", model="grok-4.6").supports_native_tool_calls
    deepseek = load_llm("DeepSeek", api_key="stub")
    assert deepseek.supports_native_tool_calls
    assert deepseek.native_tool_api == "chat_completions"
    assert deepseek.class_name() == "OpenAILike"
    assert OpenAIOAuth.supports_native_tool_calls
    assert GrokOAuth.supports_native_tool_calls
