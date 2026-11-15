import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.fast_agent.fast_agent import FastAgent
from mobilerun.agent.selectors import LlamaIndexLLMSelector
from mobilerun.tools.ui.provider import StateProvider

MALFORMED_RESPONSE = """I'm still on the main feed. I need to tap the Profile tab.

<function_calls>
<invoke name="click">
<｜DSML｜ name="index">189</｜DSML｜>
</invoke>
</function_calls>"""

DSML_ONLY_RESPONSE = """I will tap the Profile tab.
<｜DSML｜tool_calls>
<｜DSML｜invoke name="click">
<｜DSML｜parameter name="index">189</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""

CORRUPTED_PARAMETER_RESPONSE = """Settings home is visible. Marking the task complete.
<function_calls>
<invoke name="complete">
<parameter name="success">true</｜｜DSML｜｜>
<parameter name="message">Completed 20 verified cycles.</parameter>
</invoke>
</function_calls>"""

COMPLETE_RESPONSE = """The task is complete.
<function_calls>
<invoke name="complete">
<parameter name="success">true</parameter>
<parameter name="message">Done</parameter>
</invoke>
</function_calls>"""

CLICK_RESPONSE = """I will tap the target.
<function_calls>
<invoke name="click">
<parameter name="index">12</parameter>
</invoke>
</function_calls>"""


class _SequencedResponder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def __call__(self, _llm, messages, *, stream):
        self.requests.append(list(messages))
        if not self.responses:
            raise AssertionError("FastAgent made an unexpected extra LLM request")
        content = self.responses.pop(0)
        return ChatResponse(message=ChatMessage(role="assistant", content=content))


class _PromptResolver:
    def get_prompt(self, name):
        if name == "fast_agent_system":
            return "Use the documented XML tools: {{ tool_descriptions }}"
        if name == "fast_agent_user":
            return "Goal: {{ goal }}"
        return None


class _StateProvider(StateProvider):
    requires_coordinate_tools = False

    def __init__(self):
        super().__init__(None)

    async def get_state(self):
        return SimpleNamespace(
            formatted_text="Settings home with Network & internet visible",
            focused_text="",
            elements=[],
            phone_state={
                "packageName": "com.android.settings",
                "currentApp": "com.android.settings.Settings",
            },
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
        return {name: {} for name in self.tools}

    async def execute(self, name, parameters, _action_ctx, workflow_ctx=None):
        self.executed.append((name, parameters))
        if name == "complete":
            await self.shared_state.complete(
                parameters.get("success", False),
                message=parameters.get("message", ""),
            )
        return ActionResult(success=True, summary=f"{name} executed")


def _run_sequence(responses, pending_message=None):
    async def _run():
        shared_state = MobileAgentState()
        if pending_message:
            shared_state.queue_user_message(pending_message)
        registry = _Registry(shared_state)
        responder = _SequencedResponder(responses)
        config = SimpleNamespace(
            fast_agent=SimpleNamespace(vision=False, parallel_tools=False),
            max_steps=20,
            streaming=False,
            after_sleep_action=0,
        )
        action_ctx = SimpleNamespace(
            driver=SimpleNamespace(),
            credential_manager=None,
            ui=None,
        )
        agent = FastAgent(
            action_selector=LlamaIndexLLMSelector(SimpleNamespace()),
            agent_config=config,
            registry=registry,
            action_ctx=action_ctx,
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
            result = await agent.run(input="Exercise the malformed-call guard")

        return result, responder, registry, agent, shared_state

    return asyncio.run(_run())


def _correction_messages(shared_state):
    return [
        message.content
        for message in shared_state.message_history
        if message.role == "user"
        and message.content
        and "tool-call markup that could not be parsed" in message.content
    ]


def _request_text(request):
    return "\n".join(message.content or "" for message in request)


def test_three_malformed_responses_stop_without_a_fourth_request():
    result, responder, registry, _agent, shared_state = _run_sequence(
        [MALFORMED_RESPONSE] * 3
    )

    assert result["success"] is False
    assert "3 consecutive times" in result["reason"]
    assert len(responder.requests) == 3
    assert registry.executed == []

    corrections = _correction_messages(shared_state)
    assert len(corrections) == 2
    assert "attempt 1/3" in corrections[0]
    assert "attempt 2/3" in corrections[1]
    assert "<function_calls>" in corrections[0]
    assert "<invoke" in corrections[0]
    assert "<parameter" in corrections[0]
    assert "｜" not in corrections[0]
    assert "No tool calls were provided" not in corrections[0]
    assert "attempt 1/3" in _request_text(responder.requests[1])
    assert "attempt 2/3" in _request_text(responder.requests[2])


def test_dsml_without_xml_wrapper_also_stops_after_three_responses():
    result, responder, registry, _agent, _shared_state = _run_sequence(
        [DSML_ONLY_RESPONSE] * 3
    )

    assert result["success"] is False
    assert "3 consecutive times" in result["reason"]
    assert len(responder.requests) == 3
    assert registry.executed == []


def test_dsml_inside_parameter_never_executes_and_stops_after_three_responses():
    result, responder, registry, _agent, _shared_state = _run_sequence(
        [CORRUPTED_PARAMETER_RESPONSE] * 3
    )

    assert result["success"] is False
    assert "3 consecutive times" in result["reason"]
    assert len(responder.requests) == 3
    assert registry.executed == []


@pytest.mark.parametrize(
    "malformed",
    [MALFORMED_RESPONSE, DSML_ONLY_RESPONSE],
    ids=["xml_wrapped", "dsml_only"],
)
def test_valid_block_after_malformed_block_never_executes(malformed):
    result, responder, registry, _agent, shared_state = _run_sequence(
        [f"{malformed}\nThe Profile tab is open.\n{COMPLETE_RESPONSE}"] * 3
    )

    assert result["success"] is False
    assert "3 consecutive times" in result["reason"]
    assert len(responder.requests) == 3
    assert registry.executed == []
    assert len(_correction_messages(shared_state)) == 2


def test_terminal_guard_explicitly_drains_pending_external_messages():
    result, _responder, _registry, _agent, shared_state = _run_sequence(
        [MALFORMED_RESPONSE] * 3,
        pending_message="Use the visible Profile tab",
    )

    assert result["success"] is False
    assert shared_state.pending_user_messages == []


def test_two_malformed_responses_can_recover_with_a_valid_call():
    result, responder, registry, agent, _shared_state = _run_sequence(
        [MALFORMED_RESPONSE, MALFORMED_RESPONSE, COMPLETE_RESPONSE]
    )

    assert result["success"] is True
    assert len(responder.requests) == 3
    assert [name for name, _params in registry.executed] == ["complete"]
    assert agent._consecutive_malformed_tool_calls == 0


def test_valid_call_resets_the_malformed_response_streak():
    responses = [
        MALFORMED_RESPONSE,
        MALFORMED_RESPONSE,
        CLICK_RESPONSE,
        MALFORMED_RESPONSE,
        MALFORMED_RESPONSE,
        MALFORMED_RESPONSE,
    ]

    result, responder, registry, _agent, _shared_state = _run_sequence(responses)

    assert result["success"] is False
    assert len(responder.requests) == 6
    assert [name for name, _params in registry.executed] == ["click"]


def test_plain_response_keeps_generic_feedback_and_resets_the_streak():
    responses = [
        MALFORMED_RESPONSE,
        MALFORMED_RESPONSE,
        "I need to inspect the screen again.",
        MALFORMED_RESPONSE,
        MALFORMED_RESPONSE,
        COMPLETE_RESPONSE,
    ]

    result, responder, registry, _agent, _shared_state = _run_sequence(responses)

    assert result["success"] is True
    assert len(responder.requests) == 6
    assert "No tool calls were provided" in _request_text(responder.requests[3])
    assert [name for name, _params in registry.executed] == ["complete"]
