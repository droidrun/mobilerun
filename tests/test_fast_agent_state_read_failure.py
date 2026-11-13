import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.fast_agent.fast_agent import FastAgent
from mobilerun.tools.driver import DeviceDisconnectedError

READ_ERROR = "Failed to get state after 7 attempts: device 'emulator-5554' not found"

CLICK_RESPONSE = """I will tap the target.
<function_calls>
<invoke name="click">
<parameter name="index">12</parameter>
</invoke>
</function_calls>"""

COMPLETE_RESPONSE = """The task is complete.
<function_calls>
<invoke name="complete">
<parameter name="success">true</parameter>
<parameter name="message">Done</parameter>
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


class _FailingStateProvider:
    """Returns a valid state for the first `ok_reads` reads, then raises."""

    requires_coordinate_tools = False

    def __init__(self, ok_reads, error):
        self.ok_reads = ok_reads
        self.error = error
        self.reads = 0

    async def get_state(self):
        self.reads += 1
        if self.reads > self.ok_reads:
            raise self.error
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

    async def execute(self, name, parameters, _action_ctx, workflow_ctx=None):
        self.executed.append((name, parameters))
        if name == "complete":
            await self.shared_state.complete(
                parameters.get("success", False),
                message=parameters.get("message", ""),
            )
        return ActionResult(success=True, summary=f"{name} executed")


def _run(responses, state_provider):
    async def _run_agent():
        shared_state = MobileAgentState()
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
            llm=SimpleNamespace(),
            agent_config=config,
            registry=registry,
            action_ctx=action_ctx,
            state_provider=state_provider,
            shared_state=shared_state,
            prompt_resolver=_PromptResolver(),
        )

        with (
            patch(
                "mobilerun.agent.fast_agent.fast_agent.acall_with_retries",
                new=responder,
            ),
            patch(
                "mobilerun.agent.fast_agent.fast_agent.get_usage_from_response",
                return_value=None,
            ),
        ):
            result = await agent.run(input="Exercise a failed state read")

        return result, responder, registry

    return asyncio.run(_run_agent())


def test_failed_state_read_stops_before_llm_call_and_action():
    # Step 1 sees the screen and taps. Step 2's state read fails: the previous
    # screen must not be sent to the LLM or acted on.
    provider = _FailingStateProvider(ok_reads=1, error=Exception(READ_ERROR))
    result, responder, registry = _run(
        [CLICK_RESPONSE, CLICK_RESPONSE, COMPLETE_RESPONSE], provider
    )

    assert result["success"] is False
    assert "Could not read the device state" in result["reason"]
    assert READ_ERROR in result["reason"]
    assert len(responder.requests) == 1
    assert registry.executed == [("click", {"index": 12})]


def test_failed_first_state_read_makes_no_llm_call():
    provider = _FailingStateProvider(ok_reads=0, error=Exception(READ_ERROR))
    result, responder, registry = _run([COMPLETE_RESPONSE], provider)

    assert result["success"] is False
    assert READ_ERROR in result["reason"]
    assert responder.requests == []
    assert registry.executed == []


def test_device_disconnected_error_still_propagates():
    provider = _FailingStateProvider(
        ok_reads=0, error=DeviceDisconnectedError("device gone")
    )

    with pytest.raises(DeviceDisconnectedError):
        _run([COMPLETE_RESPONSE], provider)
