import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from llama_index.core.base.llms.types import ChatMessage, ChatResponse

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.fast_agent.fast_agent import FastAgent
from mobilerun.agent.selectors import LlamaIndexLLMSelector
from mobilerun.tools.ui.provider import StateProvider

FAILING_INDEX = 99


def _click(index):
    return f'<invoke name="click"><parameter name="index">{index}</parameter></invoke>'


def _complete(message):
    return (
        '<invoke name="complete"><parameter name="success">true</parameter>'
        f'<parameter name="message">{message}</parameter></invoke>'
    )


def _reply(*invokes):
    return (
        "I will act now.\n<function_calls>\n"
        + "\n".join(invokes)
        + "\n</function_calls>"
    )


FINAL = _reply(_complete("real"))


class _Responder:
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
        return {name: {} for name in self.tools}

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


def _run(responses):
    async def _go():
        shared_state = MobileAgentState()
        registry = _Registry(shared_state)
        responder = _Responder(responses)
        agent = FastAgent(
            action_selector=LlamaIndexLLMSelector(SimpleNamespace()),
            agent_config=SimpleNamespace(
                fast_agent=SimpleNamespace(vision=False, parallel_tools=True),
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
            result = await agent.run(input="Exercise block execution")
        return result, responder, registry

    return asyncio.run(_go())


def _last_user_text(request):
    return request[-1].content or ""


def test_failed_call_skips_the_rest_of_its_block():
    result, responder, registry = _run(
        [_reply(_click(FAILING_INDEX), _complete("invented")), FINAL]
    )

    assert registry.executed == [
        ("click", {"index": FAILING_INDEX}),
        ("complete", {"success": True, "message": "real"}),
    ]
    assert result["reason"] == "real"
    results = _last_user_text(responder.requests[1])
    assert "No element at index 99" in results
    assert "Not executed because click failed." in results


def test_calls_before_the_failure_still_run():
    _result, responder, registry = _run(
        [_reply(_click(1), _click(FAILING_INDEX), _click(3)), FINAL]
    )

    assert registry.executed[:2] == [
        ("click", {"index": 1}),
        ("click", {"index": FAILING_INDEX}),
    ]
    assert ("click", {"index": 3}) not in registry.executed
    assert "Not executed because click failed." in _last_user_text(
        responder.requests[1]
    )


def test_successful_calls_all_run():
    _result, _responder, registry = _run([_reply(_click(1), _click(2)), FINAL])

    assert registry.executed[:2] == [("click", {"index": 1}), ("click", {"index": 2})]
