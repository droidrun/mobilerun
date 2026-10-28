import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.mock import MockLLM
from llama_index.core.workflow import Context
from selector_helpers import StubSelector

from mobilerun import MobileAgent, MobileConfig
from mobilerun.agent.droid import droid_agent
from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.executor.executor_agent import ExecutorAgent
from mobilerun.agent.fast_agent.events import FastAgentInputEvent
from mobilerun.agent.fast_agent.fast_agent import FastAgent
from mobilerun.agent.selectors import (
    ActionSelector,
    BaseActionSelector,
    DecisionSelector,
    FallbackSelector,
    LlamaIndexLLMSelector,
)
from mobilerun.agent.tool_registry import ToolRegistry
from mobilerun.agent.utils import llm_loader
from mobilerun.config_manager import AgentConfig
from mobilerun.tools.ui.state import UIState


@pytest.mark.parametrize("reasoning", [False, True])
def test_fallback_chain_does_not_load_an_unused_action_llm(monkeypatch, reasoning):
    config = MobileConfig()
    config.agent.reasoning = reasoning
    action_role = "executor" if reasoning else "fast_agent"
    del config.llm_profiles[action_role]
    helper = MockLLM()
    roles = ["manager", "app_opener"] if reasoning else ["app_opener"]
    load = Mock(return_value=dict.fromkeys(roles, helper))
    monkeypatch.setattr(llm_loader, "load_llms_from_profiles", load)
    fallback_llm = MockLLM()
    selector = FallbackSelector(
        DecisionSelector(SimpleNamespace(decide=AsyncMock())), [LlamaIndexLLMSelector(fallback_llm)],
    )

    agent = MobileAgent("Select", config=config, action_selectors=selector)

    assert agent.action_selector is selector
    assert agent.app_opener_llm is helper
    assert agent.app_opener_llm is not fallback_llm
    assert getattr(agent, f"{action_role}_llm") is None
    load.assert_called_once_with(config.llm_profiles, profile_names=roles)


@pytest.mark.parametrize("requires_action_llm", [False, True])
def test_model_loading_queries_declared_selector_properties(monkeypatch, requires_action_llm):
    config = MobileConfig()
    llm = MockLLM()
    roles = ["fast_agent", "app_opener"] if requires_action_llm else ["app_opener"]
    load = Mock(return_value=dict.fromkeys(roles, llm))
    monkeypatch.setattr(llm_loader, "load_llms_from_profiles", load)
    selector = StubSelector(select=AsyncMock(), requires_action_llm=requires_action_llm)

    agent = MobileAgent("Select", config=config, action_selectors=selector)

    assert agent.action_selector is selector
    load.assert_called_once_with(config.llm_profiles, profile_names=roles)


def test_custom_selector_can_supply_a_shared_llm(monkeypatch):
    llm = MockLLM()
    selector = StubSelector(select=AsyncMock(), shared_llm=llm)
    load = Mock(side_effect=AssertionError("The shared LLM must not be reloaded"))
    monkeypatch.setattr(droid_agent, "load_agent_llms", load)

    agent = MobileAgent("Select", action_selectors=selector)

    assert agent.action_selector is selector
    for role in ("manager", "executor", "fast_agent", "app_opener", "structured_output"):
        assert getattr(agent, f"{role}_llm") is llm
    load.assert_not_called()


def test_per_role_selector_property_does_not_override_helper_llms(monkeypatch):
    action_llm = MockLLM()
    helper = MockLLM()
    selector = StubSelector(select=AsyncMock(), shared_llm=action_llm)
    monkeypatch.setattr(droid_agent, "load_agent_llms", Mock(return_value={"app_opener": helper}))

    agent = MobileAgent("Select", config=MobileConfig(), action_selectors={"fast_agent": selector})

    assert agent.action_selector is selector
    assert agent.fast_agent_llm is action_llm
    assert agent.app_opener_llm is helper


def test_base_class_defaults_need_no_dummy_llm():
    class CustomSelector(BaseActionSelector):
        async def select(self, request):
            return None

    selector = CustomSelector()
    agent = MobileAgent("Select", action_selectors=selector)
    assert agent.action_selector is selector
    assert agent.app_opener_llm is None


@pytest.mark.parametrize("selector_class", [ActionSelector, BaseActionSelector])
def test_selector_contract_requires_select_implementation(selector_class):
    with pytest.raises(TypeError, match="abstract.*select"):
        selector_class()


def test_mobile_agent_rejects_select_only_objects_without_the_contract():
    with pytest.raises(TypeError, match="implementing ActionSelector"):
        MobileAgent("Select", action_selectors=SimpleNamespace(select=AsyncMock()))


@pytest.mark.parametrize("agent_class", [FastAgent, ExecutorAgent])
def test_action_workflows_reject_selectors_without_the_contract(agent_class):
    with pytest.raises(TypeError, match="action_selector must implement ActionSelector"):
        agent_class(action_selector=SimpleNamespace(select=AsyncMock()))


@pytest.mark.parametrize("kind", ["llm", "custom", "minimal"])
def test_fast_agent_queries_abstention_reason_and_preserves_defaults(kind):
    async def run():
        if kind == "llm":
            selector = LlamaIndexLLMSelector(MockLLM())
            expected = "LLM response is None. This is a critical error."
        else:
            selector = StubSelector(select=AsyncMock(return_value=None))
            expected = "Action selector abstained without a fallback."
            if kind == "custom":
                expected = "No suitable action from my selector."
                selector.abstention_reason = expected
        state = UIState(
            elements=[], formatted_text="", focused_text="", phone_state={},
            screen_width=100, screen_height=100,
        )
        provider = SimpleNamespace(
            get_state=AsyncMock(return_value=state), get_action_candidates=lambda state: (),
        )
        shared = MobileAgentState(instruction="Select")
        shared.message_history = [ChatMessage(role="user", content="Select")]
        agent = FastAgent(
            selector, AgentConfig(), ToolRegistry(), SimpleNamespace(ui=None), provider,
            shared_state=shared,
        )
        agent.system_prompt = ChatMessage(role="system", content="Choose an action")
        with (
            patch.object(Context, "write_event_to_stream"),
            patch("mobilerun.agent.selectors.llama_index_llm.acall_with_retries", new_callable=AsyncMock) as infer,
        ):
            infer.return_value = None
            event = await agent.handle_llm_input(Context(agent), FastAgentInputEvent())
        assert event.reason == expected

    asyncio.run(run())
