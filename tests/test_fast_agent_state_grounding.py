"""Failed reads must not present an old UI snapshot as current evidence."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse

from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.fast_agent.events import FastAgentInputEvent
from mobilerun.agent.fast_agent.fast_agent import FastAgent
from mobilerun.agent.utils.signatures import build_tool_registry
from mobilerun.config_manager.config_manager import AgentConfig


def _state(text):
    return SimpleNamespace(
        formatted_text=text,
        focused_text=text,
        elements=[{"text": text}],
        phone_state={"packageName": "com.android.chrome"},
    )


@pytest.mark.parametrize("native_tools", [False, True], ids=["xml", "native"])
@pytest.mark.parametrize("vision", [False, True])
def test_prompt_only_instructs_screenshot_analysis_when_vision_enabled(
    native_tools, vision
):
    async def run():
        registry, _ = await build_tool_registry()
        config = AgentConfig()
        config.fast_agent.vision = vision
        agent = FastAgent(
            llm=SimpleNamespace(
                supports_native_tool_calls=native_tools, class_name=lambda: "TestLLM"
            ),
            agent_config=config,
            registry=registry,
            action_ctx=SimpleNamespace(),
            state_provider=SimpleNamespace(requires_coordinate_tools=False),
            shared_state=MobileAgentState(telemetry_config_enabled=False),
        )
        prompt = (await agent._build_system_prompt()).content
        assert ("Analyze the current device state and screenshot" in prompt) is vision
        assert "\n2. Explain your reasoning\n" in prompt
        if not vision:
            assert "\n- **phone_state**:" in prompt
            assert "Vision is disabled" in prompt
            assert "Web View" in prompt
            assert "not exposed" in prompt
            assert "wait" in prompt
            assert "re-read" in prompt
            assert "could not be read" in prompt
            assert "accessibility-exposed UI elements" in prompt
        else:
            assert "supplied screenshot" in prompt
            assert "Vision is disabled" not in prompt

    asyncio.run(run())


@pytest.mark.parametrize("native_tools", [False, True], ids=["xml", "native"])
def test_failed_state_read_invalidates_current_evidence_and_recovers(native_tools):
    async def run():
        registry, _ = await build_tool_registry()
        config = AgentConfig()
        config.fast_agent.vision = False
        config.streaming = False
        shared = MobileAgentState(
            message_history=[ChatMessage(role="user", content="Read the city and ISP")]
        )
        first = _state("Philadelphia, Verizon Business, US")
        recovered = _state("New York, Example ISP, US")
        provider = SimpleNamespace(
            get_state=AsyncMock(side_effect=[first, RuntimeError("read failed"), recovered])
        )
        action_ctx = SimpleNamespace(driver=SimpleNamespace(), ui=None)
        agent = FastAgent(
            llm=SimpleNamespace(
                supports_native_tool_calls=native_tools, class_name=lambda: "TestLLM"
            ),
            agent_config=config,
            registry=registry,
            action_ctx=action_ctx,
            state_provider=provider,
            shared_state=shared,
        )
        agent.system_prompt = ChatMessage(role="system", content="Read the screen")
        requests = []

        async def respond(_llm, messages, **kwargs):
            requests.append(messages)
            return ChatResponse(message=ChatMessage(role="assistant", content="Continue"))

        ctx = SimpleNamespace(write_event_to_stream=lambda event: None)
        with (
            patch("mobilerun.agent.fast_agent.fast_agent.acall_with_retries", new=respond),
            patch("mobilerun.agent.fast_agent.fast_agent.get_usage_from_response", return_value=None),
            patch.dict("os.environ", {"MOBILERUN_STREAM_SCREENSHOTS": "0"}),
        ):
            await agent.handle_llm_input(ctx, FastAgentInputEvent())
            assert "Philadelphia, Verizon Business, US" in requests[0][-1].content
            assert action_ctx.ui is first

            await agent.handle_llm_input(ctx, FastAgentInputEvent())
            failed_prompt = requests[1][1].content
            assert "Philadelphia, Verizon Business, US" not in failed_prompt
            assert "Device state unavailable" in failed_prompt
            assert "could not be read" in failed_prompt
            assert action_ctx.ui is None
            assert shared.formatted_device_state == ""
            assert shared.focused_text == ""
            assert shared.a11y_tree == []
            assert shared.phone_state == {}

            await agent.handle_llm_input(ctx, FastAgentInputEvent())
            recovered_prompt = requests[2][1].content
            assert "New York, Example ISP, US" in recovered_prompt
            assert "Device state unavailable" not in recovered_prompt
            assert action_ctx.ui is recovered

    asyncio.run(run())
