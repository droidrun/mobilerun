"""Reasoning managers must ground final answers in observed device state."""

import asyncio
from pathlib import Path

import pytest

from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.manager.manager_agent import ManagerAgent
from mobilerun.agent.manager.stateless_manager_agent import StatelessManagerAgent
from mobilerun.agent.utils.prompt_resolver import PromptResolver
from mobilerun.config_manager.config_manager import AgentConfig, ManagerConfig


@pytest.mark.parametrize("template", ["system", "stateless", "rev1", "trained"])
@pytest.mark.parametrize("vision", [False, True])
def test_manager_prompt_grounds_screen_reading_results(template, vision):
    agent = object.__new__(StatelessManagerAgent)
    path = (
        Path(__file__).resolve().parents[1]
        / f"mobilerun/config/prompts/manager/{template}.jinja2"
    )
    agent.agent_config = AgentConfig(
        manager=ManagerConfig(system_prompt=str(path), vision=vision)
    )
    agent.vision = vision
    agent.prompt_resolver = PromptResolver()
    agent.shared_state = MobileAgentState(instruction="Read city and ISP")
    prompt = asyncio.run(agent._build_prompt())
    for text in (
        "observed",
        "faithfully",
        "app or page",
        "could not be read",
        'success="false"',
        "host-side HTTP",
        "previous state or memory",
    ):
        assert text in prompt
    if vision:
        assert "supplied screenshot" in prompt
        assert "Vision is disabled" not in prompt
    else:
        assert "Vision is disabled" in prompt


@pytest.mark.parametrize("vision", [False, True])
def test_stateful_manager_passes_vision_to_prompt(vision):
    agent = object.__new__(ManagerAgent)
    agent.agent_config = AgentConfig(manager=ManagerConfig(vision=vision))
    agent.config = agent.agent_config.manager
    agent.shared_state = MobileAgentState(instruction="Read city and ISP")
    agent.registry = None
    agent.action_ctx = None
    agent.output_model = None
    agent.prompt_resolver = PromptResolver()

    prompt = asyncio.run(agent._build_system_prompt())
    assert ("supplied screenshot" in prompt) is vision
    assert ("Vision is disabled" in prompt) is not vision
