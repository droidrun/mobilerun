import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from llama_index.core.llms.mock import MockLLM
from llama_index.core.workflow import Context, StartEvent
from pydantic import BaseModel
from selector_helpers import CompletedRun, StubSelector

from mobilerun import MobileAgent, MobileConfig
from mobilerun.agent.droid import droid_agent
from mobilerun.agent.selectors import DecisionSelector, LlamaIndexLLMSelector
from mobilerun.agent.utils import llm_loader
from mobilerun.cli import main as cli_main


class _Output(BaseModel):
    answer: str


@pytest.fixture
def cli_agents(monkeypatch):
    """Use real agent initialization and a completed run, never a real device."""
    monkeypatch.setattr(cli_main, "_setup_cli_logging", lambda _: None)
    monkeypatch.setattr(cli_main, "print_telemetry_message", lambda **kw: None)
    monkeypatch.setattr(cli_main, "_cleanup_android_keyboard", AsyncMock())
    agents = []

    def initialize(**kwargs):
        agents.append(MobileAgent(**kwargs))
        return SimpleNamespace(run=lambda: CompletedRun())

    monkeypatch.setattr(cli_main, "MobileAgent", initialize)
    return agents


@pytest.mark.parametrize("reasoning", [False, True])
def test_legacy_shared_llm_is_wrapped_and_preserves_roles(monkeypatch, caplog, reasoning):
    config = MobileConfig()
    config.agent.reasoning = reasoning
    llm = MockLLM()
    load = Mock(side_effect=AssertionError("A supplied LLM must not load profiles"))
    monkeypatch.setattr(droid_agent, "load_agent_llms", load)
    monkeypatch.setattr(logging.getLogger("mobilerun"), "propagate", True)

    agent = MobileAgent("Select", config=config, llms=llm, output_model=_Output)

    for role in ("manager", "executor", "fast_agent", "app_opener", "structured_output"):
        assert getattr(agent, f"{role}_llm") is llm
    assert isinstance(agent.action_selector, LlamaIndexLLMSelector)
    assert agent.action_selector.llm is llm
    if reasoning:
        assert agent.executor_agent.action_selector is agent.action_selector
    assert "MobileAgent(llms=...) is deprecated" in caplog.text
    load.assert_not_called()


def test_legacy_shared_llm_needs_no_config(monkeypatch):
    llm = MockLLM()
    load = Mock(side_effect=AssertionError("A supplied LLM must not load profiles"))
    monkeypatch.setattr(droid_agent, "load_agent_llms", load)

    agent = MobileAgent("Select", llms=llm)

    assert agent.action_selector.llm is llm
    assert agent.app_opener_llm is llm
    load.assert_not_called()


@pytest.mark.parametrize("reasoning", [False, True])
def test_legacy_llm_dictionary_merges_missing_roles(monkeypatch, reasoning):
    config = MobileConfig()
    config.agent.reasoning = reasoning
    role = "executor" if reasoning else "fast_agent"
    llm = MockLLM()
    helper = MockLLM()
    provided = {role: llm}
    missing_roles = ["manager", "app_opener"] if reasoning else ["app_opener"]
    load = Mock(return_value=dict.fromkeys(missing_roles, helper))
    monkeypatch.setattr(llm_loader, "load_llms_from_profiles", load)

    agent = MobileAgent("Select", config=config, llms=provided)

    assert agent.action_selector.llm is llm
    assert agent.app_opener_llm is helper
    if reasoning:
        assert agent.manager_llm is helper
    load.assert_called_once_with(config.llm_profiles, profile_names=missing_roles)
    assert provided == {role: llm}


def test_legacy_llm_takes_precedence_over_configured_decision_model(monkeypatch):
    from mobilerun.config_manager import DecisionProfile

    config = MobileConfig()
    config.decision_profiles["fast_agent"] = DecisionProfile()
    llm = MockLLM()
    load = Mock(side_effect=AssertionError("Explicit LLM must bypass decision loading"))
    monkeypatch.setattr(droid_agent, "load_agent_decision_models", load)

    agent = MobileAgent("Select", config=config, llms=llm)

    assert agent.action_selector.llm is llm
    load.assert_not_called()


@pytest.mark.parametrize("llm_selector", [False, True])
@pytest.mark.parametrize("dictionary", [False, True])
def test_explicit_selector_ignores_legacy_llms(monkeypatch, caplog, llm_selector, dictionary):
    legacy_llm = MockLLM()
    selected_llm = MockLLM() if llm_selector else None
    selector = (
        LlamaIndexLLMSelector(selected_llm)
        if llm_selector else StubSelector(select=AsyncMock())
    )
    legacy_llms = {"fast_agent": legacy_llm} if dictionary else legacy_llm
    load = Mock(side_effect=AssertionError("Explicit selector needs no config here"))
    monkeypatch.setattr(droid_agent, "load_agent_llms", load)
    monkeypatch.setattr(logging.getLogger("mobilerun"), "propagate", True)

    agent = MobileAgent("Select", llms=legacy_llms, action_selectors=selector)

    assert agent.action_selector is selector
    for role in ("manager", "executor", "fast_agent", "app_opener", "structured_output"):
        assert getattr(agent, f"{role}_llm") is selected_llm
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert "MobileAgent(llms=...) is deprecated" in warnings[0].message
    assert "The supplied llms argument was ignored" in warnings[1].message
    load.assert_not_called()


@pytest.mark.parametrize("reasoning", [False, True])
def test_selector_with_legacy_llms_still_loads_helpers_from_config(monkeypatch, reasoning):
    config = MobileConfig()
    config.agent.reasoning = reasoning
    legacy_llm = MockLLM()
    configured_llm = MockLLM()
    selector = StubSelector(select=AsyncMock())
    load = Mock(return_value=dict.fromkeys(
        ("manager", "executor", "fast_agent", "app_opener"), configured_llm,
    ))
    monkeypatch.setattr(droid_agent, "load_agent_llms", load)

    agent = MobileAgent("Select", config=config, llms=legacy_llm, action_selectors=selector)

    assert agent.action_selector is selector
    assert agent.app_opener_llm is configured_llm
    if reasoning:
        assert agent.manager_llm is configured_llm
        assert agent.executor_agent.action_selector is selector
    load.assert_called_once_with(config=agent.config, output_model=None)


def test_legacy_none_llms_still_loads_config(monkeypatch, caplog):
    llm = MockLLM()
    load = Mock(return_value={"fast_agent": llm, "app_opener": llm})
    monkeypatch.setattr(droid_agent, "load_agent_llms", load)
    monkeypatch.setattr(logging.getLogger("mobilerun"), "propagate", True)

    agent = MobileAgent("Select", config=MobileConfig(), llms=None)

    assert agent.action_selector.llm is llm
    assert "MobileAgent(llms=...) is deprecated" in caplog.text
    load.assert_called_once()


def test_action_selectors_can_be_supplied_positionally(monkeypatch):
    llm = MockLLM()
    load = Mock(side_effect=AssertionError("A supplied LLM must not load profiles"))
    monkeypatch.setattr(droid_agent, "load_agent_llms", load)

    selector = LlamaIndexLLMSelector(llm)
    agent = MobileAgent("Select", None, selector)

    assert agent.action_selector.llm is llm
    assert agent.action_selector is selector
    assert agent.app_opener_llm is llm
    load.assert_not_called()


def test_decision_selector_needs_no_config_or_llms(monkeypatch):
    model = SimpleNamespace(decide=AsyncMock())
    load = Mock(side_effect=AssertionError("Decision-only execution needs no LLM"))
    monkeypatch.setattr(droid_agent, "load_agent_llms", load)

    agent = MobileAgent("Select", action_selectors=DecisionSelector(model))

    assert agent.action_selector.model is model
    for role in ("manager", "executor", "fast_agent", "app_opener", "structured_output"):
        assert getattr(agent, f"{role}_llm") is None
    model.decide.assert_not_called()
    load.assert_not_called()


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("structured", [False, True])
def test_decision_selector_loads_only_required_helper_profiles(monkeypatch, reasoning, structured):
    config = MobileConfig()
    config.agent.reasoning = reasoning
    roles = ["manager", "app_opener"] if reasoning else ["app_opener"]
    if structured:
        roles.append("structured_output")
    config.llm_profiles = {role: config.llm_profiles[role] for role in roles}
    for profile in config.llm_profiles.values():
        profile.kwargs["api_key"] = "test-key"
    model = SimpleNamespace(decide=AsyncMock())
    helper = MockLLM()
    load = Mock(return_value=helper)
    monkeypatch.setattr("mobilerun.agent.utils.llm_picker.load_llm", load)
    decision_load = Mock(side_effect=AssertionError("Explicit model must bypass profile loading"))
    monkeypatch.setattr(droid_agent, "load_agent_decision_models", decision_load)

    agent = MobileAgent(
        "Select", config=config, action_selectors=DecisionSelector(model),
        output_model=_Output if structured else None,
    )

    assert agent.action_selector.model is model
    assert agent.executor_llm is None
    assert agent.fast_agent_llm is None
    assert agent.app_opener_llm is helper
    if structured:
        assert agent.structured_output_llm is helper
    if reasoning:
        assert agent.manager_agent.llm is helper
        assert agent.executor_agent.action_selector is agent.action_selector
    assert load.call_count == len(roles)
    # Internal loading flags must never leak into provider constructor kwargs.
    for call, role in zip(load.call_args_list, roles, strict=True):
        profile = config.llm_profiles[role]
        assert call.kwargs == {"provider_name": profile.provider, **profile.to_load_llm_kwargs()}
    assert config.decision_profiles == {}
    decision_load.assert_not_called()


@pytest.mark.parametrize("reasoning", [False, True])
def test_decision_selector_merges_only_missing_helper_llms(monkeypatch, reasoning):
    config = MobileConfig()
    config.agent.reasoning = reasoning
    action_role = "executor" if reasoning else "fast_agent"
    del config.llm_profiles[action_role]
    helper = MockLLM()
    provided = {"app_opener": helper}
    model = SimpleNamespace(decide=AsyncMock())
    load = Mock(return_value={"manager": helper})
    monkeypatch.setattr(llm_loader, "load_llms_from_profiles", load)
    monkeypatch.setattr(droid_agent, "load_agent_llms", Mock(return_value=provided))

    agent = MobileAgent("Select", config=config, action_selectors=DecisionSelector(model))

    assert agent.action_selector.model is model
    assert agent.app_opener_llm is helper
    if reasoning:
        assert agent.manager_llm is helper
        load.assert_called_once_with(config.llm_profiles, profile_names=["manager"])
    else:
        load.assert_not_called()
    assert provided == {"app_opener": helper}


def test_decision_model_kwarg_requires_an_action_selector_wrapper():
    with pytest.raises(TypeError, match="Pass a DecisionSelector via action_selectors"):
        MobileAgent(
            "Select", decision_model=SimpleNamespace(decide=AsyncMock()),
            action_selectors=StubSelector(select=AsyncMock()),
        )


@pytest.mark.parametrize("model", [object(), SimpleNamespace(decide=None)])
def test_decision_selector_requires_decide_method(model):
    with pytest.raises(TypeError, match="must implement decide"):
        MobileAgent("Select", action_selectors=DecisionSelector(model))


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("output_model", [None, _Output])
def test_supplied_llm_selector_preserves_shared_llm_roles(monkeypatch, reasoning, output_model):
    config = MobileConfig()
    config.agent.reasoning = reasoning
    llm = MockLLM()
    selector = LlamaIndexLLMSelector(llm)
    load = Mock(side_effect=AssertionError("A supplied shared LLM must not load profiles"))
    monkeypatch.setattr(droid_agent, "load_agent_llms", load)
    agent = MobileAgent("Select", config=config, action_selectors=selector, output_model=output_model)
    for role in ("manager", "executor", "fast_agent", "app_opener", "structured_output"):
        assert getattr(agent, f"{role}_llm") is llm
    assert agent.action_selector is selector
    assert agent._require_action_selector() is selector
    if reasoning:
        assert agent.manager_agent.llm is llm
        assert agent.executor_agent.action_selector is selector
    load.assert_not_called()


def test_shared_llm_selector_needs_no_config_even_with_structured_output(monkeypatch):
    llm = MockLLM()
    load = Mock(side_effect=AssertionError("No config is needed for a supplied native LLM"))
    monkeypatch.setattr(droid_agent, "load_agent_llms", load)
    agent = MobileAgent("Select", action_selectors=LlamaIndexLLMSelector(llm), output_model=_Output)
    assert agent.app_opener_llm is llm
    assert agent.structured_output_llm is llm
    load.assert_not_called()


def test_external_agent_still_ignores_supplied_llm(monkeypatch):
    config = MobileConfig()
    config.agent.name = "external-agent"
    load = Mock(side_effect=AssertionError("External agents must not load native LLMs"))
    monkeypatch.setattr(droid_agent, "load_agent_llms", load)
    agent = MobileAgent("Select", config=config, action_selectors=LlamaIndexLLMSelector(MockLLM()))
    for role in ("manager", "executor", "fast_agent", "app_opener", "structured_output"):
        assert getattr(agent, f"{role}_llm") is None
    load.assert_not_called()


def test_config_role_resolution_keeps_structured_output_fallback(monkeypatch):
    llm = MockLLM()
    helper = MockLLM()
    monkeypatch.setattr(droid_agent, "load_agent_llms", Mock(return_value={
        "fast_agent": llm, "app_opener": helper,
    }))
    agent = MobileAgent("Select", config=MobileConfig())
    assert agent.action_selector.llm is llm
    assert agent.fast_agent_llm is llm
    assert agent.app_opener_llm is helper
    assert agent.structured_output_llm is llm


def test_config_dictionary_is_merged_before_role_assignment(monkeypatch):
    config = MobileConfig()
    config.agent.reasoning = True
    manager = MockLLM()
    executor = MockLLM()
    helper = MockLLM()
    provided = {"manager": manager}
    monkeypatch.setattr(droid_agent, "load_agent_llms", Mock(return_value=provided))
    missing = Mock(return_value={"executor": executor, "app_opener": helper})
    monkeypatch.setattr(llm_loader, "load_llms_from_profiles", missing)

    agent = MobileAgent("Select", config=config)

    missing.assert_called_once_with(config.llm_profiles, profile_names=["executor", "app_opener"])
    assert agent.manager_llm is manager
    assert agent.executor_llm is executor
    assert agent.app_opener_llm is helper
    assert agent.fast_agent_llm is None
    assert agent.structured_output_llm is None
    assert agent.action_selector.llm is executor
    assert provided == {"manager": manager}


def test_complete_config_dictionary_does_not_reload_models(monkeypatch):
    llm = MockLLM()
    provided = {"fast_agent": llm, "app_opener": llm}
    monkeypatch.setattr(droid_agent, "load_agent_llms", Mock(return_value=provided))
    load = Mock(side_effect=AssertionError("Complete dictionaries must not reload models"))
    monkeypatch.setattr(llm_loader, "load_llms_from_profiles", load)
    agent = MobileAgent("Select", config=MobileConfig())
    assert agent.fast_agent_llm is llm
    load.assert_not_called()


@pytest.mark.parametrize("value", [object(), [], "invalid"])
def test_invalid_loaded_llm_type_retains_original_error(monkeypatch, value):
    monkeypatch.setattr(droid_agent, "load_agent_llms", Mock(return_value=value))
    with pytest.raises(ValueError) as raised:
        MobileAgent("Select", config=MobileConfig())
    assert str(raised.value) == f"Invalid LLM type: {type(value)}"


def test_no_config_or_selector_retains_original_error():
    with pytest.raises(ValueError) as raised:
        MobileAgent("Select")
    assert str(raised.value) == (
        "Either 'llms' or 'config' must be provided. "
        "If llms is not provided, config is required to load LLMs from profiles."
    )


def test_explicit_none_structured_role_does_not_use_fallback_or_new_validation(monkeypatch):
    llm = MockLLM()
    monkeypatch.setattr(droid_agent, "load_agent_llms", Mock(return_value={
        "fast_agent": llm, "app_opener": llm, "structured_output": None,
    }))
    agent = MobileAgent("Select", config=MobileConfig(), output_model=_Output)
    assert agent.fast_agent_llm is llm
    assert agent.structured_output_llm is None


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("structured", [False, True])
def test_default_required_profiles_are_unchanged(reasoning, structured):
    config = MobileConfig()
    config.agent.reasoning = reasoning
    expected = ["manager", "executor", "app_opener"] if reasoning else ["fast_agent", "app_opener"]
    if structured:
        expected.append("structured_output")
    assert llm_loader.validate_llm_profiles(config, object if structured else None) == expected


@pytest.mark.parametrize("custom_selector", [False, True])
def test_selector_does_not_change_existing_profile_loading(monkeypatch, custom_selector):
    config = MobileConfig()
    config.tools.disabled_tools = ["open_app"]
    llm = MockLLM()
    load = Mock(return_value={"fast_agent": llm, "app_opener": llm})
    monkeypatch.setattr(droid_agent, "load_agent_llms", load)
    selector = StubSelector(select=AsyncMock()) if custom_selector else None
    MobileAgent("Select", config=config, action_selectors=selector)
    assert load.call_args.kwargs == {"config": config, "output_model": None}


@pytest.mark.parametrize("platform", ["Android", "iOS"])
def test_helper_availability_does_not_change_original_tool_registry(monkeypatch, platform):
    async def run():
        config = MobileConfig()
        config.device.platform = platform
        config.telemetry.enabled = False
        driver = SimpleNamespace(
            platform=platform, supported={"start_app", "get_apps"},
            supported_buttons=set(), get_date=AsyncMock(return_value="2026-10-23"),
        )
        provider = SimpleNamespace(supported=set())
        selector = StubSelector(select=AsyncMock())
        monkeypatch.setattr(droid_agent, "load_agent_llms", Mock(return_value={
            "fast_agent": None, "app_opener": None,
        }))
        agent = MobileAgent(
            "Open app", config=config, action_selectors=selector,
            driver=driver, state_provider=provider,
        )
        with patch.object(Context, "write_event_to_stream"):
            await agent.start_handler(Context(agent), StartEvent())
        assert "open_app" in agent.registry.tools

    asyncio.run(run())


@pytest.mark.parametrize("temperature", [None, 0.0, 0.2])
def test_cli_custom_provider_retains_original_constructor_defaults(monkeypatch, cli_agents, temperature):
    config = MobileConfig()
    load = Mock(return_value=MockLLM())
    monkeypatch.setattr(cli_main, "load_llm", load)
    monkeypatch.setattr(cli_main.ConfigLoader, "load", lambda _: config)
    assert asyncio.run(cli_main.run_command(
        "Select", provider="Example", model="test", temperature=temperature,
        base_url="https://example.test", api_base="https://api.example.test",
        max_tokens=123, debug=False,
    )) is True
    assert len(cli_agents) == 1
    expected: dict[str, object] = {
        "model": "test",
        "base_url": "https://example.test",
        "api_base": "https://api.example.test",
        "max_tokens": 123,
    }
    if temperature is not None:
        expected["temperature"] = temperature
    load.assert_called_once_with("Example", **expected)


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("temperature", [None, 0.0, 0.2])
def test_cli_profile_overrides_reach_every_required_llm(monkeypatch, cli_agents, reasoning, temperature):
    config = MobileConfig()
    config.agent.reasoning = reasoning
    roles = ["manager", "executor", "app_opener"] if reasoning else ["fast_agent", "app_opener"]
    load = Mock(return_value=dict.fromkeys(roles, MockLLM()))
    monkeypatch.setattr(llm_loader, "load_llms_from_profiles", load)
    monkeypatch.setattr(cli_main.ConfigLoader, "load", lambda _: config)
    assert asyncio.run(cli_main.run_command(
        "Select", temperature=temperature, base_url="https://example.test",
        api_base="https://api.example.test", max_tokens=123, debug=False,
    )) is True
    assert len(cli_agents) == 1
    overrides: dict[str, object] = {
        "base_url": "https://example.test",
        "api_base": "https://api.example.test",
        "max_tokens": 123,
    }
    if temperature is not None:
        overrides["temperature"] = temperature
    load.assert_called_once_with(
        config.llm_profiles, profile_names=roles,
        **{role: overrides for role in roles},
    )


def test_custom_constructor_options_are_copied_before_overrides(monkeypatch):
    load = Mock(return_value=MockLLM())
    monkeypatch.setattr(llm_loader, "load_llm", load)
    options = {"temperature": 0.8, "base_url": "https://original.test"}

    llm_loader.load_agent_llms(
        MobileConfig(), custom_provider="Example", custom_model="test",
        custom_llm_kwargs=options, temperature=0.0,
        base_url="https://override.test", api_base="https://api.example.test",
    )

    load.assert_called_once_with(
        provider_name="Example", model="test", temperature=0.0,
        base_url="https://override.test", api_base="https://api.example.test",
    )
    assert options == {"temperature": 0.8, "base_url": "https://original.test"}


def test_existing_custom_loader_temperature_default_is_unchanged(monkeypatch):
    load = Mock(return_value=MockLLM())
    monkeypatch.setattr(llm_loader, "load_llm", load)
    llm_loader.load_agent_llms(MobileConfig(), custom_provider="Example", custom_model="test")
    load.assert_called_once_with(provider_name="Example", model="test", temperature=0.3)


def test_selector_only_initialization_does_not_load_llms(monkeypatch, caplog):
    load = Mock()
    monkeypatch.setattr(droid_agent, "load_agent_llms", load)
    monkeypatch.setattr(logging.getLogger("mobilerun"), "propagate", True)
    selector = StubSelector(select=AsyncMock())
    agent = MobileAgent("Select", action_selectors=selector)
    assert agent.action_selector is selector
    assert "deprecated" not in caplog.text
    load.assert_not_called()


def test_existing_workflow_options_still_reach_model_loading(monkeypatch):
    load = Mock(return_value={"fast_agent": MockLLM(), "app_opener": MockLLM()})
    monkeypatch.setattr(droid_agent, "load_agent_llms", load)
    MobileAgent("Select", config=MobileConfig(), verbose=True, temperature=0.2)
    assert load.call_args.kwargs["verbose"] is True
    assert load.call_args.kwargs["temperature"] == 0.2


@pytest.mark.parametrize("reasoning", [False, True])
def test_config_resolves_required_native_selector(monkeypatch, reasoning):
    config = MobileConfig()
    config.agent.reasoning = reasoning
    llm = MockLLM()
    monkeypatch.setattr(droid_agent, "load_agent_llms", Mock(return_value={
        "manager": llm, "executor": llm, "fast_agent": llm, "app_opener": llm,
    }))

    agent = MobileAgent("Select", config=config)

    selector = agent._require_action_selector()
    assert isinstance(selector, LlamaIndexLLMSelector)
    assert selector.llm is llm
    if reasoning:
        assert agent.executor_agent.action_selector is selector


def test_external_agent_can_initialize_without_native_selector(monkeypatch):
    config = MobileConfig()
    config.agent.name = "external-agent"
    load = Mock(side_effect=AssertionError("External agents must not load native LLMs"))
    monkeypatch.setattr(droid_agent, "load_agent_llms", load)

    agent = MobileAgent("Select", config=config)

    assert agent.action_selector is None
    assert agent.executor_agent is None
    load.assert_not_called()
    with pytest.raises(ValueError, match="Native agent execution requires an action_selector"):
        agent._require_action_selector()


@pytest.mark.parametrize("reasoning", [False, True])
def test_action_selector_dictionary_preserves_supplied_roles(monkeypatch, reasoning):
    config = MobileConfig()
    config.agent.reasoning = reasoning
    role = "executor" if reasoning else "fast_agent"
    fast = DecisionSelector(SimpleNamespace(decide=AsyncMock()))
    executor = DecisionSelector(SimpleNamespace(decide=AsyncMock()))
    provided = {"fast_agent": fast, "executor": executor}
    helper = MockLLM()
    load = Mock(return_value={"manager": helper, "app_opener": helper})
    monkeypatch.setattr(droid_agent, "load_agent_llms", load)
    decisions = Mock(side_effect=AssertionError("Supplied selectors must not be reloaded"))
    monkeypatch.setattr(droid_agent, "load_agent_decision_models", decisions)

    agent = MobileAgent("Select", config=config, action_selectors=provided)

    assert agent.action_selector is provided[role]
    assert agent._require_action_selector() is provided[role]
    assert agent.action_selectors == provided
    assert agent.action_selectors is not provided
    assert agent.app_opener_llm is helper
    if reasoning:
        assert agent.manager_llm is helper
        assert agent.executor_agent.action_selector is executor
    load.assert_called_once_with(config=agent.config, output_model=None, load_action_llm=False)
    decisions.assert_not_called()
    assert provided == {"fast_agent": fast, "executor": executor}


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("configured_decision", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_action_selector_dictionary_loads_missing_active_role(
    monkeypatch, reasoning, configured_decision, empty,
):
    config = MobileConfig()
    config.agent.reasoning = reasoning
    role = "executor" if reasoning else "fast_agent"
    other_role = "fast_agent" if reasoning else "executor"
    other = DecisionSelector(SimpleNamespace(decide=AsyncMock()))
    provided = {} if empty else {other_role: other}
    model = SimpleNamespace(decide=AsyncMock())
    decisions = Mock(return_value={role: model} if configured_decision else {})
    monkeypatch.setattr(droid_agent, "load_agent_decision_models", decisions)
    llm = MockLLM()
    helper = MockLLM()
    load = Mock(return_value={role: llm, "manager": helper, "app_opener": helper})
    monkeypatch.setattr(droid_agent, "load_agent_llms", load)

    agent = MobileAgent("Select", config=config, action_selectors=provided)

    if configured_decision:
        assert isinstance(agent.action_selector, DecisionSelector)
        assert agent.action_selector.model is model
    else:
        assert isinstance(agent.action_selector, LlamaIndexLLMSelector)
        assert agent.action_selector.llm is llm
    assert agent.action_selectors == {**provided, role: agent.action_selector}
    assert role not in provided
    if not empty:
        assert agent.action_selectors[other_role] is other
    assert agent.app_opener_llm is helper
    expected = {"config": agent.config, "output_model": None}
    if configured_decision:
        expected["load_action_llm"] = False
    load.assert_called_once_with(**expected)
    decisions.assert_called_once_with(agent.config)


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("decision", [False, True])
def test_action_selector_dictionary_loads_only_helper_profiles(monkeypatch, reasoning, decision):
    config = MobileConfig()
    config.agent.reasoning = reasoning
    role = "executor" if reasoning else "fast_agent"
    # The supplied role must work without any corresponding profile or API key.
    del config.llm_profiles[role]
    helper_roles = ["manager", "app_opener"] if reasoning else ["app_opener"]
    helper_roles.append("structured_output")
    for name in helper_roles:
        config.llm_profiles[name].kwargs["api_key"] = "test-key"
    model = SimpleNamespace(decide=AsyncMock()) if decision else MockLLM()
    selector = DecisionSelector(model) if decision else LlamaIndexLLMSelector(model)
    helper = MockLLM()
    load = Mock(return_value=helper)
    monkeypatch.setattr("mobilerun.agent.utils.llm_picker.load_llm", load)

    agent = MobileAgent(
        "Select", config=config, action_selectors={role: selector}, output_model=_Output,
    )

    assert agent.action_selector is selector
    assert agent.app_opener_llm is helper
    assert agent.structured_output_llm is helper
    assert getattr(agent, f"{role}_llm") is (None if decision else model)
    assert load.call_count == len(helper_roles)
    for call, name in zip(load.call_args_list, helper_roles, strict=True):
        profile = config.llm_profiles[name]
        assert call.kwargs == {"provider_name": profile.provider, **profile.to_load_llm_kwargs()}


@pytest.mark.parametrize("decision", [False, True])
def test_action_selector_dictionary_works_without_config(monkeypatch, decision):
    model = SimpleNamespace(decide=AsyncMock()) if decision else MockLLM()
    selector = DecisionSelector(model) if decision else LlamaIndexLLMSelector(model)
    load = Mock(side_effect=AssertionError("Selector-only execution needs no config"))
    monkeypatch.setattr(droid_agent, "load_agent_llms", load)

    agent = MobileAgent("Select", action_selectors={"fast_agent": selector})

    assert agent.action_selector is selector
    assert agent.action_selectors == {"fast_agent": selector}
    assert agent.fast_agent_llm is (None if decision else model)
    assert agent.app_opener_llm is None
    assert agent.manager_llm is None
    assert agent.structured_output_llm is None
    load.assert_not_called()


def test_mobile_agent_rejects_removed_singular_selector_argument():
    selector = DecisionSelector(SimpleNamespace(decide=AsyncMock()))
    with pytest.raises(TypeError, match="Use action_selectors, not action_selector="):
        MobileAgent("Select", action_selector=selector)


@pytest.mark.parametrize("value", [[], object(), "invalid"])
def test_action_selectors_requires_a_selector_or_dictionary(value):
    with pytest.raises(TypeError, match="action_selectors must be a selector or a dictionary"):
        MobileAgent("Select", action_selectors=value)


@pytest.mark.parametrize("selector", [None, object(), SimpleNamespace(select=None)])
def test_action_selector_dictionary_rejects_invalid_selectors(selector):
    with pytest.raises(TypeError, match="fast_agent.*must implement ActionSelector"):
        MobileAgent("Select", action_selectors={"fast_agent": selector})
