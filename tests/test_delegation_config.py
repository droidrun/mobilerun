import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from llama_index.core.llms.mock import MockLLM
from selector_helpers import CompletedRun

from mobilerun import DelegationConfig, MobileAgent, MobileConfig
from mobilerun.agent.droid import droid_agent
from mobilerun.agent.selectors import DecisionSelector, DelegatingSelector, LlamaIndexLLMSelector
from mobilerun.agent.utils import llm_loader, selector_loader
from mobilerun.cli import main as cli_main
from mobilerun.config_manager import DecisionProfile, LLMProfile
from mobilerun.config_manager.migrations import CURRENT_VERSION


def _config(reasoning=False, reverse=False):
    role = "executor" if reasoning else "fast_agent"
    config = MobileConfig()
    config.agent.reasoning = reasoning
    config.agent.delegation = DelegationConfig()
    config.telemetry.enabled = False
    config.logging.save_trajectory = "none"
    if reverse:
        config.llm_profiles[role] = LLMProfile(provider="OpenAI", model="test-primary")
        config.decision_profiles["delegate"] = DecisionProfile()
    else:
        config.decision_profiles[role] = DecisionProfile()
        config.llm_profiles["delegate"] = LLMProfile(provider="OpenAI", model="test-delegate")
    return config


def _mock_models(monkeypatch):
    model = SimpleNamespace(decide=AsyncMock())
    llm = MockLLM()
    decisions = Mock(side_effect=lambda profiles, names: {name: model for name in names})
    llms = Mock(side_effect=lambda profiles, profile_names, **kw: {name: llm for name in profile_names})
    monkeypatch.setattr(selector_loader, "load_decision_models_from_profiles", decisions)
    monkeypatch.setattr(selector_loader, "load_llms_from_profiles", llms)
    return model, llm, decisions, llms


@pytest.mark.parametrize("delegation", [None, {}])
def test_optional_delegation_config_round_trips(delegation):
    config = MobileConfig.from_dict({"agent": {"delegation": delegation}})
    restored = MobileConfig.from_dict(config.to_dict())
    assert restored.agent.delegation == (None if delegation is None else DelegationConfig())
    assert MobileConfig().agent.delegation is None


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_loader_resolves_both_model_families_and_only_overrides_llms(monkeypatch, reasoning, reverse):
    config = _config(reasoning, reverse)
    model, llm, decisions, llms = _mock_models(monkeypatch)
    selector = selector_loader.load_delegating_selector(config, temperature=0.2)
    role = "executor" if reasoning else "fast_agent"
    decision_role, llm_role = ("delegate", role) if reverse else (role, "delegate")
    decisions.assert_called_once_with(config.decision_profiles, [decision_role])
    llms.assert_called_once_with(
        config.llm_profiles, profile_names=[llm_role], **{llm_role: {"temperature": 0.2}},
    )
    decision_selector = selector.delegate if reverse else selector.primary
    llm_selector = selector.primary if reverse else selector.delegate
    assert isinstance(decision_selector, DecisionSelector)
    assert decision_selector.model is model
    assert isinstance(llm_selector, LlamaIndexLLMSelector)
    assert llm_selector.llm is llm


def test_loader_supports_explicit_primary_profile_name(monkeypatch):
    config = _config()
    config.decision_profiles["custom"] = config.decision_profiles.pop("fast_agent")
    config.agent.delegation.primary = "custom"
    _, _, decisions, _ = _mock_models(monkeypatch)
    selector_loader.load_delegating_selector(config)
    decisions.assert_called_once_with(config.decision_profiles, ["custom"])


@pytest.mark.parametrize("missing_role", ["primary", "delegate"])
def test_missing_delegation_profile_has_a_clear_error(monkeypatch, missing_role):
    config = _config()
    setattr(config.agent.delegation, missing_role, "missing")
    _mock_models(monkeypatch)
    with pytest.raises(KeyError, match="Selector profile 'missing' not found"):
        selector_loader.load_delegating_selector(config)


def test_loader_requires_delegation_config():
    with pytest.raises(ValueError, match="agent.delegation must be configured"):
        selector_loader.load_delegating_selector(MobileConfig())


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_mobile_agent_loads_configured_chain_without_unused_action_llms(monkeypatch, reasoning, reverse):
    config = _config(reasoning, reverse)
    _mock_models(monkeypatch)
    helper = MockLLM()
    roles = ["manager", "app_opener"] if reasoning else ["app_opener"]
    load_helpers = Mock(return_value=dict.fromkeys(roles, helper))
    monkeypatch.setattr(llm_loader, "load_llms_from_profiles", load_helpers)

    agent = MobileAgent("Select", config=config)

    assert isinstance(agent.action_selector, DelegatingSelector)
    assert agent.app_opener_llm is helper
    assert agent.executor_llm is None
    assert agent.fast_agent_llm is None
    if reasoning:
        assert agent.manager_agent.llm is helper
        assert agent.executor_agent.action_selector is agent.action_selector
    load_helpers.assert_called_once_with(config.llm_profiles, profile_names=roles)


@pytest.mark.parametrize("override", ["selector", "legacy_llms", "external_agent"])
def test_explicit_overrides_and_external_agents_do_not_load_configured_chain(monkeypatch, override):
    config = _config()
    load = Mock(side_effect=AssertionError("The configured chain must not be loaded"))
    monkeypatch.setattr(droid_agent, "load_delegating_selector", load)
    llm = MockLLM()
    kwargs = {}
    if override == "selector":
        kwargs["action_selectors"] = LlamaIndexLLMSelector(llm)
    elif override == "legacy_llms":
        kwargs["llms"] = llm
    else:
        config.agent.name = "external-agent"
    MobileAgent("Select", config=config, **kwargs)
    load.assert_not_called()


def test_cli_runs_new_config_with_existing_device_and_step_flags(monkeypatch, tmp_path):
    # Exercise file loading without depending on local benchmark configs or keys.
    config_path = tmp_path / "delegation.yaml"
    config_data = _config().to_dict()
    config_data["_version"] = CURRENT_VERSION
    config_path.write_text(
        yaml.safe_dump(config_data),
        encoding="utf-8",
    )
    _mock_models(monkeypatch)
    monkeypatch.setattr(llm_loader, "load_llms_from_profiles", Mock(return_value={"app_opener": MockLLM()}))
    monkeypatch.setattr(cli_main, "_setup_cli_logging", lambda _: None)
    monkeypatch.setattr(cli_main, "print_telemetry_message", lambda **kw: None)
    monkeypatch.setattr(cli_main, "_cleanup_android_keyboard", AsyncMock())
    agents = []

    def initialize(**kwargs):
        agents.append(MobileAgent(**kwargs))
        return SimpleNamespace(run=lambda: CompletedRun())

    monkeypatch.setattr(cli_main, "MobileAgent", initialize)
    assert asyncio.run(cli_main.run_command(
        "Check timer and write a note", config_path=str(config_path),
        device="emulator-5554", steps=30,
    )) is True
    assert len(agents) == 1
    agent = agents[0]
    assert isinstance(agent.action_selector, DelegatingSelector)
    assert agent.config.device.serial == "emulator-5554"
    assert agent.config.agent.max_steps == 30
