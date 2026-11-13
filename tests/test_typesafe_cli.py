import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from click.testing import CliRunner

from mobilerun.cli import configure_wizard
from mobilerun.cli import main as cli_main
from mobilerun.config_manager.config_manager import (
    DecisionProfile,
    DelegationConfig,
    LLMProfile,
    MobileConfig,
)


class _Done:
    async def stream_events(self):
        if False:
            yield None

    def __await__(self):
        async def done():
            return SimpleNamespace(success=True)

        return done().__await__()


def _run(monkeypatch, config, **kwargs):
    monkeypatch.setattr(cli_main.ConfigLoader, "load", lambda _path=None: config)
    monkeypatch.setattr(cli_main, "_setup_cli_logging", lambda _: None)
    monkeypatch.setattr(cli_main, "print_telemetry_message", lambda **kw: None)
    monkeypatch.setattr(cli_main, "_cleanup_android_keyboard", AsyncMock())
    load_llm = Mock()
    monkeypatch.setattr(cli_main, "load_llm", load_llm)
    agents = []

    def agent(**agent_kwargs):
        agents.append(agent_kwargs)
        return SimpleNamespace(run=lambda: _Done())

    monkeypatch.setattr(cli_main, "MobileAgent", agent)
    success = asyncio.run(cli_main.run_command("Open Clock", **kwargs))
    return success, agents, load_llm


def test_typesafe_provider_makes_jev_pick_actions(monkeypatch):
    success, agents, load_llm = _run(
        monkeypatch, MobileConfig(), provider="TypeSafe", model="jev-1.14.0"
    )

    assert success
    load_llm.assert_not_called()
    config = agents[0]["config"]
    assert agents[0]["action_selectors"] is None
    assert config.decision_profiles["fast_agent"] == DecisionProfile(model="jev-1.14.0")
    assert config.agent.delegation is None


def test_typesafe_needs_no_model_and_follows_reasoning(monkeypatch):
    success, agents, _ = _run(
        monkeypatch, MobileConfig(), provider="typesafe", reasoning=True
    )

    assert success
    profiles = agents[0]["config"].decision_profiles
    assert profiles == {"executor": DecisionProfile()}


def test_delegate_hands_steps_to_the_configured_llm(monkeypatch):
    config = MobileConfig()
    config.llm_profiles["fast_agent"] = LLMProfile(
        provider="OpenAI", model="gpt-6.1-sol"
    )
    success, agents, _ = _run(monkeypatch, config, provider="TypeSafe", delegate=True)

    assert success
    config = agents[0]["config"]
    assert config.agent.delegation == DelegationConfig()
    assert config.llm_profiles["delegate"] == config.llm_profiles["fast_agent"]
    assert config.llm_profiles["delegate"] is not config.llm_profiles["fast_agent"]


def test_delegate_keeps_an_existing_delegate_profile(monkeypatch):
    config = MobileConfig()
    config.llm_profiles["delegate"] = LLMProfile(provider="Anthropic", model="x")
    _, agents, _ = _run(monkeypatch, config, provider="TypeSafe", delegate=True)

    assert agents[0]["config"].llm_profiles["delegate"].provider == "Anthropic"


def test_delegate_without_a_decision_model_fails_before_running(monkeypatch):
    success, agents, _ = _run(monkeypatch, MobileConfig(), delegate=True)

    assert not success
    assert agents == []


def test_no_delegate_turns_off_configured_delegation(monkeypatch):
    config = MobileConfig()
    config.decision_profiles["fast_agent"] = DecisionProfile()
    config.agent.delegation = DelegationConfig(delegate="gpt")
    _, agents, _ = _run(monkeypatch, config, delegate=False)

    assert agents[0]["config"].agent.delegation is None


def test_run_cli_forwards_the_delegate_flag(monkeypatch):
    run_command = AsyncMock(return_value=True)
    monkeypatch.setattr(cli_main, "run_command", run_command)

    result = CliRunner().invoke(
        cli_main.cli, ["run", "Open Clock", "-p", "TypeSafe", "--delegate"]
    )

    assert result.exit_code == 0, result.output
    kwargs = run_command.await_args.kwargs
    assert kwargs["provider"] == "TypeSafe"
    assert kwargs["model"] is None
    assert kwargs["delegate"] is True


def _answers(monkeypatch, *answers):
    replies = iter(answers)
    monkeypatch.setattr(
        configure_wizard, "_select_with_back", lambda *a, **kw: next(replies)
    )


def _console():
    return SimpleNamespace(print=lambda *a, **kw: None)


@pytest.mark.parametrize("mode", ["alone", "delegate"])
def test_wizard_sets_jev_for_both_action_roles(monkeypatch, mode):
    config = MobileConfig()
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    _answers(monkeypatch, mode, "env")

    configure_wizard._configure_decision_model(_console(), config)

    assert config.decision_profiles == {
        "fast_agent": DecisionProfile(api_key_source="env"),
        "executor": DecisionProfile(api_key_source="env"),
    }
    if mode == "delegate":
        assert config.agent.delegation == DelegationConfig()
        assert config.llm_profiles["delegate"] == config.llm_profiles["fast_agent"]
    else:
        assert config.agent.delegation is None
    assert configure_wizard._decision_mode(config) == mode


def test_wizard_saves_a_pasted_typesafe_key(monkeypatch):
    config = MobileConfig()
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(
        configure_wizard,
        "load_env_key_sources",
        lambda: {"typesafe": SimpleNamespace(shell="", saved="")},
    )
    saved = Mock()
    monkeypatch.setattr(configure_wizard, "save_env_keys", saved)
    monkeypatch.setattr(configure_wizard, "text_prompt", lambda *a, **kw: "ts-new")
    _answers(monkeypatch, "alone", "paste")

    configure_wizard._configure_decision_model(_console(), config)

    saved.assert_called_once_with({"typesafe": "ts-new"})
    assert config.decision_profiles["fast_agent"].api_key_source == "file"


def test_wizard_turns_the_decision_model_off(monkeypatch):
    config = MobileConfig()
    config.decision_profiles["fast_agent"] = DecisionProfile()
    config.decision_profiles["executor"] = DecisionProfile()
    config.agent.delegation = DelegationConfig()
    _answers(monkeypatch, "off")

    configure_wizard._configure_decision_model(_console(), config)

    assert config.decision_profiles == {}
    assert config.agent.delegation is None
    assert configure_wizard._decision_mode(config) == "off"


def test_wizard_back_leaves_the_config_unchanged(monkeypatch):
    config = MobileConfig()
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    _answers(monkeypatch, "delegate", configure_wizard._BACK)

    configure_wizard._configure_decision_model(_console(), config)

    assert config.decision_profiles == {}
    assert config.agent.delegation is None


def test_changing_the_llm_also_changes_the_delegate(monkeypatch):
    config = MobileConfig()
    config.llm_profiles["delegate"] = LLMProfile(provider="GoogleGenAI", model="g")
    monkeypatch.setattr(
        "mobilerun.agent.providers.setup_service.save_env_keys", lambda keys: None
    )
    from mobilerun.agent.providers.registry import resolve_provider_variant

    variant = resolve_provider_variant("openai", "api_key")
    configure_wizard._apply_model_selection(
        config,
        family_id="openai",
        variant=variant,
        selected_auth_mode="api_key",
        selected_model="gpt-6.1-sol",
        selected_api_key="sk-test",
        selected_api_key_source="env",
        selected_base_url=None,
        credential_path=None,
    )

    assert config.llm_profiles["delegate"].model == "gpt-6.1-sol"
    assert config.llm_profiles["delegate"] == config.llm_profiles["fast_agent"]


def test_typesafe_flag_keeps_the_configured_profile(monkeypatch):
    config = MobileConfig()
    config.decision_profiles["fast_agent"] = DecisionProfile(
        model="jev-1",
        api_key_source="env",
        base_url="https://typesafe.test",
        kwargs={"timeout": 30},
    )
    _, agents, _ = _run(monkeypatch, config, provider="TypeSafe", model="jev-2")

    assert agents[0]["config"].decision_profiles["fast_agent"] == DecisionProfile(
        model="jev-2",
        api_key_source="env",
        base_url="https://typesafe.test",
        kwargs={"timeout": 30},
    )


def test_typesafe_base_url_goes_to_jev_not_the_helper_llms(monkeypatch):
    _, agents, _ = _run(
        monkeypatch,
        MobileConfig(),
        provider="TypeSafe",
        base_url="https://typesafe.test",
        delegate=True,
    )

    assert "base_url" not in agents[0]
    profile = agents[0]["config"].decision_profiles["fast_agent"]
    assert profile.base_url == "https://typesafe.test"


def test_delegate_replaces_a_delegation_to_another_decision_model(monkeypatch):
    config = MobileConfig()
    config.decision_profiles["jev"] = DecisionProfile()
    config.agent.delegation = DelegationConfig(delegate="jev", description="taps")
    _, agents, _ = _run(monkeypatch, config, provider="TypeSafe", delegate=True)

    config = agents[0]["config"]
    assert config.agent.delegation == DelegationConfig()
    assert config.llm_profiles["delegate"] == config.llm_profiles["fast_agent"]


def test_delegate_keeps_a_delegation_to_an_llm(monkeypatch):
    config = MobileConfig()
    config.llm_profiles["gpt"] = LLMProfile(provider="OpenAI", model="gpt-6.1-sol")
    kept = DelegationConfig(delegate="gpt", min_confidence=0.5)
    config.agent.delegation = kept
    _, agents, _ = _run(monkeypatch, config, provider="TypeSafe", delegate=True)

    assert agents[0]["config"].agent.delegation == kept
    assert "delegate" not in agents[0]["config"].llm_profiles


def test_wizard_ignores_an_empty_pasted_key(monkeypatch):
    config = MobileConfig()
    monkeypatch.setattr(
        configure_wizard,
        "load_env_key_sources",
        lambda: {"typesafe": SimpleNamespace(shell="", saved="")},
    )
    saved = Mock()
    monkeypatch.setattr(configure_wizard, "save_env_keys", saved)
    monkeypatch.setattr(configure_wizard, "text_prompt", lambda *a, **kw: "  ")
    _answers(monkeypatch, "alone", "paste")

    configure_wizard._configure_decision_model(_console(), config)

    saved.assert_not_called()
    assert config.decision_profiles == {}


def test_wizard_off_keeps_an_llm_only_delegation(monkeypatch):
    config = MobileConfig()
    kept = DelegationConfig(delegate="manager")
    config.agent.delegation = kept
    _answers(monkeypatch, "off")

    configure_wizard._configure_decision_model(_console(), config)

    assert config.agent.delegation is kept


def test_wizard_reselecting_jev_keeps_profile_and_delegation(monkeypatch):
    config = MobileConfig()
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    for role in ("fast_agent", "executor"):
        config.decision_profiles[role] = DecisionProfile(
            model="jev-1", base_url="https://typesafe.test"
        )
    config.llm_profiles["delegate"] = LLMProfile(provider="OpenAI", model="g")
    kept = DelegationConfig(min_confidence=0.5)
    config.agent.delegation = kept
    _answers(monkeypatch, "delegate", "env")

    configure_wizard._configure_decision_model(_console(), config)

    assert config.decision_profiles["fast_agent"] == DecisionProfile(
        model="jev-1", api_key_source="env", base_url="https://typesafe.test"
    )
    assert config.agent.delegation is kept


def test_wizard_delegate_copies_another_llm_when_fast_agent_is_missing(monkeypatch):
    config = MobileConfig()
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    del config.llm_profiles["fast_agent"]
    _answers(monkeypatch, "delegate", "env")

    configure_wizard._configure_decision_model(_console(), config)

    assert config.agent.delegation == DelegationConfig()
    assert config.llm_profiles["delegate"] == config.llm_profiles["manager"]


def test_wizard_delegate_without_any_llm_runs_jev_alone(monkeypatch):
    config = MobileConfig()
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    config.llm_profiles.clear()
    _answers(monkeypatch, "delegate", "env")

    configure_wizard._configure_decision_model(_console(), config)

    assert config.agent.delegation is None
    assert "fast_agent" in config.decision_profiles


def test_typesafe_flag_overrides_the_named_delegation_primary(monkeypatch):
    config = MobileConfig()
    config.decision_profiles["jev"] = DecisionProfile(model="jev-1", kwargs={"a": 1})
    config.llm_profiles["gpt"] = LLMProfile(provider="OpenAI", model="gpt-6.1-sol")
    config.agent.delegation = DelegationConfig(primary="jev", delegate="gpt")
    _, agents, _ = _run(monkeypatch, config, provider="TypeSafe", model="jev-2")

    profiles = agents[0]["config"].decision_profiles
    assert profiles == {"jev": DecisionProfile(model="jev-2", kwargs={"a": 1})}


def test_typesafe_flag_with_no_delegate_overrides_the_action_role(monkeypatch):
    config = MobileConfig()
    config.decision_profiles["jev"] = DecisionProfile(model="jev-1")
    config.agent.delegation = DelegationConfig(primary="jev", delegate="manager")
    _, agents, _ = _run(
        monkeypatch, config, provider="TypeSafe", model="jev-2", delegate=False
    )

    config = agents[0]["config"]
    assert config.agent.delegation is None
    assert config.decision_profiles["fast_agent"] == DecisionProfile(model="jev-2")


def test_typesafe_flag_reaches_the_loaded_primary_selector(monkeypatch):
    from mobilerun.agent.utils import selector_loader

    config = MobileConfig()
    config.decision_profiles["jev"] = DecisionProfile(model="jev-1")
    config.agent.delegation = DelegationConfig(primary="jev", delegate="manager")
    _, agents, _ = _run(monkeypatch, config, provider="TypeSafe", model="jev-2")
    loaded = []
    monkeypatch.setattr(
        selector_loader,
        "load_decision_models_from_profiles",
        lambda profiles, names: loaded.append(profiles[names[0]]) or {names[0]: Mock()},
    )
    monkeypatch.setattr(
        selector_loader,
        "load_llms_from_profiles",
        lambda profiles, profile_names, **kw: {profile_names[0]: Mock()},
    )

    selector_loader.load_delegating_selector(agents[0]["config"])

    assert loaded == [DecisionProfile(model="jev-2")]


@pytest.mark.parametrize("delegate_profile", ["delegate", "jev_big"])
def test_typesafe_delegate_keeps_the_named_primary(monkeypatch, delegate_profile):
    config = MobileConfig()
    config.decision_profiles["jev"] = DecisionProfile(model="jev-1")
    config.decision_profiles["jev_big"] = DecisionProfile(model="jev-big")
    config.agent.delegation = DelegationConfig(primary="jev", delegate=delegate_profile)
    _, agents, _ = _run(
        monkeypatch, config, provider="TypeSafe", model="jev-2", delegate=True
    )

    config = agents[0]["config"]
    assert config.agent.delegation == DelegationConfig(primary="jev")
    assert config.decision_profiles["jev"] == DecisionProfile(model="jev-2")
    assert config.llm_profiles["delegate"] == config.llm_profiles["fast_agent"]
