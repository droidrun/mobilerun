import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from click.testing import CliRunner

import mobilerun.cli.run as cli_run
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
            return SimpleNamespace(success=True, reason=None, steps=0)

        return done().__await__()


def _run(monkeypatch, config, **kwargs):
    monkeypatch.setattr(cli_run.ConfigLoader, "load", lambda _path=None: config)
    monkeypatch.setattr(cli_run, "_setup_cli_logging", lambda _: None)
    monkeypatch.setattr(cli_run, "print_telemetry_message", lambda **kw: None)
    monkeypatch.setattr(cli_run, "_cleanup_android_keyboard", AsyncMock())
    load_llm = Mock()
    monkeypatch.setattr(cli_run, "load_llm", load_llm)
    agents = []

    def agent(**agent_kwargs):
        agents.append(agent_kwargs)
        return SimpleNamespace(run=lambda: _Done())

    monkeypatch.setattr(cli_run, "MobileAgent", agent)
    success = asyncio.run(cli_run.run_command("Open Clock", **kwargs))
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


def test_typesafe_needs_no_model_and_follows_reasoning(monkeypatch):
    success, agents, _ = _run(
        monkeypatch, MobileConfig(), provider="typesafe", reasoning=True
    )

    assert success
    config = agents[0]["config"]
    assert config.decision_profiles == {"executor": DecisionProfile()}
    assert config.llm_profiles["delegate"] == config.llm_profiles["executor"]


def test_no_delegate_runs_jev_alone(monkeypatch):
    _, agents, _ = _run(
        monkeypatch, MobileConfig(), provider="TypeSafe", no_delegate=True
    )

    config = agents[0]["config"]
    assert config.agent.delegation is None
    assert "delegate" not in config.llm_profiles


def test_typesafe_without_any_llm_runs_jev_alone(monkeypatch):
    config = MobileConfig()
    config.llm_profiles.clear()
    success, agents, _ = _run(monkeypatch, config, provider="TypeSafe")

    assert success
    assert agents[0]["config"].agent.delegation is None


def test_delegate_hands_steps_to_the_configured_llm(monkeypatch):
    config = MobileConfig()
    config.llm_profiles["fast_agent"] = LLMProfile(
        provider="OpenAI", model="gpt-6.1-sol"
    )
    success, agents, _ = _run(monkeypatch, config, provider="TypeSafe")

    assert success
    config = agents[0]["config"]
    assert config.agent.delegation == DelegationConfig()
    assert config.llm_profiles["delegate"] == config.llm_profiles["fast_agent"]
    assert config.llm_profiles["delegate"] is not config.llm_profiles["fast_agent"]


def test_delegate_keeps_an_existing_delegate_profile(monkeypatch):
    config = MobileConfig()
    config.llm_profiles["delegate"] = LLMProfile(provider="Anthropic", model="x")
    _, agents, _ = _run(monkeypatch, config, provider="TypeSafe")

    assert agents[0]["config"].llm_profiles["delegate"].provider == "Anthropic"


def test_delegate_model_without_a_decision_model_fails_before_running(monkeypatch):
    success, agents, _ = _run(
        monkeypatch, MobileConfig(), delegate_model="OpenAI/gpt-6.1-sol"
    )

    assert not success
    assert agents == []


def test_delegate_model_conflicts_with_no_delegate(monkeypatch):
    success, agents, _ = _run(
        monkeypatch,
        MobileConfig(),
        provider="TypeSafe",
        no_delegate=True,
        delegate_model="gpt-6.1-sol",
    )

    assert not success
    assert agents == []


def test_no_delegate_turns_off_configured_delegation(monkeypatch):
    config = MobileConfig()
    config.decision_profiles["fast_agent"] = DecisionProfile()
    config.agent.delegation = DelegationConfig(delegate="gpt")
    _, agents, _ = _run(monkeypatch, config, no_delegate=True)

    assert agents[0]["config"].agent.delegation is None


def test_delegate_model_sets_provider_and_model(monkeypatch):
    _, agents, _ = _run(
        monkeypatch,
        MobileConfig(),
        provider="TypeSafe",
        delegate_model="openai_oauth/gpt-6.1-sol",
    )

    config = agents[0]["config"]
    delegate = config.llm_profiles["delegate"]
    assert config.agent.delegation == DelegationConfig()
    assert (delegate.provider, delegate.model) == ("openai_oauth", "gpt-6.1-sol")
    assert delegate.auth_mode == "oauth"
    assert delegate.credential_path


def test_delegate_model_alone_keeps_the_agent_provider(monkeypatch):
    config = MobileConfig()
    config.llm_profiles["fast_agent"] = LLMProfile(
        provider="OpenAILike", model="a", base_url="https://llm.test"
    )
    _, agents, _ = _run(
        monkeypatch, config, provider="TypeSafe", delegate_model="org/model-b"
    )

    assert agents[0]["config"].llm_profiles["delegate"] == LLMProfile(
        provider="OpenAILike", model="org/model-b", base_url="https://llm.test"
    )


def test_delegate_model_on_the_agent_provider_keeps_its_settings(monkeypatch):
    config = MobileConfig()
    config.llm_profiles["fast_agent"] = LLMProfile(
        provider="OpenAI", model="gpt-6.1-sol", kwargs={"timeout": 30}
    )
    _, agents, _ = _run(
        monkeypatch, config, provider="TypeSafe", delegate_model="openai/gpt-6-luna"
    )

    assert agents[0]["config"].llm_profiles["delegate"] == LLMProfile(
        provider="OpenAI", model="gpt-6-luna", kwargs={"timeout": 30}
    )


def test_delegate_model_replaces_the_configured_delegate_llm(monkeypatch):
    config = MobileConfig()
    config.decision_profiles["fast_agent"] = DecisionProfile()
    config.llm_profiles["gpt"] = LLMProfile(provider="OpenAI", model="gpt-6.1-sol")
    kept = DelegationConfig(delegate="gpt", min_confidence=0.5)
    config.agent.delegation = kept
    _, agents, _ = _run(monkeypatch, config, delegate_model="Anthropic/claude-x")

    config = agents[0]["config"]
    assert config.agent.delegation == kept
    assert config.llm_profiles["gpt"].provider == "Anthropic"
    assert "delegate" not in config.llm_profiles


def test_run_cli_forwards_the_delegate_options(monkeypatch):
    run_command = AsyncMock(return_value=(True, None, 0))
    monkeypatch.setattr(cli_run, "_run_task", run_command)

    result = CliRunner().invoke(
        cli_main.cli,
        ["run", "Open Clock", "-p", "TypeSafe", "--delegate-model", "openai/gpt-x"],
    )

    assert result.exit_code == 0, result.output
    kwargs = run_command.await_args.kwargs
    assert kwargs["provider"] == "TypeSafe"
    assert kwargs["model"] is None
    assert kwargs["no_delegate"] is False
    assert kwargs["delegate_model"] == "openai/gpt-x"

    result = CliRunner().invoke(
        cli_main.cli, ["run", "Open Clock", "-p", "TypeSafe", "--no-delegate"]
    )

    assert result.exit_code == 0, result.output
    assert run_command.await_args.kwargs["no_delegate"] is True


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
    _answers(monkeypatch, mode, "env", "main")

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


@pytest.mark.parametrize("own_delegate", [False, True])
def test_changing_the_llm_changes_a_delegate_that_follows_it(monkeypatch, own_delegate):
    config = MobileConfig()
    delegate = LLMProfile(provider="Anthropic", model="claude-x")
    config.llm_profiles["delegate"] = (
        delegate
        if own_delegate
        else LLMProfile(**vars(config.llm_profiles["fast_agent"]))
    )
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

    assert config.llm_profiles["fast_agent"].model == "gpt-6.1-sol"
    if own_delegate:
        assert config.llm_profiles["delegate"] == delegate
    else:
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


def test_typesafe_llm_url_goes_to_jev_not_the_helper_llms(monkeypatch):
    _, agents, _ = _run(
        monkeypatch,
        MobileConfig(),
        provider="TypeSafe",
        llm_url="https://typesafe.test",
    )

    assert "base_url" not in agents[0]
    profile = agents[0]["config"].decision_profiles["fast_agent"]
    assert profile.base_url == "https://typesafe.test"


def test_typesafe_keeps_a_delegation_to_another_decision_model(monkeypatch):
    config = MobileConfig()
    config.decision_profiles["jev"] = DecisionProfile()
    kept = DelegationConfig(delegate="jev", description="taps")
    config.agent.delegation = kept
    _, agents, _ = _run(monkeypatch, config, provider="TypeSafe")

    assert agents[0]["config"].agent.delegation == kept


def test_delegate_model_replaces_a_delegation_to_another_decision_model(monkeypatch):
    config = MobileConfig()
    config.decision_profiles["jev"] = DecisionProfile()
    config.agent.delegation = DelegationConfig(delegate="jev", description="taps")
    _, agents, _ = _run(
        monkeypatch, config, provider="TypeSafe", delegate_model="Anthropic/claude-x"
    )

    config = agents[0]["config"]
    assert config.agent.delegation == DelegationConfig()
    assert config.llm_profiles["delegate"].provider == "Anthropic"


def test_delegate_keeps_a_delegation_to_an_llm(monkeypatch):
    config = MobileConfig()
    config.llm_profiles["gpt"] = LLMProfile(provider="OpenAI", model="gpt-6.1-sol")
    kept = DelegationConfig(delegate="gpt", min_confidence=0.5)
    config.agent.delegation = kept
    _, agents, _ = _run(monkeypatch, config, provider="TypeSafe")

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
    _answers(monkeypatch, "delegate", "env", "keep")

    configure_wizard._configure_decision_model(_console(), config)

    assert config.decision_profiles["fast_agent"] == DecisionProfile(
        model="jev-1", api_key_source="env", base_url="https://typesafe.test"
    )
    assert config.agent.delegation is kept
    assert config.llm_profiles["delegate"] == LLMProfile(provider="OpenAI", model="g")


def test_wizard_delegate_copies_another_llm_when_fast_agent_is_missing(monkeypatch):
    config = MobileConfig()
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    del config.llm_profiles["fast_agent"]
    _answers(monkeypatch, "delegate", "env", "keep")

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
        monkeypatch, config, provider="TypeSafe", model="jev-2", no_delegate=True
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
        monkeypatch,
        config,
        provider="TypeSafe",
        model="jev-2",
        delegate_model="Anthropic/claude-x",
    )

    config = agents[0]["config"]
    assert config.agent.delegation == DelegationConfig(primary="jev")
    assert config.decision_profiles["jev"] == DecisionProfile(model="jev-2")
    assert config.llm_profiles["delegate"].model == "claude-x"


def test_wizard_defaults_to_delegation(monkeypatch):
    defaults = []

    def select(message, choices, default=None, **kw):
        defaults.append(default)
        return configure_wizard._BACK

    monkeypatch.setattr(configure_wizard, "_select_with_back", select)

    configure_wizard._configure_decision_model(_console(), MobileConfig())

    assert defaults == ["delegate"]


def test_wizard_delegate_can_use_another_llm(monkeypatch):
    config = MobileConfig()
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    chosen = LLMProfile(provider="Anthropic", model="claude-x")
    _answers(monkeypatch, "delegate", "env", "other")

    configure_wizard._configure_decision_model(_console(), config, lambda: chosen)

    assert config.agent.delegation == DelegationConfig()
    assert config.llm_profiles["delegate"] == chosen


def test_wizard_backing_out_of_another_llm_changes_nothing(monkeypatch):
    config = MobileConfig()
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    _answers(monkeypatch, "delegate", "env", "other")

    configure_wizard._configure_decision_model(_console(), config, lambda: None)

    assert config.decision_profiles == {}
    assert config.agent.delegation is None
    assert "delegate" not in config.llm_profiles


def test_typesafe_keeps_a_config_that_runs_jev_alone(monkeypatch):
    config = MobileConfig()
    config.decision_profiles["fast_agent"] = DecisionProfile(api_key_source="env")
    _, agents, _ = _run(monkeypatch, config, provider="TypeSafe")

    assert agents[0]["config"].agent.delegation is None


def test_delegate_model_is_not_shadowed_by_a_decision_profile(monkeypatch):
    config = MobileConfig()
    config.decision_profiles["fast_agent"] = DecisionProfile()
    config.decision_profiles["delegate"] = DecisionProfile(model="jev-big")
    config.agent.delegation = DelegationConfig(min_confidence=0.5)
    _, agents, _ = _run(monkeypatch, config, delegate_model="Anthropic/claude-x")

    config = agents[0]["config"]
    assert config.agent.delegation == DelegationConfig(
        delegate="delegate_2", min_confidence=0.5
    )
    assert config.llm_profiles["delegate_2"].model == "claude-x"


def test_delegate_model_leaves_a_shared_role_profile_alone(monkeypatch):
    config = MobileConfig()
    config.decision_profiles["fast_agent"] = DecisionProfile()
    manager = config.llm_profiles["manager"]
    config.agent.delegation = DelegationConfig(delegate="manager", min_confidence=0.4)
    _, agents, _ = _run(monkeypatch, config, delegate_model="Anthropic/claude-x")

    config = agents[0]["config"]
    assert config.llm_profiles["manager"] == manager
    assert config.llm_profiles["delegate"].model == "claude-x"
    assert config.agent.delegation == DelegationConfig(min_confidence=0.4)


def test_no_delegate_keeps_a_named_jev_primary_picking_actions(monkeypatch):
    config = MobileConfig()
    jev = DecisionProfile(model="jev-9", base_url="https://typesafe.test")
    config.decision_profiles["jev"] = jev
    config.agent.delegation = DelegationConfig(primary="jev")
    _, agents, _ = _run(monkeypatch, config, no_delegate=True)

    config = agents[0]["config"]
    assert config.agent.delegation is None
    assert config.decision_profiles["fast_agent"] == jev


def test_typesafe_no_delegate_starts_from_the_named_primary(monkeypatch):
    config = MobileConfig()
    config.decision_profiles["jev"] = DecisionProfile(
        model="jev-1", base_url="https://typesafe.test"
    )
    config.agent.delegation = DelegationConfig(primary="jev")
    _, agents, _ = _run(
        monkeypatch, config, provider="TypeSafe", model="jev-2", no_delegate=True
    )

    assert agents[0]["config"].decision_profiles["fast_agent"] == DecisionProfile(
        model="jev-2", base_url="https://typesafe.test"
    )


@pytest.mark.parametrize(
    "spec, provider, model",
    [
        ("deepseek/deepseek-flash", "DeepSeek", "deepseek-flash"),
        ("openrouter/x-ai/grok", "OpenRouter", "x-ai/grok"),
        ("xai_oauth/grok-4.7", "xai_oauth", "grok-4.7"),
    ],
)
def test_delegate_model_provider_names(monkeypatch, spec, provider, model):
    _, agents, _ = _run(
        monkeypatch, MobileConfig(), provider="TypeSafe", delegate_model=spec
    )

    delegate = agents[0]["config"].llm_profiles["delegate"]
    assert (delegate.provider, delegate.model) == (provider, model)


@pytest.mark.parametrize(
    "spec, provider, model",
    [
        ("anthropic/claude-x", "OpenRouter", "anthropic/claude-x"),
        ("Anthropic/claude-x", "Anthropic", "claude-x"),
    ],
)
def test_delegate_model_keeps_vendor_ids_on_openrouter(
    monkeypatch, spec, provider, model
):
    config = MobileConfig()
    config.llm_profiles["fast_agent"] = LLMProfile(provider="OpenRouter", model="a/b")
    _, agents, _ = _run(monkeypatch, config, provider="TypeSafe", delegate_model=spec)

    delegate = agents[0]["config"].llm_profiles["delegate"]
    assert (delegate.provider, delegate.model) == (provider, model)


def test_delegate_model_rejects_a_provider_without_an_endpoint(monkeypatch):
    success, agents, _ = _run(
        monkeypatch, MobileConfig(), provider="TypeSafe", delegate_model="OpenAILike/x"
    )

    assert not success
    assert agents == []


def test_wizard_keep_copies_a_delegate_shared_with_an_action_role(monkeypatch):
    config = MobileConfig()
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    config.llm_profiles["executor"] = LLMProfile(provider="Anthropic", model="claude-x")
    config.agent.delegation = DelegationConfig(delegate="executor", min_confidence=0.4)
    _answers(monkeypatch, "delegate", "env", "keep")

    configure_wizard._configure_decision_model(_console(), config)

    assert config.agent.delegation == DelegationConfig(min_confidence=0.4)
    assert config.llm_profiles["delegate"] == config.llm_profiles["executor"]
    assert config.llm_profiles["delegate"] is not config.llm_profiles["executor"]


def test_wizard_context_window_only_changes_ollama_profiles(monkeypatch):
    config = MobileConfig()
    config.llm_profiles["fast_agent"] = LLMProfile(provider="Anthropic", model="c")
    config.llm_profiles["delegate"] = LLMProfile(provider="Ollama", model="o")
    replies = iter(["context_window", "done"])
    monkeypatch.setattr(
        configure_wizard, "_select_with_back", lambda *a, **kw: next(replies)
    )
    monkeypatch.setattr(configure_wizard, "_prompt_int", lambda *a, **kw: 8192)

    configure_wizard._configure_advanced_settings(_console(), config)

    assert config.llm_profiles["delegate"].kwargs == {"context_window": 8192}
    assert "context_window" not in config.llm_profiles["fast_agent"].kwargs


@pytest.mark.parametrize(
    "spec, provider, model",
    [
        ("gemini/gemini-x", "GoogleGenAI", "gemini-x"),
        ("minimax/MiniMax-M3", "OpenAILike", "MiniMax-M3"),
    ],
)
def test_delegate_model_on_a_minimax_agent(monkeypatch, spec, provider, model):
    config = MobileConfig()
    config.llm_profiles["fast_agent"] = LLMProfile(
        provider="OpenAILike",
        provider_family="minimax",
        model="MiniMax-M2",
        base_url="https://minimax.test",
    )
    _, agents, _ = _run(monkeypatch, config, provider="TypeSafe", delegate_model=spec)

    delegate = agents[0]["config"].llm_profiles["delegate"]
    assert (delegate.provider, delegate.model) == (provider, model)


def test_no_delegate_keeps_a_named_llm_primary(monkeypatch):
    config = MobileConfig()
    big = LLMProfile(provider="Anthropic", model="claude-x")
    config.llm_profiles["big"] = big
    config.decision_profiles["jev"] = DecisionProfile()
    config.agent.delegation = DelegationConfig(primary="big", delegate="jev")
    _, agents, _ = _run(monkeypatch, config, no_delegate=True)

    config = agents[0]["config"]
    assert config.agent.delegation is None
    assert config.llm_profiles["fast_agent"] == big


def test_wizard_keep_survives_a_decision_profile_named_delegate(monkeypatch):
    config = MobileConfig()
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    gpt = LLMProfile(provider="OpenAI", model="gpt-6")
    config.llm_profiles["delegate"] = gpt
    config.decision_profiles["delegate"] = DecisionProfile()
    config.agent.delegation = DelegationConfig(min_confidence=0.3)
    _answers(monkeypatch, "delegate", "env", "keep")

    configure_wizard._configure_decision_model(_console(), config)

    delegation = config.agent.delegation
    assert delegation.min_confidence == 0.3
    assert config.llm_profiles[delegation.delegate] == gpt
    assert delegation.delegate not in config.decision_profiles


def test_changing_the_llm_keeps_a_delegate_with_its_own_endpoint(monkeypatch):
    config = MobileConfig()
    delegate = replace(config.llm_profiles["fast_agent"], base_url="https://own.test")
    config.llm_profiles["delegate"] = delegate
    monkeypatch.setattr(
        "mobilerun.agent.providers.setup_service.save_env_keys", lambda keys: None
    )
    from mobilerun.agent.providers.registry import resolve_provider_variant

    configure_wizard._apply_model_selection(
        config,
        family_id="openai",
        variant=resolve_provider_variant("openai", "api_key"),
        selected_auth_mode="api_key",
        selected_model="gpt-6.1-sol",
        selected_api_key="sk-test",
        selected_api_key_source="env",
        selected_base_url=None,
        credential_path=None,
    )

    assert config.llm_profiles["delegate"] == delegate


def test_cli_delegate_model_accepts_openai_oauth_spelling(monkeypatch):
    config = MobileConfig()
    monkeypatch.setattr("mobilerun.config_manager.ConfigLoader.load", lambda _: config)
    monkeypatch.setattr(cli_run, "_setup_cli_logging", lambda _: None)
    monkeypatch.setattr(
        "mobilerun.telemetry.print_telemetry_message", lambda **kw: None
    )
    monkeypatch.setattr(cli_run, "_cleanup_android_keyboard", AsyncMock())
    agents = []

    def agent(**kwargs):
        agents.append(kwargs)
        return SimpleNamespace(run=lambda: _Done())

    monkeypatch.setattr("mobilerun.MobileAgent", agent)
    # Polish keeps the constructor import at module scope; devx imports it lazily.
    if hasattr(cli_run, "MobileAgent"):
        monkeypatch.setattr(cli_run, "MobileAgent", agent)
    result = CliRunner().invoke(
        cli_main.cli,
        [
            "run",
            "Open Clock",
            "-p",
            "TypeSafe",
            "--delegate-model",
            "openai-oauth/gpt-x",
        ],
    )

    assert result.exit_code == 0, result.output
    profile = agents[0]["config"].llm_profiles["delegate"]
    assert (profile.provider, profile.model) == ("openai_oauth", "gpt-x")


def test_bad_delegation_spec_is_a_cli_usage_error(monkeypatch):
    monkeypatch.setattr(
        "mobilerun.config_manager.ConfigLoader.load", lambda _: MobileConfig()
    )
    result = CliRunner().invoke(
        cli_main.cli, ["run", "x", "--delegate-model", "openai/gpt-x", "-j"]
    )

    assert result.exit_code == 2, result.output
    assert "--delegate-model needs a decision model" in result.stderr
    assert result.stdout == ""
    assert "Traceback" not in result.output


def test_delegate_profile_config_error_is_a_json_failure(monkeypatch):
    monkeypatch.setattr(
        "mobilerun.config_manager.ConfigLoader.load", lambda _: MobileConfig()
    )
    result = CliRunner().invoke(
        cli_main.cli,
        ["run", "x", "-p", "TypeSafe", "--delegate-model", "openai-like/x", "-j"],
    )

    assert result.exit_code == 1, result.output
    assert "needs a base URL" in json.loads(result.stdout)["error"]
    assert "Traceback" not in result.output


def test_conflicting_delegate_options_fail_before_config_load(monkeypatch):
    load = Mock(side_effect=AssertionError("configuration must not be loaded"))
    monkeypatch.setattr("mobilerun.config_manager.ConfigLoader.load", load)
    result = CliRunner().invoke(
        cli_main.cli,
        ["run", "x", "-p", "TypeSafe", "--no-delegate", "--delegate-model", "a/b"],
    )

    assert result.exit_code == 2, result.output
    assert "--delegate-model cannot be used with --no-delegate" in result.stderr
    load.assert_not_called()
