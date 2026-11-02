import logging
from io import StringIO

import pytest
from rich.console import Console

import mobilerun.agent.providers.minimax as minimax_provider
import mobilerun.cli.configure_wizard as configure_wizard
import mobilerun.config_manager.config_manager as config_manager_module
from mobilerun.agent.providers.minimax import (
    MINIMAX_CHINA_BASE_URL,
    MINIMAX_GLOBAL_BASE_URL,
    MINIMAX_LEGACY_BASE_URL,
)
from mobilerun.agent.providers.registry import resolve_provider_variant
from mobilerun.agent.providers.setup_service import (
    SetupSelection,
    apply_selection_to_roles,
    create_profile_for_variant,
    family_choices,
)
from mobilerun.agent.utils.llm_picker import load_llm
from mobilerun.cli.configure_wizard import (
    ConfigureWizardCallbacks,
    ConfigureWizardState,
)
from mobilerun.config_manager.config_manager import LLMProfile, MobileConfig
from mobilerun.config_manager.env_keys import ApiKeySources


def test_minimax_registry_uses_current_global_endpoint() -> None:
    variant = resolve_provider_variant("minimax", "api_key")

    assert variant.base_url == MINIMAX_GLOBAL_BASE_URL
    assert variant.default_model == "MiniMax-M3"
    assert variant.models == (
        "MiniMax-M3",
        "MiniMax-M2.7",
        "MiniMax-M2.7-highspeed",
    )
    assert MINIMAX_CHINA_BASE_URL == "https://api.minimaxi.com/v1"
    assert MINIMAX_LEGACY_BASE_URL == "https://api.minimaxi.chat/v1"


def test_minimax_setup_profile_uses_openai_like_transport() -> None:
    variant = resolve_provider_variant("minimax", "api_key")
    profile = create_profile_for_variant(
        variant,
        SetupSelection(
            family_id="minimax",
            variant_id=variant.id,
            auth_mode="api_key",
            model="MiniMax-M2.7",
            api_key_source="env",
        ),
    )

    assert profile.provider == "OpenAILike"
    assert profile.provider_family == "minimax"
    assert profile.base_url == MINIMAX_GLOBAL_BASE_URL
    assert profile.api_base == MINIMAX_GLOBAL_BASE_URL
    assert profile.kwargs == {}


def test_minimax_issue_style_generation_normalizes_every_role_temperature() -> None:
    config = MobileConfig()
    roles = tuple(config.llm_profiles)
    selection = SetupSelection(
        family_id="minimax",
        variant_id="MiniMax",
        auth_mode="api_key",
        model="MiniMax-M2.7",
        api_key="env-test-key",
        api_key_source="env",
    )

    apply_selection_to_roles(config, selection, roles)

    assert set(config.llm_profiles) == {
        "manager",
        "executor",
        "fast_agent",
        "app_opener",
        "structured_output",
    }
    for role, profile in config.llm_profiles.items():
        assert profile.provider == "OpenAILike", role
        assert profile.provider_family == "minimax", role
        assert profile.model == "MiniMax-M2.7", role
        assert profile.base_url == MINIMAX_GLOBAL_BASE_URL, role
        assert profile.api_base == MINIMAX_GLOBAL_BASE_URL, role
        assert 0 < profile.temperature <= 1, role

    assert config.llm_profiles["manager"].temperature == 0.2
    assert config.llm_profiles["executor"].temperature == 0.1
    assert config.llm_profiles["fast_agent"].temperature == 0.2
    assert config.llm_profiles["app_opener"].temperature == 0.1
    assert config.llm_profiles["structured_output"].temperature == 0.1


def test_minimax_fast_agent_fallback_normalizes_hidden_role_temperatures() -> None:
    config = MobileConfig()
    selection = SetupSelection(
        family_id="minimax",
        variant_id="MiniMax",
        auth_mode="api_key",
        model="MiniMax-M2.7",
        api_key="env-test-key",
        api_key_source="env",
    )

    apply_selection_to_roles(config, selection, ("fast_agent",))

    assert config.llm_profiles["app_opener"].temperature == 0.1
    assert config.llm_profiles["structured_output"].temperature == 0.1


def test_non_minimax_generated_profile_preserves_zero_temperature() -> None:
    variant = resolve_provider_variant("gemini", "api_key")
    profile = create_profile_for_variant(
        variant,
        SetupSelection(
            family_id="gemini",
            variant_id=variant.id,
            auth_mode="api_key",
            model=variant.default_model or "",
            api_key_source="env",
        ),
        temperature=0.0,
    )

    assert profile.temperature == 0.0


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("env", "env-test-key"),
        ("file", "saved-test-key"),
        ("auto", "saved-test-key"),
    ],
)
def test_openai_like_minimax_profile_resolves_configured_key_source(
    monkeypatch, source: str, expected: str
) -> None:
    monkeypatch.setattr(
        config_manager_module,
        "load_env_key_sources",
        lambda: {
            "minimax": ApiKeySources(
                shell="env-test-key",
                saved="saved-test-key",
            )
        },
    )
    profile = LLMProfile(
        provider="OpenAILike",
        provider_family="minimax",
        auth_mode="api_key",
        model="MiniMax-M2.7",
        api_key_source=source,
        base_url=MINIMAX_GLOBAL_BASE_URL,
        api_base=MINIMAX_GLOBAL_BASE_URL,
    )

    assert profile.to_load_llm_kwargs()["api_key"] == expected


def test_legacy_minimax_alias_defaults_to_global_endpoint() -> None:
    llm = load_llm("MiniMax", model="MiniMax-M2.7", api_key="stub")

    assert type(llm).__name__ == "OpenAILike"
    assert llm.api_base == MINIMAX_GLOBAL_BASE_URL


def test_legacy_minimax_alias_uses_minimax_environment_key(monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-env-key")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-openai-key")

    llm = load_llm("MiniMax", model="MiniMax-M2.7")

    assert llm.api_key == "minimax-env-key"


def test_legacy_minimax_alias_prefers_explicit_api_key(monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-env-key")

    llm = load_llm(
        "MiniMax",
        model="MiniMax-M2.7",
        api_key="explicit-minimax-key",
    )

    assert llm.api_key == "explicit-minimax-key"


def test_legacy_minimax_alias_empty_explicit_key_uses_environment(monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-env-key")

    llm = load_llm(
        "MiniMax",
        model="MiniMax-M2.7",
        api_key="",
    )

    assert llm.api_key == "minimax-env-key"


def test_legacy_minimax_alias_never_falls_back_to_openai_key(monkeypatch) -> None:
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-openai-key")

    with pytest.raises(
        ValueError,
        match="Pass api_key explicitly or set MINIMAX_API_KEY",
    ):
        load_llm("MiniMax", model="MiniMax-M2.7")


def test_legacy_minimax_alias_honors_base_url_override() -> None:
    llm = load_llm(
        "MiniMax",
        model="MiniMax-M2.7",
        api_key="stub",
        base_url=MINIMAX_CHINA_BASE_URL,
    )

    assert llm.api_base == MINIMAX_CHINA_BASE_URL


def test_legacy_minimax_alias_prefers_explicit_api_base() -> None:
    custom_api_base = "https://gateway.example/v1"
    llm = load_llm(
        "MiniMax",
        model="MiniMax-M2.7",
        api_key="stub",
        base_url=MINIMAX_CHINA_BASE_URL,
        api_base=custom_api_base,
    )

    assert llm.api_base == custom_api_base


@pytest.mark.parametrize(
    "selected_endpoint",
    [MINIMAX_GLOBAL_BASE_URL, MINIMAX_CHINA_BASE_URL],
)
def test_minimax_interactive_endpoint_choices(
    monkeypatch, selected_endpoint: str
) -> None:
    captured = {}

    def fake_select(message, choices, *, default=None):
        captured["message"] = message
        captured["choices"] = choices
        captured["default"] = default
        return selected_endpoint

    monkeypatch.setattr(configure_wizard, "select_prompt", fake_select)
    variant = resolve_provider_variant("minimax", "api_key")

    assert configure_wizard._prompt_base_url_for_variant(variant) == selected_endpoint
    assert captured["message"] == "Choose MiniMax API region"
    assert captured["default"] == MINIMAX_GLOBAL_BASE_URL
    assert [choice.value for choice in captured["choices"]] == [
        MINIMAX_GLOBAL_BASE_URL,
        MINIMAX_CHINA_BASE_URL,
        configure_wizard._CUSTOM_MINIMAX_BASE_URL,
    ]


def test_minimax_interactive_custom_endpoint(monkeypatch) -> None:
    custom_base_url = "https://minimax-gateway.example/v1"
    monkeypatch.setattr(
        configure_wizard,
        "select_prompt",
        lambda *args, **kwargs: configure_wizard._CUSTOM_MINIMAX_BASE_URL,
    )
    monkeypatch.setattr(
        configure_wizard,
        "text_prompt",
        lambda message, **kwargs: custom_base_url,
    )

    assert (
        configure_wizard._prompt_base_url_for_variant(
            resolve_provider_variant("minimax", "api_key")
        )
        == custom_base_url
    )


@pytest.mark.parametrize(
    "selected_endpoint",
    [MINIMAX_GLOBAL_BASE_URL, MINIMAX_CHINA_BASE_URL],
)
def test_minimax_wizard_defaults_to_m3_for_each_region(
    monkeypatch, selected_endpoint: str
) -> None:
    config = MobileConfig()
    state = ConfigureWizardState(family_id="minimax")
    captured = {}

    def choose_default_model(
        models, *, family_id, auth_mode, default_model, allow_back=True
    ):
        captured["models"] = models
        captured["family_id"] = family_id
        captured["auth_mode"] = auth_mode
        captured["default_model"] = default_model
        captured["allow_back"] = allow_back
        return default_model

    monkeypatch.setattr(
        configure_wizard,
        "_prompt_model_choice",
        choose_default_model,
    )
    monkeypatch.setattr(
        configure_wizard,
        "_prompt_api_key_for_variant",
        lambda variant: ("stub", "env"),
    )
    monkeypatch.setattr(
        configure_wizard,
        "_prompt_base_url_for_variant",
        lambda variant: selected_endpoint,
    )
    callbacks = ConfigureWizardCallbacks(
        run_openai_oauth_login=lambda **kwargs: None,
        run_anthropic_oauth_login=lambda **kwargs: None,
        run_gemini_oauth_login=lambda **kwargs: None,
    )

    configured = configure_wizard._configure_provider_model(
        Console(file=StringIO(), force_terminal=False),
        config,
        callbacks,
        state,
        family_choices(),
        {family.id: family.display_name for family in family_choices()},
        provider_is_fixed=True,
        auth_mode_is_fixed=False,
        model_is_fixed=False,
        api_key=None,
        base_url=None,
    )

    assert configured is True
    assert captured["models"][0] == "MiniMax-M3"
    assert captured["family_id"] == "minimax"
    assert captured["auth_mode"] == "api_key"
    assert captured["default_model"] == "MiniMax-M3"
    assert state.selected_model == "MiniMax-M3"
    for profile in config.llm_profiles.values():
        assert profile.model == "MiniMax-M3"
        assert profile.base_url == selected_endpoint
        assert profile.api_base == selected_endpoint


@pytest.mark.parametrize(
    "selected_endpoint",
    [MINIMAX_GLOBAL_BASE_URL, MINIMAX_CHINA_BASE_URL],
)
def test_minimax_explicit_m27_remains_supported(
    selected_endpoint: str,
) -> None:
    variant = resolve_provider_variant("minimax", "api_key")
    profile = create_profile_for_variant(
        variant,
        SetupSelection(
            family_id="minimax",
            variant_id=variant.id,
            auth_mode="api_key",
            model="MiniMax-M2.7",
            api_key_source="env",
            base_url=selected_endpoint,
        ),
    )

    assert "MiniMax-M2.7" in variant.models
    assert profile.model == "MiniMax-M2.7"
    assert profile.base_url == selected_endpoint
    assert profile.api_base == selected_endpoint


def _run_advanced_temperature_configuration(
    monkeypatch,
    config: MobileConfig,
    temperatures: list[float],
) -> str:
    selections = iter(("temperature", "done"))
    prompted_temperatures = iter(temperatures)
    monkeypatch.setattr(
        configure_wizard,
        "_select_with_back",
        lambda *args, **kwargs: next(selections),
    )
    monkeypatch.setattr(
        configure_wizard,
        "_prompt_float",
        lambda *args, **kwargs: next(prompted_temperatures),
    )
    output = StringIO()
    configure_wizard._configure_advanced_settings(
        Console(file=output, force_terminal=False),
        config,
    )
    return output.getvalue()


def test_minimax_advanced_temperature_reprompts_until_valid(monkeypatch) -> None:
    config = MobileConfig()
    for profile in config.llm_profiles.values():
        profile.provider_family = "minimax"

    output = _run_advanced_temperature_configuration(
        monkeypatch,
        config,
        [-0.1, 0.0, 1.1, 0.1],
    )

    assert output.count("MiniMax temperature must be greater than 0") == 3
    assert all(profile.temperature == 0.1 for profile in config.llm_profiles.values())


def test_minimax_advanced_temperature_accepts_one(monkeypatch) -> None:
    config = MobileConfig()
    for profile in config.llm_profiles.values():
        profile.provider_family = "minimax"

    _run_advanced_temperature_configuration(monkeypatch, config, [1.0])

    assert all(profile.temperature == 1.0 for profile in config.llm_profiles.values())


def test_non_minimax_advanced_temperature_preserves_zero(monkeypatch) -> None:
    config = MobileConfig()

    _run_advanced_temperature_configuration(monkeypatch, config, [0.0])

    assert all(profile.temperature == 0.0 for profile in config.llm_profiles.values())


def test_noninteractive_minimax_preserves_explicit_base_url(monkeypatch) -> None:
    custom_base_url = "https://minimax-gateway.example/v1"
    state = ConfigureWizardState(
        family_id="minimax",
        selected_auth_mode="api_key",
        selected_model="MiniMax-M2.7",
        selected_api_key="stub",
        selected_api_key_source="file",
        selected_base_url=custom_base_url,
    )
    captured = {}

    def capture_selection(config, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(configure_wizard, "_apply_model_selection", capture_selection)
    monkeypatch.setattr(
        configure_wizard,
        "_prompt_base_url_for_variant",
        lambda variant: pytest.fail("noninteractive explicit base URL must not prompt"),
    )
    callbacks = ConfigureWizardCallbacks(
        run_openai_oauth_login=lambda **kwargs: None,
        run_anthropic_oauth_login=lambda **kwargs: None,
        run_gemini_oauth_login=lambda **kwargs: None,
    )

    configured = configure_wizard._configure_provider_model(
        Console(file=StringIO(), force_terminal=False),
        MobileConfig(),
        callbacks,
        state,
        family_choices(),
        {family.id: family.display_name for family in family_choices()},
        provider_is_fixed=True,
        auth_mode_is_fixed=True,
        model_is_fixed=True,
        api_key="stub",
        base_url=custom_base_url,
    )

    assert configured is True
    assert captured["selected_base_url"] == custom_base_url


def test_legacy_minimax_profile_warns_once_without_migration(caplog) -> None:
    minimax_provider._warn_about_legacy_endpoint_once.cache_clear()
    logger = logging.getLogger("mobilerun")
    previous_propagate = logger.propagate
    logger.propagate = True
    caplog.set_level(logging.WARNING, logger="mobilerun")
    profile = LLMProfile(
        provider="OpenAILike",
        provider_family="minimax",
        auth_mode="api_key",
        model="MiniMax-M2.7",
        base_url=MINIMAX_LEGACY_BASE_URL,
        api_base=MINIMAX_LEGACY_BASE_URL,
        kwargs={"api_key": "stub"},
    )

    try:
        profile.to_load_llm_kwargs()
        profile.to_load_llm_kwargs()
    finally:
        logger.propagate = previous_propagate
        minimax_provider._warn_about_legacy_endpoint_once.cache_clear()

    warnings = [
        record
        for record in caplog.records
        if "MiniMax profile uses the legacy endpoint" in record.message
    ]
    assert len(warnings) == 1
    assert profile.base_url == MINIMAX_LEGACY_BASE_URL
    assert profile.api_base == MINIMAX_LEGACY_BASE_URL
