import pytest

import mobilerun.config_manager.config_manager as config_manager_module
from mobilerun.agent.providers.orcarouter import ORCAROUTER_BASE_URL
from mobilerun.agent.providers.registry import (
    VARIANT_ENV_KEY_SLOT,
    list_models_for_variant,
    list_provider_families,
    resolve_provider_variant,
)
from mobilerun.agent.providers.setup_service import (
    SetupSelection,
    apply_selection_to_roles,
    create_profile_for_variant,
    family_choices,
)
from mobilerun.agent.utils.llm_picker import (
    SUPPORTED_PROVIDERS,
    load_llm,
    normalize_provider_name,
)
from mobilerun.config_manager.config_manager import LLMProfile, MobileConfig
from mobilerun.config_manager.env_keys import API_KEY_ENV_VARS, ApiKeySources


def test_orcarouter_is_a_named_provider_family() -> None:
    family_ids = [family.id for family in list_provider_families()]
    assert "orcarouter" in family_ids

    family = next(f for f in list_provider_families() if f.id == "orcarouter")
    assert family.display_name == "OrcaRouter"
    assert [choice.id for choice in family_choices() if choice.id == "orcarouter"]


def test_orcarouter_registry_defaults() -> None:
    variant = resolve_provider_variant("orcarouter", "api_key")

    assert variant.id == "OrcaRouter"
    assert variant.auth_mode == "api_key"
    assert variant.requires_api_key is True
    assert variant.requires_base_url is True
    assert variant.base_url == ORCAROUTER_BASE_URL == "https://api.orcarouter.ai/v1"
    assert variant.runtime_transport_provider_name == "OpenAILike"
    assert variant.default_model == "anthropic/claude-sonnet-5"
    assert list_models_for_variant("orcarouter", "api_key") == (
        "anthropic/claude-sonnet-5",
        "openai/gpt-5.6-sol",
        "google/gemini-3.1-pro-preview",
        "openai/gpt-5.4-mini",
    )


def test_orcarouter_env_key_slot_is_wired() -> None:
    assert VARIANT_ENV_KEY_SLOT["OrcaRouter"] == "orcarouter"
    assert API_KEY_ENV_VARS["orcarouter"] == "ORCAROUTER_API_KEY"


def test_orcarouter_setup_profile_uses_openai_like_transport() -> None:
    variant = resolve_provider_variant("orcarouter", "api_key")
    profile = create_profile_for_variant(
        variant,
        SetupSelection(
            family_id="orcarouter",
            variant_id=variant.id,
            auth_mode="api_key",
            model="openai/gpt-5.6-sol",
            api_key_source="env",
        ),
    )

    assert profile.provider == "OpenAILike"
    assert profile.provider_family == "orcarouter"
    assert profile.model == "openai/gpt-5.6-sol"
    assert profile.base_url == ORCAROUTER_BASE_URL
    assert profile.api_base == ORCAROUTER_BASE_URL
    assert profile.kwargs == {}


def test_orcarouter_accepts_any_catalog_model_id() -> None:
    variant = resolve_provider_variant("orcarouter", "api_key")
    profile = create_profile_for_variant(
        variant,
        SetupSelection(
            family_id="orcarouter",
            variant_id=variant.id,
            auth_mode="api_key",
            model="minimax/minimax-m3",
            api_key_source="env",
        ),
    )

    assert profile.model == "minimax/minimax-m3"


def test_orcarouter_selection_applies_to_every_role() -> None:
    config = MobileConfig()
    roles = tuple(config.llm_profiles)
    selection = SetupSelection(
        family_id="orcarouter",
        variant_id="OrcaRouter",
        auth_mode="api_key",
        model="anthropic/claude-sonnet-5",
        api_key_source="env",
    )

    apply_selection_to_roles(config, selection, roles)

    for role, profile in config.llm_profiles.items():
        assert profile.provider == "OpenAILike", role
        assert profile.provider_family == "orcarouter", role
        assert profile.model == "anthropic/claude-sonnet-5", role
        assert profile.api_base == ORCAROUTER_BASE_URL, role


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("env", "env-test-key"),
        ("file", "saved-test-key"),
        ("auto", "saved-test-key"),
    ],
)
def test_orcarouter_profile_resolves_configured_key_source(
    monkeypatch, source: str, expected: str
) -> None:
    monkeypatch.setattr(
        config_manager_module,
        "load_env_key_sources",
        lambda: {
            "orcarouter": ApiKeySources(
                shell="env-test-key",
                saved="saved-test-key",
            )
        },
    )
    profile = LLMProfile(
        provider="OpenAILike",
        provider_family="orcarouter",
        auth_mode="api_key",
        model="anthropic/claude-sonnet-5",
        api_key_source=source,
        base_url=ORCAROUTER_BASE_URL,
        api_base=ORCAROUTER_BASE_URL,
    )

    assert profile.to_load_llm_kwargs()["api_key"] == expected


def test_orcarouter_is_a_supported_runtime_provider() -> None:
    assert "OrcaRouter" in SUPPORTED_PROVIDERS
    assert normalize_provider_name("orcarouter") == "OrcaRouter"
    assert normalize_provider_name("orca") == "OrcaRouter"
    assert normalize_provider_name("OrcaRouter") == "OrcaRouter"


def test_orcarouter_alias_defaults_to_the_router_endpoint() -> None:
    llm = load_llm("OrcaRouter", model="openai/gpt-5.6-sol", api_key="stub")

    assert type(llm).__name__ == "OpenAILike"
    assert llm.api_base == ORCAROUTER_BASE_URL
    assert llm.is_chat_model is True
    assert llm.is_function_calling_model is True


def test_orcarouter_alias_uses_its_own_environment_key(monkeypatch) -> None:
    monkeypatch.setenv("ORCAROUTER_API_KEY", "orcarouter-env-key")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-openai-key")

    llm = load_llm("OrcaRouter", model="openai/gpt-5.6-sol")

    assert llm.api_key == "orcarouter-env-key"


def test_orcarouter_alias_prefers_explicit_api_key(monkeypatch) -> None:
    monkeypatch.setenv("ORCAROUTER_API_KEY", "orcarouter-env-key")

    llm = load_llm(
        "OrcaRouter",
        model="openai/gpt-5.6-sol",
        api_key="explicit-orcarouter-key",
    )

    assert llm.api_key == "explicit-orcarouter-key"


def test_orcarouter_alias_empty_explicit_key_uses_environment(monkeypatch) -> None:
    monkeypatch.setenv("ORCAROUTER_API_KEY", "orcarouter-env-key")

    llm = load_llm("OrcaRouter", model="openai/gpt-5.6-sol", api_key="")

    assert llm.api_key == "orcarouter-env-key"


def test_orcarouter_alias_never_falls_back_to_openai_key(monkeypatch) -> None:
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-openai-key")

    with pytest.raises(
        ValueError,
        match="Pass api_key explicitly or set ORCAROUTER_API_KEY",
    ):
        load_llm("OrcaRouter", model="openai/gpt-5.6-sol")


def test_orcarouter_alias_honors_base_url_override() -> None:
    custom = "https://gateway.example/v1"
    llm = load_llm(
        "OrcaRouter",
        model="openai/gpt-5.6-sol",
        api_key="stub",
        base_url=custom,
    )

    assert llm.api_base == custom


def test_orcarouter_wizard_prefills_the_router_base_url(monkeypatch) -> None:
    import mobilerun.cli.configure_wizard as configure_wizard

    captured = {}

    def fake_text_prompt(message, **kwargs):
        captured["message"] = message
        captured["default"] = kwargs.get("default")
        return kwargs.get("default", "")

    monkeypatch.setattr(configure_wizard, "text_prompt", fake_text_prompt)
    variant = resolve_provider_variant("orcarouter", "api_key")

    assert configure_wizard._prompt_base_url_for_variant(variant) == ORCAROUTER_BASE_URL
    assert captured["default"] == ORCAROUTER_BASE_URL
