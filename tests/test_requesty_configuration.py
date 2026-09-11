import pytest

import mobilerun.config_manager.config_manager as config_manager_module
from mobilerun.agent.providers.registry import (
    VARIANT_ENV_KEY_SLOT,
    resolve_provider_variant,
)
from mobilerun.agent.providers.requesty import (
    REQUESTY_BASE_URL,
    REQUESTY_DEFAULT_MODEL,
    REQUESTY_EU_BASE_URL,
    REQUESTY_MODELS,
    resolve_requesty_base_url,
)
from mobilerun.agent.providers.setup_service import (
    SetupSelection,
    create_profile_for_variant,
    family_choices,
)
from mobilerun.agent.utils.llm_picker import (
    SUPPORTED_PROVIDERS,
    load_llm,
    normalize_provider_name,
)
from mobilerun.config_manager.config_manager import LLMProfile
from mobilerun.config_manager.env_keys import API_KEY_ENV_VARS, ApiKeySources


def test_requesty_registry_uses_global_router_endpoint() -> None:
    variant = resolve_provider_variant("requesty", "api_key")

    assert variant.id == "Requesty"
    assert variant.runtime_transport_provider_name == "OpenAILike"
    assert variant.base_url == REQUESTY_BASE_URL == "https://router.requesty.ai/v1"
    assert variant.default_model == REQUESTY_DEFAULT_MODEL == "openai/gpt-4o-mini"
    assert variant.models == REQUESTY_MODELS
    assert variant.models[0] == variant.default_model
    assert REQUESTY_EU_BASE_URL == "https://router.eu.requesty.ai/v1"


def test_requesty_family_is_offered_by_the_configure_wizard() -> None:
    assert "requesty" in {family.id for family in family_choices()}


def test_requesty_env_key_slot_is_wired() -> None:
    assert VARIANT_ENV_KEY_SLOT["Requesty"] == "requesty"
    assert API_KEY_ENV_VARS["requesty"] == "REQUESTY_API_KEY"


def test_requesty_is_a_supported_picker_provider() -> None:
    assert "Requesty" in SUPPORTED_PROVIDERS
    assert normalize_provider_name("requesty") == "Requesty"
    assert normalize_provider_name("Requesty") == "Requesty"


def test_requesty_setup_profile_uses_openai_like_transport() -> None:
    variant = resolve_provider_variant("requesty", "api_key")
    profile = create_profile_for_variant(
        variant,
        SetupSelection(
            family_id="requesty",
            variant_id=variant.id,
            auth_mode="api_key",
            model="claude-sonnet-5",
            api_key_source="env",
        ),
    )

    assert profile.provider == "OpenAILike"
    assert profile.provider_family == "requesty"
    assert profile.model == "claude-sonnet-5"
    assert profile.base_url == REQUESTY_BASE_URL
    assert profile.api_base == REQUESTY_BASE_URL
    assert profile.kwargs == {}


def test_requesty_setup_profile_keeps_regional_base_url_override() -> None:
    variant = resolve_provider_variant("requesty", "api_key")
    profile = create_profile_for_variant(
        variant,
        SetupSelection(
            family_id="requesty",
            variant_id=variant.id,
            auth_mode="api_key",
            model=REQUESTY_DEFAULT_MODEL,
            api_key_source="env",
            base_url=REQUESTY_EU_BASE_URL,
        ),
    )

    assert profile.base_url == REQUESTY_EU_BASE_URL
    assert profile.api_base == REQUESTY_EU_BASE_URL


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("env", "env-test-key"),
        ("file", "saved-test-key"),
        ("auto", "saved-test-key"),
    ],
)
def test_openai_like_requesty_profile_resolves_configured_key_source(
    monkeypatch, source: str, expected: str
) -> None:
    monkeypatch.setattr(
        config_manager_module,
        "load_env_key_sources",
        lambda: {
            "requesty": ApiKeySources(
                shell="env-test-key",
                saved="saved-test-key",
            )
        },
    )
    profile = LLMProfile(
        provider="OpenAILike",
        provider_family="requesty",
        auth_mode="api_key",
        model=REQUESTY_DEFAULT_MODEL,
        api_key_source=source,
        base_url=REQUESTY_BASE_URL,
        api_base=REQUESTY_BASE_URL,
    )

    assert profile.to_load_llm_kwargs()["api_key"] == expected


def test_requesty_alias_defaults_to_global_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("REQUESTY_BASE_URL", raising=False)

    llm = load_llm("Requesty", api_key="stub")

    assert type(llm).__name__ == "OpenAILike"
    assert llm.api_base == REQUESTY_BASE_URL
    assert llm.model == REQUESTY_DEFAULT_MODEL
    assert llm.metadata.is_chat_model is True
    assert llm.metadata.is_function_calling_model is True


def test_requesty_alias_uses_requesty_environment_key(monkeypatch) -> None:
    monkeypatch.setenv("REQUESTY_API_KEY", "requesty-env-key")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-openai-key")

    llm = load_llm("Requesty", model="gpt-5.4-mini")

    assert llm.api_key == "requesty-env-key"
    assert llm.model == "gpt-5.4-mini"


def test_requesty_alias_prefers_explicit_api_key(monkeypatch) -> None:
    monkeypatch.setenv("REQUESTY_API_KEY", "requesty-env-key")

    llm = load_llm("Requesty", api_key="explicit-requesty-key")

    assert llm.api_key == "explicit-requesty-key"


def test_requesty_alias_requires_an_api_key(monkeypatch) -> None:
    monkeypatch.delenv("REQUESTY_API_KEY", raising=False)

    with pytest.raises(ValueError, match="REQUESTY_API_KEY"):
        load_llm("Requesty", model=REQUESTY_DEFAULT_MODEL)


def test_requesty_alias_honors_base_url_kwarg(monkeypatch) -> None:
    monkeypatch.delenv("REQUESTY_BASE_URL", raising=False)

    llm = load_llm("Requesty", api_key="stub", base_url=REQUESTY_EU_BASE_URL)

    assert llm.api_base == REQUESTY_EU_BASE_URL


def test_requesty_alias_honors_base_url_environment_override(monkeypatch) -> None:
    monkeypatch.setenv("REQUESTY_BASE_URL", REQUESTY_EU_BASE_URL)

    llm = load_llm("Requesty", api_key="stub")

    assert llm.api_base == REQUESTY_EU_BASE_URL


def test_resolve_requesty_base_url_precedence(monkeypatch) -> None:
    monkeypatch.setenv("REQUESTY_BASE_URL", "https://router.us.requesty.ai/v1")

    assert resolve_requesty_base_url("  https://router.ap.requesty.ai/v1 ") == (
        "https://router.ap.requesty.ai/v1"
    )
    assert resolve_requesty_base_url("") == "https://router.us.requesty.ai/v1"

    monkeypatch.delenv("REQUESTY_BASE_URL")
    assert resolve_requesty_base_url(None) == REQUESTY_BASE_URL
