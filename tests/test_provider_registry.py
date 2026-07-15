from mobilerun.agent.providers.registry import (
    list_models_for_variant,
    resolve_provider_variant,
)
from mobilerun.config_manager.config_manager import LLMProfile, MobileConfig


def test_gemini_api_key_catalog_uses_current_flash_models() -> None:
    variant = resolve_provider_variant("gemini", "api_key")
    models = list_models_for_variant("gemini", "api_key")

    assert variant.default_model == "gemini-3.1-pro-preview"
    assert models == (
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite",
    )
    assert "gemini-3.1-flash-lite-preview" not in models


def test_gemini_oauth_catalog_uses_antigravity_consumer_models() -> None:
    variant = resolve_provider_variant("gemini", "oauth")
    models = list_models_for_variant("gemini", "oauth")

    assert variant.default_model == "gemini-3.5-flash-low"
    assert models == (
        "gemini-3.5-flash-low",
        "gemini-3.5-flash-extra-low",
        "gemini-3-flash-agent",
        "gemini-3-flash",
        "gemini-pro-agent",
        "gemini-3.1-pro-low",
    )
    # Deprecated gemini-cli / Code-Assist-for-individuals models are gone.
    assert "gemini-2.5-pro" not in models
    assert "gemini-2.5-flash" not in models
    assert "gemini-2.5-flash-lite" not in models
    assert "gemini-3.5-flash" not in models


def test_anthropic_catalogs_include_opus_4_8_without_changing_defaults() -> None:
    api_key_variant = resolve_provider_variant("anthropic", "api_key")
    api_key_models = list_models_for_variant("anthropic", "api_key")
    oauth_variant = resolve_provider_variant("anthropic", "oauth")
    oauth_models = list_models_for_variant("anthropic", "oauth")

    assert api_key_variant.default_model == "claude-sonnet-4-6"
    assert api_key_models == (
        "claude-sonnet-4-6",
        "claude-opus-4-8",
        "claude-opus-4-6",
        "claude-haiku-4-5",
    )
    assert oauth_variant.default_model == "claude-opus-4-7"
    assert oauth_models == (
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-haiku-4-5",
    )


def test_openai_oauth_catalog_hides_unsupported_codex_model() -> None:
    variant = resolve_provider_variant("openai", "oauth")
    models = list_models_for_variant("openai", "oauth")

    assert variant.default_model == "gpt-5.5"
    assert models == ("gpt-5.5", "gpt-5.4", "gpt-5.4-mini")
    assert "gpt-5.3-codex" not in models


def test_openai_api_key_catalog_uses_current_default_model() -> None:
    variant = resolve_provider_variant("openai", "api_key")
    models = list_models_for_variant("openai", "api_key")

    assert variant.default_model == "gpt-5.5"
    assert models == ("gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano")


def test_litellm_family_resolves_api_key_variant() -> None:
    variant = resolve_provider_variant("litellm", "api_key")

    assert variant.id == "LiteLLM"
    assert variant.runtime_provider_name == "LiteLLM"
    assert variant.requires_api_key is True
    assert variant.default_model is None
    assert variant.models == ()


def test_default_profiles_use_stable_gemini_flash_lite() -> None:
    config = MobileConfig()

    assert LLMProfile().model == "gemini-3.1-flash-lite"
    assert {profile.model for profile in config.llm_profiles.values()} == {
        "gemini-3.1-flash-lite"
    }
