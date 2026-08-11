import pytest

from mobilerun.agent.providers.registry import (
    list_models_for_variant,
    normalize_model_id_for_variant,
    resolve_provider_variant,
)
from mobilerun.config_manager.config_manager import LLMProfile, MobileConfig


def test_gemini_api_key_catalog_uses_current_flash_models() -> None:
    variant = resolve_provider_variant("gemini", "api_key")
    models = list_models_for_variant("gemini", "api_key")

    assert variant.default_model == "gemini-3.1-pro-preview"
    assert models == (
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
    )
    assert "gemini-3.1-flash-lite" not in models
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
        "gemini-3.6-flash-low",
        "gemini-3.6-flash-medium",
        "gemini-3.6-flash-high",
    )
    # Live legacy ids remain custom-only rather than advertised.
    assert "gemini-2.5-pro" not in models
    assert "gemini-2.5-flash" not in models
    assert "gemini-2.5-flash-lite" not in models
    assert "gemini-3.5-flash" not in models
    assert "gemini-3.1-flash-lite" not in models
    assert "gemini-3.1-pro-high" not in models


def test_anthropic_catalogs_include_claude_5_without_changing_defaults() -> None:
    api_key_variant = resolve_provider_variant("anthropic", "api_key")
    api_key_models = list_models_for_variant("anthropic", "api_key")
    oauth_variant = resolve_provider_variant("anthropic", "oauth")
    oauth_models = list_models_for_variant("anthropic", "oauth")

    assert api_key_variant.default_model == "claude-sonnet-4-6"
    assert api_key_models == (
        "claude-sonnet-4-6",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-opus-4-8",
        "claude-opus-4-6",
        "claude-haiku-4-5",
    )
    assert oauth_variant.default_model == "claude-opus-4-7"
    assert oauth_models == (
        "claude-opus-4-7",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-haiku-4-5",
    )


def test_openai_oauth_catalog_hides_unsupported_codex_model() -> None:
    variant = resolve_provider_variant("openai", "oauth")
    models = list_models_for_variant("openai", "oauth")

    assert variant.default_model == "gpt-5.5"
    assert models == (
        "gpt-5.5",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.4",
        "gpt-5.4-mini",
    )
    assert "gpt-5.3-codex" not in models


def test_openai_api_key_catalog_uses_current_default_model() -> None:
    variant = resolve_provider_variant("openai", "api_key")
    models = list_models_for_variant("openai", "api_key")

    assert variant.default_model == "gpt-5.5"
    assert models == (
        "gpt-5.5",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    )


def test_litellm_family_resolves_api_key_variant() -> None:
    variant = resolve_provider_variant("litellm", "api_key")

    assert variant.id == "LiteLLM"
    assert variant.runtime_provider_name == "LiteLLM"
    assert variant.requires_api_key is True
    assert variant.default_model is None
    assert variant.models == ()


def test_default_profiles_use_stable_gemini_flash_lite() -> None:
    config = MobileConfig()

    assert LLMProfile().model == "gemini-3.5-flash-lite"
    assert {profile.model for profile in config.llm_profiles.values()} == {
        "gemini-3.5-flash-lite"
    }


@pytest.mark.parametrize(
    ("auth_mode", "model", "expected"),
    [
        ("api_key", "gpt-5.6", "gpt-5.6-sol"),
        ("api_key", "openai/gpt-5.6", "gpt-5.6-sol"),
        ("oauth", "gpt-5.6", "gpt-5.6-sol"),
        ("oauth", "openai/gpt-5.6", "gpt-5.6-sol"),
        ("oauth", "openai-codex/gpt-5.6", "gpt-5.6-sol"),
        ("api_key", "gpt-5.6-terra", "gpt-5.6-terra"),
        ("oauth", "gpt-5.6-luna", "gpt-5.6-luna"),
    ],
)
def test_openai_model_aliases_normalize_to_catalog_ids(
    auth_mode: str, model: str, expected: str
) -> None:
    assert normalize_model_id_for_variant("openai", auth_mode, model) == expected


@pytest.mark.parametrize("auth_mode", ["api_key", "oauth"])
def test_openai_unknown_models_pass_through(auth_mode: str) -> None:
    model = "vendor/custom-model"

    assert normalize_model_id_for_variant("openai", auth_mode, model) == model
