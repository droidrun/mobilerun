import pytest

from mobilerun.agent.providers.registry import (
    list_models_for_variant,
    model_display_name_for_variant,
    normalize_model_id_for_variant,
    resolve_provider_variant,
)
from mobilerun.config_manager.config_manager import LLMProfile, MobileConfig


def test_gemini_api_key_catalog_uses_current_flash_models() -> None:
    variant = resolve_provider_variant("gemini", "api_key")
    models = list_models_for_variant("gemini", "api_key")

    assert models[0] == variant.default_model == LLMProfile().model
    assert variant.default_model == "gemini-3.8-flash"
    assert models == (
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro-preview",
    )
    assert "gemini-3-flash-preview" not in models
    assert "gemini-3.1-flash-lite" not in models
    assert "gemini-3.1-flash-lite-preview" not in models


def test_gemini_oauth_catalog_uses_antigravity_consumer_models() -> None:
    variant = resolve_provider_variant("gemini", "oauth")
    models = list_models_for_variant("gemini", "oauth")

    assert variant.default_model == "gemini-3.8-flash-tiered"
    assert models == (
        "gemini-3.8-flash-tiered",
        "gemini-3.8-flash-low",
        "gemini-3.8-flash-medium",
        "gemini-3.8-flash-high",
        "gemini-3.7-flash-tiered",
        "gemini-3.5-flash-lite",
        "gemini-pro-agent",
        "gemini-3.1-pro-low",
    )
    # Slow legacy ids remain custom-only rather than advertised.
    assert "gemini-3-flash" not in models
    assert "gemini-3.6-flash-high" not in models
    # Live legacy ids remain custom-only rather than advertised.
    assert "gemini-2.5-pro" not in models
    assert "gemini-2.5-flash" not in models
    assert "gemini-2.5-flash-lite" not in models
    assert "gemini-3.5-flash" not in models
    assert "gemini-3.1-flash-lite" not in models
    assert "gemini-3.1-pro-high" not in models


@pytest.mark.parametrize(
    ("family_id", "auth_mode", "model_id", "expected"),
    [
        (
            "gemini",
            "oauth",
            "gemini-3.8-flash-tiered",
            "gemini-3.8-flash",
        ),
        (
            "gemini",
            "oauth",
            "gemini-3.7-flash-tiered",
            "gemini-3.7-flash",
        ),
        (
            "gemini",
            "oauth",
            "gemini-3.8-flash-high",
            "gemini-3.8-flash-high",
        ),
        ("gemini", "api_key", "gemini-3.8-flash", "gemini-3.8-flash"),
        ("openai", "oauth", "gpt-6-astra", "gpt-6-astra"),
    ],
)
def test_model_display_names_preserve_canonical_ids_except_explicit_overrides(
    family_id: str,
    auth_mode: str,
    model_id: str,
    expected: str,
) -> None:
    assert model_display_name_for_variant(family_id, auth_mode, model_id) == expected


def test_anthropic_catalogs_list_the_current_lineup() -> None:
    api_key_variant = resolve_provider_variant("anthropic", "api_key")
    api_key_models = list_models_for_variant("anthropic", "api_key")
    oauth_variant = resolve_provider_variant("anthropic", "oauth")
    oauth_models = list_models_for_variant("anthropic", "oauth")

    assert api_key_variant.default_model == "claude-sonnet-5"
    assert api_key_models == (
        "claude-sonnet-5",
        "claude-opus-5-5",
        "claude-fable-5-1",
    )
    assert oauth_variant.default_model == "claude-sonnet-5"
    assert oauth_models == api_key_models


def test_openai_oauth_catalog_hides_unsupported_codex_model() -> None:
    variant = resolve_provider_variant("openai", "oauth")
    models = list_models_for_variant("openai", "oauth")

    assert variant.default_model == "gpt-6-astra"
    assert models == (
        "gpt-6-astra",
        "gpt-6-sol",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    )
    assert "gpt-5.3-codex" not in models
    assert "gpt-5.5" not in models


def test_openai_api_key_catalog_uses_current_default_model() -> None:
    variant = resolve_provider_variant("openai", "api_key")
    models = list_models_for_variant("openai", "api_key")

    assert variant.default_model == "gpt-6-astra"
    assert models == (
        "gpt-6-astra",
        "gpt-6-sol",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    )
    # Listed live, but does not drive the agent's XML tool protocol.
    assert "gpt-6-luna" not in models


def test_default_profiles_use_gemini_3_8_flash() -> None:
    config = MobileConfig()

    assert LLMProfile().model == "gemini-3.8-flash"
    assert {profile.model for profile in config.llm_profiles.values()} == {
        "gemini-3.8-flash"
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
        ("api_key", "gpt-6-astra", "gpt-6-astra"),
        ("oauth", "gpt-6-astra", "gpt-6-astra"),
        ("api_key", "openai/gpt-6-sol", "gpt-6-sol"),
        ("oauth", "openai-codex/gpt-6-luna", "gpt-6-luna"),
        ("api_key", "openai/gpt-5.5", "gpt-5.5"),
        ("oauth", "openai-codex/gpt-5.5", "gpt-5.5"),
        ("api_key", "openai/gpt-5.4-mini", "gpt-5.4-mini"),
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


@pytest.mark.parametrize("auth_mode", ["api_key", "oauth"])
def test_openai_gpt_6_alias_is_not_invented(auth_mode: str) -> None:
    assert normalize_model_id_for_variant("openai", auth_mode, "gpt-6") == "gpt-6"
