from __future__ import annotations

from mobilerun.agent.providers.anthropic import (
    ANTHROPIC_API_DEFAULT_MODEL,
    ANTHROPIC_API_MODELS,
    ANTHROPIC_OAUTH_DEFAULT_MODEL,
    ANTHROPIC_OAUTH_MODELS,
)
from mobilerun.agent.providers.minimax import MINIMAX_GLOBAL_BASE_URL
from mobilerun.agent.providers.orcarouter import ORCAROUTER_BASE_URL
from mobilerun.agent.providers.types import (
    ProviderFamilySpec,
    ProviderVariantSpec,
)
from mobilerun.config_manager.credential_paths import (
    ANTHROPIC_OAUTH_CREDENTIAL_PATH,
    GEMINI_OAUTH_CREDENTIAL_PATH,
    OPENAI_OAUTH_CREDENTIAL_PATH,
)

# Canonical mapping from variant ID to env-key slot name.
# Imported by setup_service, configure_wizard, and config_manager.
VARIANT_ENV_KEY_SLOT: dict[str, str] = {
    "GoogleGenAI": "google",
    "OpenAIResponses": "openai",
    "Anthropic": "anthropic",
    "ZAI": "zai",
    "ZAI_Coding": "zai",
    "MiniMax": "minimax",
    "OrcaRouter": "orcarouter",
}

OPENAI_MODEL_ALIASES: dict[str, str] = {
    "gpt-5.6": "gpt-5.6-sol",
}


PROVIDER_FAMILIES: tuple[ProviderFamilySpec, ...] = (
    ProviderFamilySpec(
        id="gemini",
        display_name="Gemini",
        variants=(
            ProviderVariantSpec(
                id="GoogleGenAI",
                runtime_provider_name="GoogleGenAI",
                auth_mode="api_key",
                default_model="gemini-3.1-pro-preview",
                models=(
                    "gemini-3.5-flash",
                    "gemini-3.6-flash",
                    "gemini-3.5-flash-lite",
                    "gemini-3-flash-preview",
                    "gemini-3.1-pro-preview",
                ),
                requires_api_key=True,
            ),
            ProviderVariantSpec(
                id="gemini_oauth_code_assist",
                runtime_provider_name="gemini_oauth_code_assist",
                auth_mode="oauth",
                # Antigravity consumer entitlement; ids from fetchAvailableModels.
                default_model="gemini-3.5-flash-low",
                models=(
                    "gemini-3.5-flash-low",
                    "gemini-3.5-flash-extra-low",
                    "gemini-3-flash-agent",
                    "gemini-3-flash",
                    "gemini-pro-agent",
                    "gemini-3.1-pro-low",
                    "gemini-3.6-flash-low",
                    "gemini-3.6-flash-medium",
                    "gemini-3.6-flash-high",
                ),
                credential_path=str(GEMINI_OAUTH_CREDENTIAL_PATH),
            ),
        ),
    ),
    ProviderFamilySpec(
        id="openai",
        display_name="OpenAI",
        variants=(
            ProviderVariantSpec(
                id="OpenAIResponses",
                runtime_provider_name="OpenAIResponses",
                auth_mode="api_key",
                default_model="gpt-5.5",
                models=(
                    "gpt-5.5",
                    "gpt-5.6-sol",
                    "gpt-5.6-terra",
                    "gpt-5.6-luna",
                    "gpt-5.4",
                    "gpt-5.4-mini",
                    "gpt-5.4-nano",
                ),
                requires_api_key=True,
            ),
            ProviderVariantSpec(
                id="openai_oauth",
                runtime_provider_name="openai_oauth",
                auth_mode="oauth",
                default_model="gpt-5.5",
                models=(
                    "gpt-5.5",
                    "gpt-5.6-sol",
                    "gpt-5.6-terra",
                    "gpt-5.6-luna",
                    "gpt-5.4",
                    "gpt-5.4-mini",
                ),
                credential_path=str(OPENAI_OAUTH_CREDENTIAL_PATH),
            ),
        ),
        notes=("OpenAI OAuth uses a restricted model catalog.",),
    ),
    ProviderFamilySpec(
        id="anthropic",
        display_name="Anthropic",
        variants=(
            ProviderVariantSpec(
                id="Anthropic",
                runtime_provider_name="Anthropic",
                auth_mode="api_key",
                default_model=ANTHROPIC_API_DEFAULT_MODEL,
                models=ANTHROPIC_API_MODELS,
                requires_api_key=True,
            ),
            ProviderVariantSpec(
                id="anthropic_oauth",
                runtime_provider_name="anthropic_oauth",
                auth_mode="oauth",
                default_model=ANTHROPIC_OAUTH_DEFAULT_MODEL,
                models=ANTHROPIC_OAUTH_MODELS,
                credential_path=str(ANTHROPIC_OAUTH_CREDENTIAL_PATH),
            ),
        ),
    ),
    ProviderFamilySpec(
        id="ollama",
        display_name="Ollama",
        variants=(
            ProviderVariantSpec(
                id="Ollama",
                runtime_provider_name="Ollama",
                auth_mode="none",
                default_model="llama3.2:3b",
                models=(),
                requires_base_url=True,
                base_url="http://localhost:11434",
            ),
        ),
    ),
    ProviderFamilySpec(
        id="openai_like",
        display_name="OpenAI Compatible",
        variants=(
            ProviderVariantSpec(
                id="OpenAILike",
                runtime_provider_name="OpenAILike",
                auth_mode="api_key",
                default_model=None,
                models=(),
                requires_api_key=True,
                requires_base_url=True,
            ),
        ),
        notes=(
            "Use for OpenRouter, LM Studio, vLLM, and other OpenAI-compatible endpoints.",
        ),
    ),
    ProviderFamilySpec(
        id="minimax",
        display_name="MiniMax",
        variants=(
            ProviderVariantSpec(
                id="MiniMax",
                runtime_provider_name="MiniMax",
                runtime_transport_provider_name="OpenAILike",
                auth_mode="api_key",
                default_model="MiniMax-M3",
                models=(
                    "MiniMax-M3",
                    "MiniMax-M2.7",
                    "MiniMax-M2.5-highspeed",
                ),
                requires_api_key=True,
                requires_base_url=True,
                base_url=MINIMAX_GLOBAL_BASE_URL,
            ),
        ),
    ),
    ProviderFamilySpec(
        id="zai",
        display_name="ZAI",
        variants=(
            ProviderVariantSpec(
                id="ZAI",
                runtime_provider_name="ZAI",
                runtime_transport_provider_name="OpenAILike",
                auth_mode="api_key",
                default_model="glm-5",
                models=(
                    "glm-5",
                    "glm-5v-turbo",
                    "glm-4.7",
                ),
                requires_api_key=True,
                requires_base_url=True,
                base_url="https://api.z.ai/api/paas/v4",
            ),
            ProviderVariantSpec(
                id="ZAI_Coding",
                runtime_provider_name="ZAI",
                runtime_transport_provider_name="OpenAILike",
                auth_mode="coding_api",
                default_model="glm-4.7",
                models=("glm-4.7",),
                requires_api_key=True,
                requires_base_url=True,
                base_url="https://api.z.ai/api/coding/paas/v4",
            ),
        ),
        notes=(
            "ZAI is exposed as a first-class provider family while reusing the OpenAI-compatible transport.",
            "Use auth mode `coding_api` for the GLM Coding Plan endpoint.",
        ),
    ),
    ProviderFamilySpec(
        id="orcarouter",
        display_name="OrcaRouter",
        variants=(
            ProviderVariantSpec(
                id="OrcaRouter",
                runtime_provider_name="OrcaRouter",
                runtime_transport_provider_name="OpenAILike",
                auth_mode="api_key",
                default_model="anthropic/claude-sonnet-5",
                models=(
                    "anthropic/claude-sonnet-5",
                    "openai/gpt-5.6-sol",
                    "google/gemini-3.1-pro-preview",
                    "openai/gpt-5.4-mini",
                ),
                requires_api_key=True,
                requires_base_url=True,
                base_url=ORCAROUTER_BASE_URL,
            ),
        ),
        notes=(
            "OrcaRouter is exposed as a first-class provider family while reusing the OpenAI-compatible transport.",
            "Models use the `vendor/model` ids from the router catalog; any id the account can reach is accepted.",
        ),
    ),
)


def list_provider_families() -> tuple[ProviderFamilySpec, ...]:
    return PROVIDER_FAMILIES


def get_provider_family(family_id: str) -> ProviderFamilySpec:
    for family in PROVIDER_FAMILIES:
        if family.id == family_id:
            return family
    raise KeyError(f"Unknown provider family: {family_id}")


def list_auth_modes(family_id: str) -> tuple[str, ...]:
    family = get_provider_family(family_id)
    return tuple(variant.auth_mode for variant in family.variants)


def resolve_provider_variant(
    family_id: str, auth_mode: str | None = None
) -> ProviderVariantSpec:
    family = get_provider_family(family_id)
    if auth_mode is None:
        if len(family.variants) != 1:
            raise ValueError(
                f"Provider family {family_id} requires an auth mode selection."
            )
        return family.variants[0]
    for variant in family.variants:
        if variant.auth_mode == auth_mode:
            return variant
    raise KeyError(f"Unknown auth mode {auth_mode!r} for provider family {family_id}")


def list_models_for_variant(
    family_id: str, auth_mode: str | None = None
) -> tuple[str, ...]:
    return resolve_provider_variant(family_id, auth_mode).models


def normalize_model_id_for_variant(
    family_id: str, auth_mode: str | None, model_id: str
) -> str:
    """Normalize accepted model aliases to the canonical model id for a variant."""
    variant = resolve_provider_variant(family_id, auth_mode)
    allowed_model_ids = set(variant.models)

    alias_prefixes: tuple[str, ...] = ()
    if family_id == "openai" and auth_mode == "api_key":
        alias_prefixes = ("openai/",)
    elif family_id == "openai" and auth_mode == "oauth":
        alias_prefixes = ("openai-codex/", "openai/")

    candidate = model_id
    for prefix in alias_prefixes:
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break

    if family_id == "openai":
        candidate = OPENAI_MODEL_ALIASES.get(candidate, candidate)

    if candidate in allowed_model_ids:
        return candidate

    return model_id
