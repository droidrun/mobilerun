from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import httpx

from droidrun.agent.providers import (
    VARIANT_ENV_KEY_SLOT,
    ProviderFamilySpec,
    ProviderVariantSpec,
    list_provider_families,
    resolve_provider_variant,
)
from droidrun.config_manager.config_manager import DroidConfig, LLMProfile
from droidrun.config_manager.env_keys import load_env_keys, save_env_keys

DEFAULT_KWARGS_BY_VARIANT: dict[str, dict[str, int]] = {
    "anthropic_oauth": {"max_tokens": 1024},
    "gemini_oauth_code_assist": {"max_tokens": 1024},
}

HIDDEN_ROLE_FALLBACKS: tuple[str, ...] = ("app_opener", "structured_output")
_ZAI_GLOBAL_BASE_URL = "https://api.z.ai/api/paas/v4"
_ZAI_CODING_GLOBAL_BASE_URL = "https://api.z.ai/api/coding/paas/v4"


@dataclass(frozen=True)
class SetupSelection:
    family_id: str
    variant_id: str
    auth_mode: str
    model: str
    api_key: str | None = None
    api_key_source: str = "auto"
    base_url: str | None = None
    credential_path: str | None = None


def family_choices() -> tuple[ProviderFamilySpec, ...]:
    return list_provider_families()


def auth_mode_choices(family_id: str) -> tuple[str, ...]:
    family = next(f for f in list_provider_families() if f.id == family_id)
    return tuple(variant.auth_mode for variant in family.variants)


def variant_models(family_id: str, auth_mode: str) -> tuple[str, ...]:
    variant = resolve_provider_variant(family_id, auth_mode)
    return tuple(variant.models)


def _probe_zai_chat_completions(
    *,
    base_url: str,
    api_key: str,
    model_id: str,
    timeout_seconds: float = 5.0,
) -> bool:
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            json={
                "model": model_id,
                "stream": False,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "ping"}],
            },
            timeout=timeout_seconds,
        )
        return response.is_success
    except Exception:
        return False


def _resolve_zai_selection(
    selection: SetupSelection,
    base_url: str | None,
) -> tuple[str | None, str]:
    """Resolve the effective ZAI base URL and model.

    We only support the two user-facing modes:
    - api_key -> global endpoint
    - coding_api -> coding-global endpoint

    For coding_api, probe glm-5 first and fall back to glm-4.7 when the
    provided token/plan does not expose glm-5 on the coding endpoint.
    """
    effective_base_url = base_url
    effective_model = selection.model

    if selection.family_id != "zai":
        return effective_base_url, effective_model

    if not effective_base_url:
        effective_base_url = (
            _ZAI_CODING_GLOBAL_BASE_URL
            if selection.auth_mode == "coding_api"
            else _ZAI_GLOBAL_BASE_URL
        )

    if (
        selection.auth_mode == "coding_api"
        and selection.api_key
        and selection.model == "glm-5"
        and not _probe_zai_chat_completions(
            base_url=effective_base_url,
            api_key=selection.api_key,
            model_id="glm-5",
        )
        and _probe_zai_chat_completions(
            base_url=effective_base_url,
            api_key=selection.api_key,
            model_id="glm-4.7",
        )
    ):
        effective_model = "glm-4.7"

    return effective_base_url, effective_model


def create_profile_for_variant(
    variant: ProviderVariantSpec,
    selection: SetupSelection,
    *,
    temperature: float = 0.2,
) -> LLMProfile:
    base_url = selection.base_url or variant.base_url
    base_url, resolved_model = _resolve_zai_selection(selection, base_url)
    kwargs: dict[str, str | int] = dict(DEFAULT_KWARGS_BY_VARIANT.get(variant.id, {}))
    env_slot = VARIANT_ENV_KEY_SLOT.get(variant.id)
    runtime_provider_name = (
        variant.runtime_transport_provider_name or variant.runtime_provider_name
    )

    if variant.id == "OpenAILike":
        if selection.api_key_source != "env":
            kwargs["api_key"] = selection.api_key or "stub"
    elif variant.id in {"ZAI", "ZAI_Coding"}:
        if selection.api_key_source != "env":
            kwargs["api_key"] = selection.api_key or "stub"
    # OpenAI models require temperature=1
    if selection.family_id == "openai":
        temperature = 1.0

    return LLMProfile(
        provider=runtime_provider_name,
        provider_family=selection.family_id,
        auth_mode=selection.auth_mode,
        model=resolved_model,
        temperature=temperature,
        api_key_source=selection.api_key_source,
        base_url=base_url,
        api_base=(
            base_url
            if runtime_provider_name in {"OpenAILike", "MiniMax"}
            else None
        ),
        credential_path=selection.credential_path or variant.credential_path,
        kwargs=kwargs if env_slot is None else {},
    )


def apply_selection_to_roles(
    config: DroidConfig,
    selection: SetupSelection,
    roles: Iterable[str],
) -> DroidConfig:
    variant = resolve_provider_variant(selection.family_id, selection.auth_mode)
    env_slot = VARIANT_ENV_KEY_SLOT.get(variant.id)
    if selection.api_key and env_slot and selection.api_key_source != "env":
        existing = load_env_keys()
        existing[env_slot] = selection.api_key
        try:
            save_env_keys(existing)
        except OSError:
            pass

    if variant.id == "anthropic_oauth":
        config.agent.streaming = False

    for role in roles:
        if role not in config.llm_profiles:
            continue
        current = config.llm_profiles[role]
        config.llm_profiles[role] = create_profile_for_variant(
            variant,
            selection,
            temperature=current.temperature,
        )

    if "fast_agent" in roles:
        fast_agent_profile = config.llm_profiles.get("fast_agent")
        if fast_agent_profile is not None:
            for hidden_role in HIDDEN_ROLE_FALLBACKS:
                if hidden_role not in config.llm_profiles:
                    continue
                current = config.llm_profiles[hidden_role]
                config.llm_profiles[hidden_role] = LLMProfile(
                    provider=fast_agent_profile.provider,
                    model=fast_agent_profile.model,
                    temperature=current.temperature,
                    base_url=fast_agent_profile.base_url,
                    api_base=fast_agent_profile.api_base,
                    provider_family=fast_agent_profile.provider_family,
                    auth_mode=fast_agent_profile.auth_mode,
                    api_key_source=fast_agent_profile.api_key_source,
                    credential_path=fast_agent_profile.credential_path,
                    kwargs=dict(fast_agent_profile.kwargs),
                )

    return config
