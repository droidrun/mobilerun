"""API key lookup and persistence helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

from mobilerun.config_manager.auth_profile_store import AuthProfileStore
from mobilerun.config_manager.credential_paths import AUTH_PROFILES_PATH

API_KEY_ENV_VARS = {
    "google": "GOOGLE_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "xai": "XAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "zai": "ZAI_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "requesty": "REQUESTY_API_KEY",
}

_API_KEYS_SECTION = "apiKeys"


@dataclass(frozen=True)
class ApiKeySources:
    """API key values from the shell and the saved env file."""

    shell: str = ""
    saved: str = ""


def _load_saved_api_keys() -> dict[str, str]:
    """Read the apiKeys section from the shared auth-profiles.json."""
    if not AUTH_PROFILES_PATH.exists():
        return {}
    try:
        data = AuthProfileStore(AUTH_PROFILES_PATH).read_profile()
        section = data.get(_API_KEYS_SECTION)
        if isinstance(section, dict):
            return {k: str(v) for k, v in section.items() if v}
        return {}
    except Exception:
        return {}


def load_env_key_sources() -> dict[str, ApiKeySources]:
    """Load API keys from shell env vars and the shared auth-profiles file.

    The returned mapping keeps the two sources separate so callers can decide
    whether to prefer the live shell environment or the persisted file.
    """
    saved_keys = _load_saved_api_keys()
    result: dict[str, ApiKeySources] = {}

    for slot, env_var in API_KEY_ENV_VARS.items():
        result[slot] = ApiKeySources(
            shell=os.environ.get(env_var, "") or "",
            saved=saved_keys.get(slot, ""),
        )

    return result


def load_env_keys() -> dict[str, str]:
    """Load API keys. The saved file takes precedence over shell env vars.

    Returns:
        Dict mapping slot name (e.g. "google") to key value.
    """
    result: dict[str, str] = {}
    for slot, sources in load_env_key_sources().items():
        result[slot] = sources.saved or sources.shell
    return result


def resolve_env_key(slot: str, source: str = "auto") -> str:
    """Resolve an API key for a provider slot from a specific source.

    Args:
        slot: Provider slot name, e.g. "openai" or "anthropic".
        source: "auto" (saved file first, then shell), "env" (shell only), or
            "file" (saved env file only).
    """
    sources = load_env_key_sources().get(slot, ApiKeySources())
    if source == "env":
        return sources.shell
    if source == "file":
        return sources.saved
    return sources.saved or sources.shell


def save_env_keys(keys: dict[str, str]) -> None:
    """Persist API keys to the apiKeys section of auth-profiles.json.

    Also sets them as env vars in the current process.

    Args:
        keys: Dict mapping slot name (e.g. "google") to key value.
    """
    known_keys = {slot: val for slot, val in keys.items() if slot in API_KEY_ENV_VARS}

    def _update(existing: dict) -> None:
        api_keys = existing.get(_API_KEYS_SECTION, {})
        if not isinstance(api_keys, dict):
            api_keys = {}
        for slot, val in known_keys.items():
            if val:
                api_keys[slot] = val
            else:
                api_keys.pop(slot, None)
        existing[_API_KEYS_SECTION] = api_keys

    # Persist first. In particular, a malformed file must not be replaced or
    # leave the current process believing a key was saved successfully.
    AuthProfileStore(AUTH_PROFILES_PATH).update_profile(_update)
    for slot, val in known_keys.items():
        env_var = API_KEY_ENV_VARS[slot]
        if val:
            os.environ[env_var] = val
        else:
            os.environ.pop(env_var, None)
