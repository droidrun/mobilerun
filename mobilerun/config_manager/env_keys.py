"""API key lookup and persistence helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path

from droidrun.config_manager.credential_paths import AUTH_PROFILES_PATH

API_KEY_ENV_VARS = {
    "google": "GOOGLE_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "zai": "ZAI_API_KEY",
    "minimax": "MINIMAX_API_KEY",
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
        data = json.loads(AUTH_PROFILES_PATH.read_text(encoding="utf-8"))
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
    AUTH_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if AUTH_PROFILES_PATH.exists():
        try:
            loaded = json.loads(AUTH_PROFILES_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}

    api_keys = existing.get(_API_KEYS_SECTION, {})
    if not isinstance(api_keys, dict):
        api_keys = {}

    for slot, val in keys.items():
        env_var = API_KEY_ENV_VARS.get(slot)
        if not env_var:
            continue
        if val:
            api_keys[slot] = val
            os.environ[env_var] = val
        else:
            api_keys.pop(slot, None)
            os.environ.pop(env_var, None)

    existing[_API_KEYS_SECTION] = api_keys

    tmp_path = AUTH_PROFILES_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    os.replace(tmp_path, AUTH_PROFILES_PATH)
    try:
        os.chmod(AUTH_PROFILES_PATH, 0o600)
    except OSError:
        pass
