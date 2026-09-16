from __future__ import annotations

import os

REQUESTY_BASE_URL = "https://router.requesty.ai/v1"
REQUESTY_EU_BASE_URL = "https://router.eu.requesty.ai/v1"
REQUESTY_US_BASE_URL = "https://router.us.requesty.ai/v1"
REQUESTY_AP_BASE_URL = "https://router.ap.requesty.ai/v1"

REQUESTY_API_KEY_ENV_VAR = "REQUESTY_API_KEY"
REQUESTY_BASE_URL_ENV_VAR = "REQUESTY_BASE_URL"

# Requesty accepts either a full "<vendor>/<model>" catalog id or a managed
# policy id (a short, stable name for a Requesty-maintained routing chain).
# Both come from the live catalog: GET /v1/models and GET /v1/models/managed.
REQUESTY_DEFAULT_MODEL = "openai/gpt-4o-mini"
REQUESTY_MODELS: tuple[str, ...] = (
    REQUESTY_DEFAULT_MODEL,
    "gpt-6-astra",
    "gpt-5.4-mini",
    "claude-sonnet-5",
    "claude-haiku-4-5",
    "gemini-3.8-flash",
    "grok-4.6",
)


def resolve_requesty_base_url(base_url: str | None = None) -> str:
    """Return the Requesty base URL to use.

    An explicit ``base_url`` wins, then ``REQUESTY_BASE_URL`` (how users pick
    a regional router such as the EU endpoint), then the global default.
    """
    if isinstance(base_url, str) and base_url.strip():
        return base_url.strip()
    env_base_url = os.environ.get(REQUESTY_BASE_URL_ENV_VAR, "")
    if env_base_url.strip():
        return env_base_url.strip()
    return REQUESTY_BASE_URL
