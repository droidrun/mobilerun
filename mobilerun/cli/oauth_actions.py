from __future__ import annotations

import os

from mobilerun.agent.utils.oauth.anthropic_oauth_llm import (
    DEFAULT_SETUP_TOKEN_SCOPE,
    AnthropicOAuthLLM,
)
from mobilerun.agent.utils.oauth.anthropic_oauth_llm import (
    _is_headless_environment as anthropic_headless,
)
from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
    DEFAULT_MODEL as GEMINI_OAUTH_DEFAULT_MODEL,
)
from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
    GeminiOAuthCodeAssistLLM,
)
from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
    _is_headless_environment as gemini_headless,
)
from mobilerun.agent.utils.oauth.grok_oauth_llm import (
    DEFAULT_GROK_MODEL,
    GrokOAuth,
)
from mobilerun.agent.utils.oauth.login_timeout import OAuthLoginDeadline
from mobilerun.agent.utils.oauth.openai_oauth_llm import (
    DEFAULT_OPENAI_OAUTH_CALLBACK_HOST,
    DEFAULT_OPENAI_OAUTH_CALLBACK_PATH,
    DEFAULT_OPENAI_OAUTH_CALLBACK_PORT,
    DEFAULT_OPENAI_OAUTH_CREDENTIAL_PATH,
    OpenAIOAuth,
)
from mobilerun.config_manager.auth_profile_store import AuthProfileStore

SETUP_TOKEN_EXPIRES_IN_SECONDS = 365 * 24 * 60 * 60


def run_openai_oauth_login(
    credential_path: str,
    model: str | None,
    timeout: float = 300.0,
    callback_host: str = DEFAULT_OPENAI_OAUTH_CALLBACK_HOST,
    callback_port: int = DEFAULT_OPENAI_OAUTH_CALLBACK_PORT,
    callback_path: str = DEFAULT_OPENAI_OAUTH_CALLBACK_PATH,
    open_browser: bool = True,
) -> None:
    deadline = OAuthLoginDeadline(
        timeout,
        timeout_message="OpenAI OAuth login timed out.",
    )
    llm = OpenAIOAuth(
        model=model,
        oauth_credential_path=credential_path,
        timeout=timeout,
    )
    llm.login(
        open_browser=open_browser,
        timeout_seconds=deadline.remaining(),
        callback_host=callback_host,
        callback_port=callback_port,
        callback_path=callback_path,
        redirect_host=callback_host,
        deadline=deadline,
    )


def run_gemini_oauth_login(
    credential_path: str,
    model: str | None,
    timeout: float = 300.0,
    callback_host: str = "127.0.0.1",
    callback_port: int = 0,
    callback_path: str = "/oauth2callback",
    open_browser: bool = True,
) -> None:
    deadline = OAuthLoginDeadline(
        timeout,
        timeout_message="Gemini OAuth login timed out.",
    )
    llm = GeminiOAuthCodeAssistLLM(
        model=model or GEMINI_OAUTH_DEFAULT_MODEL,
        credential_path=credential_path,
        timeout=timeout,
    )
    access_token = llm.login(
        open_browser=open_browser,
        timeout_seconds=deadline.remaining(),
        callback_host=callback_host,
        callback_port=callback_port,
        callback_path=callback_path,
        deadline=deadline,
        persist_credentials=False,
    )
    # Verify the Antigravity consumer entitlement resolves before declaring
    # success (catches scope / header / endpoint problems at login time). Raise
    # on failure so the caller does not print a misleading success message.
    try:
        models = llm.fetch_available_models(
            deadline=deadline,
            access_token=access_token,
        )
    except TimeoutError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Gemini OAuth login received a token, but the Antigravity entitlement "
            f"check (fetchAvailableModels) failed: {exc}. The login is not "
            "usable; verify your Google One / AI access and retry."
        ) from exc
    if not models:
        raise RuntimeError(
            "Gemini OAuth login returned no usable models; credentials were not saved."
        )
    llm._persist_credentials(deadline=deadline)
    print(f"✓ Gemini (Antigravity) login OK — {len(models)} models available.")


def run_grok_oauth_login(
    credential_path: str,
    model: str | None,
    timeout: float = 300.0,
    open_browser: bool = True,
    device_code: bool = False,
    no_browser: bool | None = None,
) -> None:
    """Authenticate directly with xAI and save Mobilerun-owned credentials."""
    deadline = OAuthLoginDeadline(
        timeout,
        timeout_message="XAI OAuth login timed out.",
    )
    llm = GrokOAuth(
        model=model or DEFAULT_GROK_MODEL,
        oauth_credential_path=credential_path,
        timeout=timeout,
    )
    llm.login(
        open_browser=(
            open_browser if no_browser is None else open_browser and not no_browser
        ),
        timeout_seconds=deadline.remaining(),
        device_code=device_code,
        deadline=deadline,
    )


def run_anthropic_setup_token_oauth(
    *,
    credential_path: str | None = None,
    timeout: float = 300.0,
    callback_host: str = "127.0.0.1",
    callback_port: int = 0,
    callback_path: str = "/callback",
    open_browser: bool = True,
) -> str:
    deadline = OAuthLoginDeadline(
        timeout,
        timeout_message="Anthropic OAuth login timed out.",
    )
    llm = AnthropicOAuthLLM(
        credential_path=None,
        authorize_url="https://claude.com/cai/oauth/authorize",
        login_scope=DEFAULT_SETUP_TOKEN_SCOPE,
        timeout=timeout,
    )
    token = llm.login(
        open_browser=open_browser,
        timeout_seconds=deadline.remaining(),
        callback_host=callback_host,
        callback_port=callback_port,
        callback_path=callback_path,
        expires_in=SETUP_TOKEN_EXPIRES_IN_SECONDS,
        deadline=deadline,
    )
    if credential_path is not None:
        save_anthropic_setup_token(
            credential_path,
            token,
            deadline=deadline,
        )
    return token


def save_anthropic_setup_token(
    credential_path: str,
    token: str,
    *,
    deadline: OAuthLoginDeadline | None = None,
) -> None:
    AuthProfileStore(credential_path).update_slot(
        "claudeAiOauth",
        {
            "accessToken": token,
            "refreshToken": None,
            "expiresAt": None,
            "scopes": [],
        },
        lock_timeout=deadline.remaining() if deadline is not None else None,
        before_commit=deadline.check if deadline is not None else None,
    )


def run_anthropic_oauth_setup(credential_path: str) -> None:
    run_anthropic_setup_token_oauth(credential_path=credential_path)


def get_default_openai_credential_path() -> str:
    return str(DEFAULT_OPENAI_OAUTH_CREDENTIAL_PATH)


def login_requires_input(service):
    """Use each provider's existing flow selection to identify manual prompts."""
    headless = {"anthropic": anthropic_headless, "gemini": gemini_headless}.get(service)
    return bool(headless and (
        headless() or os.environ.get("MOBILERUN_OAUTH_MANUAL", "").lower() in {"1", "true", "yes"}
    ))
