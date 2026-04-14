from __future__ import annotations

import json
import os
from pathlib import Path

from droidrun.agent.utils.oauth.anthropic_oauth_llm import (
    AnthropicOAuthLLM,
    DEFAULT_SETUP_TOKEN_SCOPE,
)
from droidrun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
    GeminiOAuthCodeAssistLLM,
)
from droidrun.agent.utils.oauth.openai_oauth_llm import (
    DEFAULT_OPENAI_OAUTH_CALLBACK_HOST,
    DEFAULT_OPENAI_OAUTH_CALLBACK_PATH,
    DEFAULT_OPENAI_OAUTH_CALLBACK_PORT,
    DEFAULT_OPENAI_OAUTH_CREDENTIAL_PATH,
    OpenAIOAuth,
)

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
    llm = OpenAIOAuth(model=model, oauth_credential_path=credential_path)
    llm.login(
        open_browser=open_browser,
        timeout_seconds=timeout,
        callback_host=callback_host,
        callback_port=callback_port,
        callback_path=callback_path,
        redirect_host=callback_host,
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
    llm = GeminiOAuthCodeAssistLLM(
        model=model or "gemini-3.1-pro-preview",
        credential_path=credential_path,
    )
    llm.login(
        open_browser=open_browser,
        timeout_seconds=timeout,
        callback_host=callback_host,
        callback_port=callback_port,
        callback_path=callback_path,
    )


def run_anthropic_setup_token_oauth(
    *,
    timeout: float = 300.0,
    callback_host: str = "127.0.0.1",
    callback_port: int = 0,
    callback_path: str = "/callback",
    open_browser: bool = True,
) -> str:
    llm = AnthropicOAuthLLM(
        credential_path=None,
        authorize_url="https://claude.com/cai/oauth/authorize",
        login_scope=DEFAULT_SETUP_TOKEN_SCOPE,
    )
    return llm.login(
        open_browser=open_browser,
        timeout_seconds=timeout,
        callback_host=callback_host,
        callback_port=callback_port,
        callback_path=callback_path,
        expires_in=SETUP_TOKEN_EXPIRES_IN_SECONDS,
    )


def save_anthropic_setup_token(credential_path: str, token: str) -> None:
    cred_path = Path(credential_path).expanduser()
    cred_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, object] = {}
    if cred_path.exists():
        try:
            loaded = json.loads(cred_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}

    existing["claudeAiOauth"] = {
        "accessToken": token,
        "refreshToken": None,
        "expiresAt": None,
        "scopes": [],
    }
    cred_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    os.chmod(cred_path, 0o600)


def run_anthropic_oauth_setup(credential_path: str) -> None:
    token = run_anthropic_setup_token_oauth()
    save_anthropic_setup_token(credential_path, token)


def get_default_openai_credential_path() -> str:
    return str(DEFAULT_OPENAI_OAUTH_CREDENTIAL_PATH)
