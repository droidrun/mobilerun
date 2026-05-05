import base64
import hashlib
import json
import os
import secrets
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from mobilerun.config_manager.credential_paths import ANTHROPIC_OAUTH_CREDENTIAL_PATH

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_API_BASE = "https://api.anthropic.com"
DEFAULT_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
DEFAULT_AUTHORIZE_URL = "https://platform.claude.com/oauth/authorize"
DEFAULT_MODERN_AUTHORIZE_URL = "https://claude.com/cai/oauth/authorize"
DEFAULT_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
DEFAULT_CREDENTIAL_PATH = str(ANTHROPIC_OAUTH_CREDENTIAL_PATH)
DEFAULT_LOGIN_SCOPE = "org:create_api_key user:profile"
DEFAULT_MODERN_LOGIN_SCOPE = (
    "org:create_api_key user:profile user:inference "
    "user:sessions:claude_code user:mcp_servers user:file_upload"
)
DEFAULT_SETUP_TOKEN_SCOPE = "user:inference"
DEFAULT_REFRESH_SCOPE = (
    "user:inference user:profile user:file_upload user:mcp_servers user:sessions:claude_code"
)
DEFAULT_OAUTH_BETA = "oauth-2025-04-20"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_CC_VERSION = "2.1.85.000"
DEFAULT_CC_ENTRYPOINT = "cli"
_IGNORED_REQUEST_KWARGS = {
    "formatted",
}


def _b64_no_pad(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _pkce_pair() -> tuple[str, str]:
    verifier = _b64_no_pad(secrets.token_bytes(64))
    challenge = _b64_no_pad(hashlib.sha256(verifier.encode("utf-8")).digest())
    return verifier, challenge


def _is_headless_environment() -> bool:
    """Detect SSH, WSL, or missing display where browser popups won't work."""
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return True
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return True
    return False


def _normalize_manual_code(raw: str, expected_state: str) -> str:
    value = raw.strip()
    if not value:
        return value

    first_token = value.split()[0]

    if "error=" in first_token or "code=" in first_token:
        parsed = urlparse(first_token)
        params = parse_qs(parsed.query)
        error = params.get("error", [None])[0]
        if error:
            desc = params.get("error_description", [error])[0]
            raise RuntimeError(f"OAuth error: {desc}")
        code = params.get("code", [None])[0]
        state_from_url = params.get("state", [None])[0]
        if state_from_url and state_from_url != expected_state:
            raise RuntimeError("OAuth manual code state mismatch.")
        if isinstance(code, str) and code:
            return code

    if "#" in first_token:
        code_part, fragment = first_token.split("#", 1)
        if fragment and fragment != expected_state:
            raise RuntimeError("OAuth manual code state mismatch.")
        return code_part

    return first_token


class AnthropicOAuthLLM(CustomLLM):
    """Anthropic OAuth-backed LLM using Claude subscription OAuth tokens."""

    model: str = Field(default=DEFAULT_MODEL, description="Anthropic model id.")
    max_tokens: Optional[int] = Field(default=None, gt=0)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    timeout: float = Field(default=30.0, gt=0)

    access_token: Optional[str] = Field(default=None, description="OAuth access token.")
    refresh_token: Optional[str] = Field(default=None, description="OAuth refresh token.")
    client_id: str = Field(default=DEFAULT_CLIENT_ID)
    authorize_url: str = Field(default=DEFAULT_AUTHORIZE_URL)
    token_url: str = Field(default=DEFAULT_TOKEN_URL)
    login_scope: str = Field(default=DEFAULT_LOGIN_SCOPE)
    refresh_scope: str = Field(default=DEFAULT_REFRESH_SCOPE)
    refresh_buffer_seconds: int = Field(default=300, ge=0)
    credential_path: Optional[str] = Field(default=DEFAULT_CREDENTIAL_PATH)

    api_base: str = Field(default=DEFAULT_API_BASE)
    anthropic_version: str = Field(default=DEFAULT_ANTHROPIC_VERSION)
    oauth_beta: str = Field(default=DEFAULT_OAUTH_BETA)
    user_agent: Optional[str] = Field(default="claude-cli/2.1.85")

    billing_header_mode: Literal["auto", "always", "never"] = Field(default="auto")
    cc_version: str = Field(default=DEFAULT_CC_VERSION)
    cc_entrypoint: str = Field(default=DEFAULT_CC_ENTRYPOINT)
    cch_value: str = Field(default="00000")
    inject_identifier: bool = Field(default=False)
    identifier_text: str = Field(
        default="You are Claude Code, Anthropic's official CLI for Claude."
    )

    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)

    _session: requests.Session = PrivateAttr()
    _cached_access_token: Optional[str] = PrivateAttr(default=None)
    _cached_refresh_token: Optional[str] = PrivateAttr(default=None)
    _access_token_expiry: Optional[float] = PrivateAttr(default=None)

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: Optional[int] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = 30.0,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        client_id: str = DEFAULT_CLIENT_ID,
        authorize_url: str = DEFAULT_AUTHORIZE_URL,
        token_url: str = DEFAULT_TOKEN_URL,
        login_scope: str = DEFAULT_LOGIN_SCOPE,
        refresh_scope: str = DEFAULT_REFRESH_SCOPE,
        refresh_buffer_seconds: int = 300,
        credential_path: Optional[str] = DEFAULT_CREDENTIAL_PATH,
        api_base: str = DEFAULT_API_BASE,
        anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
        oauth_beta: str = DEFAULT_OAUTH_BETA,
        user_agent: Optional[str] = "claude-cli/2.1.85",
        billing_header_mode: Literal["auto", "always", "never"] = "auto",
        cc_version: str = DEFAULT_CC_VERSION,
        cc_entrypoint: str = DEFAULT_CC_ENTRYPOINT,
        cch_value: str = "00000",
        inject_identifier: bool = False,
        identifier_text: str = "You are Claude Code, Anthropic's official CLI for Claude.",
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            access_token=access_token,
            refresh_token=refresh_token,
            client_id=client_id,
            authorize_url=authorize_url,
            token_url=token_url,
            login_scope=login_scope,
            refresh_scope=refresh_scope,
            refresh_buffer_seconds=refresh_buffer_seconds,
            credential_path=credential_path,
            api_base=api_base,
            anthropic_version=anthropic_version,
            oauth_beta=oauth_beta,
            user_agent=user_agent,
            billing_header_mode=billing_header_mode,
            cc_version=cc_version,
            cc_entrypoint=cc_entrypoint,
            cch_value=cch_value,
            inject_identifier=inject_identifier,
            identifier_text=identifier_text,
            additional_kwargs=additional_kwargs or {},
            callback_manager=callback_manager or CallbackManager([]),
        )
        self._session = requests.Session()
        self._cached_access_token = access_token
        self._cached_refresh_token = refresh_token
        if credential_path:
            self._load_credentials_from_file(credential_path)

    @classmethod
    def class_name(cls) -> str:
        return "AnthropicOAuthLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=200_000,
            num_output=self.max_tokens or -1,
            model_name=self.model,
            is_chat_model=True,
            is_function_calling_model=True,
        )

    def _load_credentials_from_file(self, credential_path: str) -> None:
        path = Path(credential_path).expanduser()
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return

        nested = payload.get("claudeAiOauth")
        if isinstance(nested, dict):
            file_access = nested.get("accessToken")
            file_refresh = nested.get("refreshToken")
            expires_at = nested.get("expiresAt")
        else:
            file_access = payload.get("access_token")
            file_refresh = payload.get("refresh_token")
            expires_at = payload.get("expires_at")

        if not self._cached_access_token and isinstance(file_access, str):
            self._cached_access_token = file_access
        if not self._cached_refresh_token and isinstance(file_refresh, str):
            self._cached_refresh_token = file_refresh
        if isinstance(expires_at, (int, float)):
            self._access_token_expiry = float(expires_at) / 1000.0

    def _persist_credentials(self) -> None:
        if not self.credential_path:
            return
        path = Path(self.credential_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        existing: Dict[str, Any] = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except Exception:
                existing = {}

        existing["claudeAiOauth"] = {
            "accessToken": self._cached_access_token,
            "refreshToken": self._cached_refresh_token,
            "expiresAt": int(self._access_token_expiry * 1000)
            if self._access_token_expiry
            else None,
            "scopes": self.refresh_scope.split(),
        }

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def _token_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "anthropic-beta": self.oauth_beta,
        }
        if self.user_agent:
            headers["User-Agent"] = self.user_agent
        return headers

    def _access_token_is_stale(self) -> bool:
        if not self._access_token_expiry:
            return False
        return time.time() >= (self._access_token_expiry - self.refresh_buffer_seconds)

    def _refresh_access_token(self) -> str:
        refresh_token = self._cached_refresh_token or self.refresh_token
        if not refresh_token:
            raise ValueError(
                "No refresh token available. Provide `refresh_token` or credentials file."
            )

        res = self._session.post(
            self.token_url,
            headers=self._token_headers(),
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.client_id,
                "scope": self.refresh_scope,
            },
            timeout=self.timeout,
        )
        res.raise_for_status()
        data = res.json()

        access_token = data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError(f"Token refresh succeeded but no access_token returned: {data}")

        expires_in = data.get("expires_in", 28_800)
        try:
            expires_in_s = int(expires_in)
        except (TypeError, ValueError):
            expires_in_s = 28_800

        self._cached_access_token = access_token
        self._cached_refresh_token = data.get("refresh_token") or refresh_token
        self._access_token_expiry = time.time() + expires_in_s
        self._persist_credentials()
        return access_token

    def _exchange_authorization_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        code_verifier: str,
        state: str,
        expires_in: Optional[int] = None,
    ) -> str:
        request_body: Dict[str, Any] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self.client_id,
            "code_verifier": code_verifier,
            "state": state,
        }
        if expires_in is not None:
            request_body["expires_in"] = expires_in

        res = self._session.post(
            self.token_url,
            headers=self._token_headers(),
            json=request_body,
            timeout=self.timeout,
        )
        res.raise_for_status()
        data = res.json()

        access_token = data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError(
                f"OAuth code exchange succeeded but no access_token returned: {data}"
            )

        refresh_token = data.get("refresh_token")
        if isinstance(refresh_token, str) and refresh_token:
            self._cached_refresh_token = refresh_token

        expires_in = data.get("expires_in", 28_800)
        try:
            expires_in_s = int(expires_in)
        except (TypeError, ValueError):
            expires_in_s = 28_800

        self._cached_access_token = access_token
        self._access_token_expiry = time.time() + expires_in_s
        self._persist_credentials()
        return access_token

    def _build_auth_url(
        self,
        *,
        redirect_uri: str,
        code_challenge: str,
        state: str,
    ) -> str:
        query = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": self.login_scope,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
        if "/cai/oauth/authorize" in self.authorize_url:
            query["code"] = "true"
        return f"{self.authorize_url}?{urlencode(query)}"

    def login(
        self,
        *,
        open_browser: bool = True,
        timeout_seconds: float = 300.0,
        callback_host: str = "127.0.0.1",
        callback_port: int = 0,
        callback_path: str = "/callback",
        expires_in: Optional[int] = None,
    ) -> str:
        # Headless environments: skip local server, use hosted callback page
        use_headless = _is_headless_environment() or os.environ.get(
            "DROIDRUN_OAUTH_MANUAL", ""
        ).lower() in ("1", "true", "yes")
        if use_headless:
            return self.login_headless(
                open_browser=open_browser,
                timeout_seconds=timeout_seconds,
                expires_in=expires_in,
            )

        # Desktop: browser callback server
        result: Dict[str, Optional[str]] = {"code": None, "state": None, "error": None}
        done = threading.Event()

        code_verifier, code_challenge = _pkce_pair()
        state = _b64_no_pad(secrets.token_bytes(32))
        original_authorize_url = self.authorize_url

        if "/cai/oauth/authorize" not in self.authorize_url:
            self.authorize_url = DEFAULT_MODERN_AUTHORIZE_URL

        class _OAuthHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path != callback_path:
                    self.send_response(404)
                    self.end_headers()
                    return

                params = parse_qs(parsed.query)
                result["code"] = params.get("code", [None])[0]
                result["state"] = params.get("state", [None])[0]
                result["error"] = params.get("error", [None])[0]

                ok = result["code"] is not None and result["error"] is None
                self.send_response(200 if ok else 400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                if ok:
                    self.wfile.write(
                        b"<html><body><h3>Login complete. You can close this tab.</h3></body></html>"
                    )
                else:
                    self.wfile.write(
                        b"<html><body><h3>Login failed. Return to your terminal.</h3></body></html>"
                    )
                done.set()

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

        try:
            httpd = HTTPServer((callback_host, callback_port), _OAuthHandler)
        except OSError as exc:
            self.authorize_url = original_authorize_url
            print(
                f"Could not bind callback server on {callback_host}:{callback_port} ({exc}). "
                "Falling back to manual code entry."
            )
            return self.login_headless(
                open_browser=open_browser,
                timeout_seconds=timeout_seconds,
                expires_in=expires_in,
            )

        actual_port = httpd.server_address[1]
        redirect_uri = f"http://localhost:{actual_port}{callback_path}"
        auth_url = self._build_auth_url(
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            state=state,
        )

        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()

        try:
            print(f"Open this URL to login:\n{auth_url}\n")
            if open_browser:
                webbrowser.open(auth_url)

            if not done.wait(timeout=timeout_seconds):
                raise TimeoutError("OAuth login timed out before callback was received.")

            if result["error"]:
                raise RuntimeError(f"OAuth callback returned error: {result['error']}")
            if result["state"] != state:
                raise RuntimeError("OAuth callback state mismatch.")
            if not result["code"]:
                raise RuntimeError("OAuth callback did not include an authorization code.")

            return self._exchange_authorization_code(
                code=result["code"],
                redirect_uri=redirect_uri,
                code_verifier=code_verifier,
                state=state,
                expires_in=expires_in,
            )
        finally:
            self.authorize_url = original_authorize_url
            httpd.shutdown()
            httpd.server_close()

    def login_headless(
        self,
        *,
        open_browser: bool = False,
        timeout_seconds: float = 300.0,
        input_fn: Any = input,
        expires_in: Optional[int] = None,
    ) -> str:
        """Headless OAuth flow for SSH/WSL environments.

        Redirects to Anthropic's hosted callback page which displays the
        authorization code on screen.
        """
        code_verifier, code_challenge = _pkce_pair()
        state = _b64_no_pad(secrets.token_bytes(32))
        redirect_uri = "https://platform.claude.com/oauth/code/callback"
        original_authorize_url = self.authorize_url

        if "/cai/oauth/authorize" not in self.authorize_url:
            self.authorize_url = DEFAULT_MODERN_AUTHORIZE_URL

        auth_url = self._build_auth_url(
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            state=state,
        )

        try:
            print(
                f"\nSign in with your Anthropic account:\n"
                f"\n1. Open this link in your browser:\n   {auth_url}\n"
                f"\n2. Complete sign-in, then paste the authorization code shown on the page.\n"
            )
            if open_browser:
                webbrowser.open(auth_url)

            import queue as _queue

            deadline = time.time() + timeout_seconds
            input_queue: _queue.Queue[Optional[str]] = _queue.Queue()

            def _reader() -> None:
                while True:
                    try:
                        input_queue.put(str(input_fn("Enter the authorization code: ")))
                    except (EOFError, OSError):
                        input_queue.put(None)
                        return

            threading.Thread(target=_reader, daemon=True).start()

            for attempt in range(2):
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("OAuth login timed out.")

                try:
                    raw = input_queue.get(timeout=remaining)
                except _queue.Empty:
                    raise TimeoutError("OAuth login timed out.")

                if raw is None:
                    raise RuntimeError("Login failed — stdin closed.")
                if not raw.strip():
                    if attempt == 0:
                        print("No code entered. Try again.")
                        continue
                    raise RuntimeError("Login failed.")
                try:
                    code = _normalize_manual_code(raw, state)
                except Exception:  # noqa: BLE001
                    if attempt == 0:
                        print("Invalid code. Try again.")
                        continue
                    raise RuntimeError("Login failed.")
                if code:
                    return self._exchange_authorization_code(
                        code=code,
                        redirect_uri=redirect_uri,
                        code_verifier=code_verifier,
                        state=state,
                        expires_in=expires_in,
                    )
                if attempt == 0:
                    print("Invalid code. Try again.")
                    continue
                raise RuntimeError("Login failed.")
            raise RuntimeError("Login failed.")
        finally:
            self.authorize_url = original_authorize_url

    def _resolve_access_token(self) -> str:
        env_access_token = os.environ.get("ANTHROPIC_OAUTH_TOKEN")
        if env_access_token:
            return env_access_token

        if self._cached_access_token and not self._access_token_is_stale():
            return self._cached_access_token

        if self._cached_access_token and not self._cached_refresh_token:
            return self._cached_access_token

        if self._cached_refresh_token or self.refresh_token:
            return self._refresh_access_token()

        raise ValueError(
            "No OAuth token available. Provide `access_token`, `refresh_token`, "
            "or a valid credential_path."
        )

    @staticmethod
    def _extract_text(payload: Dict[str, Any]) -> str:
        blocks = payload.get("content")
        if not isinstance(blocks, list):
            return ""
        texts: list[str] = []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                texts.append(str(block["text"]))
        return "\n".join(texts)

    def _to_provider_messages(
        self, messages: Sequence[ChatMessage]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        provider_messages: list[dict[str, Any]] = []
        system_lines: list[str] = []

        for message in messages:
            role = message.role.value
            content = message.content if isinstance(message.content, str) else ""
            if role == MessageRole.SYSTEM.value:
                if content:
                    system_lines.append(content)
                continue
            if role not in {MessageRole.USER.value, MessageRole.ASSISTANT.value}:
                role = MessageRole.USER.value
            provider_messages.append({"role": role, "content": content})

        if not provider_messages:
            provider_messages.append({"role": MessageRole.USER.value, "content": ""})
        return provider_messages, system_lines

    def _billing_header_text(self) -> str:
        return (
            "x-anthropic-billing-header: "
            f"cc_version={self.cc_version}; "
            f"cc_entrypoint={self.cc_entrypoint}; "
            f"cch={self.cch_value};"
        )

    def _use_billing_header(self, model_id: str) -> bool:
        if self.billing_header_mode == "always":
            return True
        if self.billing_header_mode == "never":
            return False
        return not model_id.startswith("claude-haiku")

    def _system_blocks(self, user_system_lines: Sequence[str], model_id: str) -> list[dict[str, str]]:
        blocks: list[dict[str, str]] = []
        if self._use_billing_header(model_id):
            blocks.append({"type": "text", "text": self._billing_header_text()})
        if self.inject_identifier and self.identifier_text:
            blocks.append({"type": "text", "text": self.identifier_text})
        for line in user_system_lines:
            if line:
                blocks.append({"type": "text", "text": line})
        return blocks

    @staticmethod
    def _sanitize_request_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Drop framework-only kwargs that are invalid for Anthropic's API."""
        return {
            key: value
            for key, value in kwargs.items()
            if key not in _IGNORED_REQUEST_KWARGS and value is not None
        }

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        provider_messages, system_lines = self._to_provider_messages(messages)
        token = self._resolve_access_token()
        request_kwargs = self._sanitize_request_kwargs(kwargs)

        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": provider_messages,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        system_blocks = self._system_blocks(system_lines, self.model)
        if system_blocks:
            payload["system"] = system_blocks

        payload.update(self.additional_kwargs)
        payload.update(request_kwargs)

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "anthropic-version": self.anthropic_version,
            "anthropic-beta": self.oauth_beta,
        }
        if self.user_agent:
            headers["User-Agent"] = self.user_agent

        res = self._session.post(
            f"{self.api_base.rstrip('/')}/v1/messages",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        res.raise_for_status()
        data = res.json()
        text = self._extract_text(data)

        assistant = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=text,
            additional_kwargs={"content_blocks": data.get("content", [])},
        )
        return ChatResponse(
            message=assistant,
            raw=data,
            additional_kwargs={
                "id": data.get("id"),
                "usage": data.get("usage"),
                "stop_reason": data.get("stop_reason"),
            },
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        del formatted
        resp = self.chat(
            [ChatMessage(role=MessageRole.USER, content=prompt)],
            **self._sanitize_request_kwargs(kwargs),
        )
        return CompletionResponse(
            text=resp.message.content or "",
            raw=resp.raw,
            additional_kwargs=resp.additional_kwargs,
        )

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        raise NotImplementedError("Streaming is not implemented for AnthropicOAuthLLM yet.")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("Streaming is not implemented for AnthropicOAuthLLM yet.")
