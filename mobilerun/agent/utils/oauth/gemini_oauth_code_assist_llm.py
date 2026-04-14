import base64
import hashlib
import json
import os
import secrets
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, ClassVar
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
from droidrun.config_manager.credential_paths import GEMINI_OAUTH_CREDENTIAL_PATH

DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com"
DEFAULT_CODE_ASSIST_API_VERSION = "v1internal"
DEFAULT_CODE_ASSIST_LOAD_METHOD = "loadCodeAssist"
DEFAULT_CODE_ASSIST_ONBOARD_METHOD = "onboardUser"
DEFAULT_TOKEN_URL = "https://oauth2.googleapis.com/token"
DEFAULT_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
DEFAULT_CREDENTIAL_PATH = str(GEMINI_OAUTH_CREDENTIAL_PATH)

# Same installed-app OAuth client used by gemini-cli.
DEFAULT_CLIENT_ID = (
    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
)
DEFAULT_CLIENT_SECRET = "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"

# LlamaIndex-internal kwargs that must never be forwarded to Google's API.
_IGNORED_REQUEST_KWARGS = {"formatted"}


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
    """Parse pasted input: full URL with code= param, code#state, or bare code."""
    value = raw.strip()
    if not value:
        return value

    first_token = value.split()[0]

    if "code=" in first_token:
        parsed = urlparse(first_token)
        params = parse_qs(parsed.query)
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


class GeminiOAuthCodeAssistLLM(CustomLLM):
    """Gemini OAuth LLM that talks to Google Code Assist endpoints.

    This class expects at least one of:
    - `access_token`
    - `refresh_token`
    - `credential_path` file containing cached OAuth credentials.

    Cached credentials are stored in droidrun's config dir.
    """

    MODEL_PRESETS: ClassVar[Dict[str, str]] = {
        "pro_preview": "gemini-3.1-pro-preview",
        "flash": "gemini-2.5-flash",
        "flash_lite": "gemini-2.5-flash-lite",
    }

    model: str = Field(default=DEFAULT_MODEL, description="Gemini model id.")
    model_preset: str = Field(
        default="pro_preview",
        description="Quick model selector key from MODEL_PRESETS.",
    )
    custom_model: Optional[str] = Field(
        default=None,
        description="Optional custom model id; overrides model/model_preset.",
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Code Assist project id returned by loadCodeAssist onboarding.",
    )
    max_tokens: Optional[int] = Field(default=None, gt=0)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    timeout: float = Field(default=30.0, gt=0)

    access_token: Optional[str] = Field(default=None, description="OAuth access token.")
    refresh_token: Optional[str] = Field(default=None, description="OAuth refresh token.")
    client_id: str = Field(default=DEFAULT_CLIENT_ID)
    client_secret: str = Field(default=DEFAULT_CLIENT_SECRET)
    authorize_url: str = Field(default=DEFAULT_AUTHORIZE_URL)
    token_url: str = Field(default=DEFAULT_TOKEN_URL)
    refresh_buffer_seconds: int = Field(default=300, ge=0)

    code_assist_endpoint: str = Field(default=DEFAULT_CODE_ASSIST_ENDPOINT)
    code_assist_api_version: str = Field(default=DEFAULT_CODE_ASSIST_API_VERSION)

    credential_path: Optional[str] = Field(
        default=DEFAULT_CREDENTIAL_PATH,
        description="Optional path to JSON OAuth credentials cache.",
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
        model_preset: str = "pro_preview",
        custom_model: Optional[str] = None,
        project_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = 30.0,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        client_id: str = DEFAULT_CLIENT_ID,
        client_secret: str = DEFAULT_CLIENT_SECRET,
        authorize_url: str = DEFAULT_AUTHORIZE_URL,
        token_url: str = DEFAULT_TOKEN_URL,
        refresh_buffer_seconds: int = 300,
        code_assist_endpoint: str = DEFAULT_CODE_ASSIST_ENDPOINT,
        code_assist_api_version: str = DEFAULT_CODE_ASSIST_API_VERSION,
        credential_path: Optional[str] = DEFAULT_CREDENTIAL_PATH,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        selected_model = custom_model
        if not selected_model:
            if model in self.MODEL_PRESETS:
                # Passed a preset key like "pro_preview" → resolve to actual id.
                selected_model = self.MODEL_PRESETS[model]
            elif model and model != DEFAULT_MODEL:
                # Explicit model name from config/CLI → honor it verbatim.
                selected_model = model
            elif model_preset in self.MODEL_PRESETS:
                # Fall back to preset only when no explicit model was provided.
                selected_model = self.MODEL_PRESETS[model_preset]
            else:
                selected_model = model

        super().__init__(
            model=selected_model,
            model_preset=model_preset,
            custom_model=custom_model,
            project_id=project_id,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            access_token=access_token,
            refresh_token=refresh_token,
            client_id=client_id,
            client_secret=client_secret,
            authorize_url=authorize_url,
            token_url=token_url,
            refresh_buffer_seconds=refresh_buffer_seconds,
            code_assist_endpoint=code_assist_endpoint,
            code_assist_api_version=code_assist_api_version,
            credential_path=credential_path,
            additional_kwargs=additional_kwargs or {},
            callback_manager=callback_manager or CallbackManager([]),
        )

        self._session = requests.Session()
        self._cached_access_token = access_token
        self._cached_refresh_token = refresh_token

        if credential_path:
            self._load_credentials_from_file(credential_path)

    @classmethod
    def available_model_presets(cls) -> Dict[str, str]:
        return dict(cls.MODEL_PRESETS)

    @classmethod
    def class_name(cls) -> str:
        return "gemini_oauth_code_assist"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=1_000_000,
            num_output=self.max_tokens or -1,
            model_name=self.model,
            is_chat_model=True,
            is_function_calling_model=False,
        )

    _NESTED_KEY = "geminiOauth"

    def _load_credentials_from_file(self, credential_path: str) -> None:
        path = Path(credential_path).expanduser()
        if not path.exists():
            return

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return

        nested = raw.get(self._NESTED_KEY)
        payload = nested if isinstance(nested, dict) else raw

        file_access = payload.get("access_token")
        file_refresh = payload.get("refresh_token")
        expiry_ms = payload.get("expiry_date")
        project_id = payload.get("project_id")

        if not self._cached_access_token and isinstance(file_access, str):
            self._cached_access_token = file_access
        if not self._cached_refresh_token and isinstance(file_refresh, str):
            self._cached_refresh_token = file_refresh
        if isinstance(expiry_ms, (int, float)):
            self._access_token_expiry = float(expiry_ms) / 1000.0
        if not self.project_id and isinstance(project_id, str) and project_id:
            object.__setattr__(self, "project_id", project_id)

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

        existing[self._NESTED_KEY] = {
            "access_token": self._cached_access_token,
            "refresh_token": self._cached_refresh_token,
            "token_type": "Bearer",
            "expiry_date": int(self._access_token_expiry * 1000)
            if self._access_token_expiry
            else None,
            "project_id": self.project_id,
        }

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    @staticmethod
    def _extract_project_id(payload: Dict[str, Any]) -> Optional[str]:
        project = payload.get("cloudaicompanionProject")
        if isinstance(project, str) and project:
            return project
        if isinstance(project, dict):
            project_id = project.get("id")
            if isinstance(project_id, str) and project_id:
                return project_id
        return None

    @staticmethod
    def _default_tier_id(allowed_tiers: Any) -> str:
        if isinstance(allowed_tiers, list):
            for tier in allowed_tiers:
                if isinstance(tier, dict) and tier.get("isDefault") and isinstance(
                    tier.get("id"), str
                ):
                    return tier["id"]
            for tier in allowed_tiers:
                if isinstance(tier, dict) and isinstance(tier.get("id"), str):
                    return tier["id"]
        return "free-tier"

    def _metadata_payload(self) -> Dict[str, str]:
        return {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }

    def _build_headers(self, token: str) -> Dict[str, str]:
        """Build Code Assist request headers matching gemini-cli expectations.

        The private v1internal endpoint requires the X-Goog-Api-Client and
        Client-Metadata headers to identify the caller as a gemini-cli-style
        client; without them, requests return 400.
        """
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "google-cloud-sdk vscode_cloudshelleditor/0.1",
            "X-Goog-Api-Client": "gl-node/22.17.0",
            "Client-Metadata": json.dumps(self._metadata_payload()),
        }

    def _ensure_project_id(self, token: str) -> Optional[str]:
        if self.project_id:
            return self.project_id

        headers = self._build_headers(token)
        metadata = self._metadata_payload()
        response = self._session.post(
            self._method_url(DEFAULT_CODE_ASSIST_LOAD_METHOD),
            headers=headers,
            json={"metadata": metadata},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        project_id = self._extract_project_id(data)
        if project_id:
            object.__setattr__(self, "project_id", project_id)
            self._persist_credentials()
            return project_id

        if data.get("currentTier"):
            return None

        tier_id = self._default_tier_id(data.get("allowedTiers"))
        onboard_response = self._session.post(
            self._method_url(DEFAULT_CODE_ASSIST_ONBOARD_METHOD),
            headers=headers,
            json={"tierId": tier_id, "metadata": metadata},
            timeout=self.timeout,
        )
        onboard_response.raise_for_status()
        onboard_data = onboard_response.json()

        project_id = self._extract_project_id(onboard_data.get("response") or onboard_data)
        if project_id:
            object.__setattr__(self, "project_id", project_id)
            self._persist_credentials()
            return project_id

        return None

    def _access_token_is_stale(self) -> bool:
        if not self._access_token_expiry:
            return False
        return time.time() >= (self._access_token_expiry - self.refresh_buffer_seconds)

    def _refresh_access_token(self) -> str:
        refresh_token = self._cached_refresh_token or self.refresh_token
        if not refresh_token:
            raise ValueError(
                "No refresh token available. Provide `refresh_token` or a credential file."
            )

        response = self._session.post(
            self.token_url,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        access_token = data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError(f"Token refresh succeeded but no access_token returned: {data}")

        expires_in = data.get("expires_in", 3600)
        try:
            expires_in_s = int(expires_in)
        except (TypeError, ValueError):
            expires_in_s = 3600

        self._cached_access_token = access_token
        self._cached_refresh_token = data.get("refresh_token") or refresh_token
        self._access_token_expiry = time.time() + expires_in_s
        self._persist_credentials()

        return access_token

    def _exchange_authorization_code(
        self,
        code: str,
        redirect_uri: str,
        code_verifier: Optional[str] = None,
    ) -> str:
        payload = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        if code_verifier:
            payload["code_verifier"] = code_verifier

        response = self._session.post(
            self.token_url,
            data=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        access_token = data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError(
                f"OAuth code exchange succeeded but no access_token returned: {data}"
            )

        refresh_token = data.get("refresh_token")
        if isinstance(refresh_token, str) and refresh_token:
            self._cached_refresh_token = refresh_token

        if not self._cached_refresh_token:
            raise RuntimeError(
                "No refresh token received from Google. "
                "Revoke access at https://myaccount.google.com/permissions and retry."
            )

        expires_in = data.get("expires_in", 3600)
        try:
            expires_in_s = int(expires_in)
        except (TypeError, ValueError):
            expires_in_s = 3600

        self._cached_access_token = access_token
        self._access_token_expiry = time.time() + expires_in_s
        self._persist_credentials()
        return access_token

    def _build_auth_url(
        self,
        redirect_uri: str,
        state: str,
        prompt_consent: bool,
        code_challenge: Optional[str] = None,
    ) -> str:
        scope = " ".join(
            [
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/userinfo.email",
                "https://www.googleapis.com/auth/userinfo.profile",
            ]
        )
        query = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": scope,
            "access_type": "offline",
            "state": state,
        }
        if prompt_consent:
            query["prompt"] = "consent"
        if code_challenge:
            query["code_challenge"] = code_challenge
            query["code_challenge_method"] = "S256"
        return f"{self.authorize_url}?{urlencode(query)}"

    def login(
        self,
        *,
        open_browser: bool = True,
        timeout_seconds: float = 300.0,
        callback_host: str = "127.0.0.1",
        callback_port: int = 0,
        callback_path: str = "/oauth2callback",
        prompt_consent: bool = True,
    ) -> str:
        result: Dict[str, Optional[str]] = {"code": None, "state": None, "error": None}
        manual_code: Dict[str, Optional[str]] = {"code": None}
        done = threading.Event()
        expected_state = secrets.token_hex(32)
        code_verifier, code_challenge = _pkce_pair()

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
            print(
                f"Could not bind callback server on {callback_host}:{callback_port} ({exc}). "
                "Falling back to manual code entry."
            )
            return self.login_manual(
                open_browser=open_browser, prompt_consent=prompt_consent
            )

        actual_port = httpd.server_address[1]
        redirect_uri = f"http://127.0.0.1:{actual_port}{callback_path}"
        auth_url = self._build_auth_url(
            redirect_uri=redirect_uri,
            state=expected_state,
            prompt_consent=prompt_consent,
            code_challenge=code_challenge,
        )

        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()

        try:
            print(f"Open this URL to login:\n{auth_url}\n")
            if open_browser:
                webbrowser.open(auth_url)

            # Only run the manual-paste race when we can't rely on the local
            # browser callback: headless envs (SSH/WSL/no-display), or when the
            # user explicitly opts in with DROIDRUN_OAUTH_MANUAL=1. On a normal
            # desktop the server always wins anyway, and a blocked input()
            # thread would intercept InquirerPy's terminal queries and lag the
            # configure wizard.
            enable_manual = _is_headless_environment() or bool(
                os.environ.get("DROIDRUN_OAUTH_MANUAL")
            )
            if enable_manual:
                def _read_manual() -> None:
                    try:
                        raw = str(input("Or paste the redirect URL / authorization code: "))
                    except Exception:
                        return
                    if done.is_set() or not raw.strip():
                        return
                    try:
                        code = _normalize_manual_code(raw, expected_state)
                    except Exception as e:  # noqa: BLE001
                        print(f"Invalid paste: {e}")
                        return
                    if code:
                        manual_code["code"] = code
                        done.set()

                manual_thread = threading.Thread(target=_read_manual, daemon=True)
                manual_thread.start()

            if not done.wait(timeout=timeout_seconds):
                raise TimeoutError("OAuth login timed out before callback was received.")

            if manual_code["code"]:
                code_to_exchange = manual_code["code"]
            else:
                if result["error"]:
                    raise RuntimeError(f"OAuth callback returned error: {result['error']}")
                if result["state"] != expected_state:
                    raise RuntimeError("OAuth callback state mismatch.")
                if not result["code"]:
                    raise RuntimeError("OAuth callback did not include an authorization code.")
                code_to_exchange = result["code"]

            return self._exchange_authorization_code(
                code_to_exchange, redirect_uri, code_verifier=code_verifier
            )
        finally:
            httpd.shutdown()
            httpd.server_close()

    def login_manual(
        self,
        *,
        open_browser: bool = True,
        input_fn: Any = input,
        prompt_consent: bool = True,
    ) -> str:
        """Manual OAuth flow for headless/VPS/WSL environments.

        Opens (or prints) the auth URL and prompts the user to paste the
        redirected URL or bare authorization code from the browser.
        """
        code_verifier, code_challenge = _pkce_pair()
        expected_state = secrets.token_hex(32)
        # Google allows any loopback redirect for installed apps. The browser
        # will fail to load the page, but the URL bar will contain the code.
        redirect_uri = "http://localhost/oauth2callback"

        auth_url = self._build_auth_url(
            redirect_uri=redirect_uri,
            state=expected_state,
            prompt_consent=prompt_consent,
            code_challenge=code_challenge,
        )

        print(f"Open this URL to login:\n{auth_url}")
        if open_browser:
            webbrowser.open(auth_url)

        raw = str(input_fn("Paste the redirect URL or authorization code: "))
        code = _normalize_manual_code(raw, expected_state)
        if not code:
            raise RuntimeError("Authorization code was empty.")

        return self._exchange_authorization_code(
            code, redirect_uri, code_verifier=code_verifier
        )

    def _resolve_access_token(self) -> str:
        env_access_token = os.environ.get("GEMINI_OAUTH_ACCESS_TOKEN")
        if env_access_token:
            return env_access_token

        if self._cached_access_token and not self._access_token_is_stale():
            return self._cached_access_token

        if self._cached_access_token and not self._cached_refresh_token:
            # If we only have an access token and no expiry/refresh, still try to use it.
            return self._cached_access_token

        if self._cached_refresh_token or self.refresh_token:
            return self._refresh_access_token()

        raise ValueError(
            "No OAuth token available. Provide `access_token`, `refresh_token`, "
            "or a valid credential_path."
        )

    @staticmethod
    def _message_text(message: ChatMessage) -> str:
        if isinstance(message.content, str):
            return message.content
        return ""

    def _to_code_assist_request(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        converted_messages = self.convert_chat_messages(messages)

        contents: list[Dict[str, Any]] = []
        system_chunks: list[str] = []

        for msg in converted_messages:
            text = self._message_text(msg)
            if msg.role == MessageRole.SYSTEM:
                if text:
                    system_chunks.append(text)
                continue

            role = "model" if msg.role == MessageRole.ASSISTANT else "user"
            contents.append({"role": role, "parts": [{"text": text}]})

        if not contents:
            contents.append({"role": "user", "parts": [{"text": ""}]})

        generation_config = {
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            generation_config["maxOutputTokens"] = self.max_tokens
        generation_config.update(kwargs.pop("generation_config", {}))

        request: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        if system_chunks:
            request["systemInstruction"] = {
                "role": "user",
                "parts": [{"text": "\n\n".join(system_chunks)}],
            }

        request_extra = dict(self.additional_kwargs)
        request_extra.update(kwargs.pop("request_extra", {}))
        request.update(request_extra)

        payload: Dict[str, Any] = {
            "model": self.model,
            "request": request,
            "userAgent": "droidrun",
            "requestId": f"droidrun-{int(time.time() * 1000)}-{secrets.token_hex(4)}",
        }
        if self.project_id:
            payload["project"] = self.project_id

        # Strip LlamaIndex-internal kwargs (e.g. ``formatted``) that Google's
        # Code Assist API rejects as unknown fields.
        safe_kwargs = {
            k: v for k, v in kwargs.items() if k not in _IGNORED_REQUEST_KWARGS
        }
        payload.update(safe_kwargs)
        return payload

    def _method_url(self, method: str) -> str:
        base = self.code_assist_endpoint.rstrip("/")
        version = self.code_assist_api_version.strip("/")
        return f"{base}/{version}:{method}"

    @staticmethod
    def _extract_text(chunk: Dict[str, Any]) -> str:
        response = chunk.get("response") or {}
        candidates = response.get("candidates") or []
        if not candidates:
            return ""

        first = candidates[0] or {}
        content = first.get("content") or {}
        parts = content.get("parts") or []

        text_parts: list[str] = []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])

        return "".join(text_parts)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Google's Code Assist private API (v1internal) does not reliably
        # accept the non-streaming generateContent endpoint for every model.
        # Route chat through streamGenerateContent and accumulate the stream,
        # matching the pattern used by gemini-cli / OpenClaw.
        token = self._resolve_access_token()
        self._ensure_project_id(token)
        payload = self._to_code_assist_request(messages, **kwargs)

        response = self._session.post(
            self._method_url("streamGenerateContent"),
            params={"alt": "sse"},
            headers={
                **self._build_headers(token),
                "Accept": "text/event-stream",
            },
            json=payload,
            timeout=self.timeout,
            stream=True,
        )
        if not response.ok:
            raise requests.HTTPError(
                f"Code Assist {response.status_code} error: {response.text}",
                response=response,
            )

        accumulated = ""
        last_raw: Dict[str, Any] = {}
        buffer: list[str] = []

        def _flush(buffer: list[str]) -> Optional[Dict[str, Any]]:
            if not buffer:
                return None
            chunk_text = "\n".join(buffer)
            try:
                return json.loads(chunk_text)
            except json.JSONDecodeError:
                return None

        for line in response.iter_lines(decode_unicode=True):
            if line is None:
                continue
            stripped = line.strip()
            if stripped.startswith("data:"):
                buffer.append(stripped[5:].strip())
                continue
            if stripped != "" or not buffer:
                continue
            raw_chunk = _flush(buffer)
            buffer = []
            if raw_chunk is None:
                continue
            last_raw = raw_chunk
            delta = self._extract_text(raw_chunk)
            if delta:
                accumulated += delta

        raw_chunk = _flush(buffer)
        if raw_chunk is not None:
            last_raw = raw_chunk
            delta = self._extract_text(raw_chunk)
            if delta:
                accumulated += delta

        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=accumulated),
            raw=last_raw,
            additional_kwargs={
                "trace_id": last_raw.get("traceId"),
                "usage": (last_raw.get("response") or {}).get("usageMetadata"),
                "model_version": (last_raw.get("response") or {}).get("modelVersion"),
            },
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        chat_response = self.chat(
            [ChatMessage(role=MessageRole.USER, content=prompt)],
            **kwargs,
        )
        return CompletionResponse(
            text=chat_response.message.content or "",
            raw=chat_response.raw,
            additional_kwargs=chat_response.additional_kwargs,
        )

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        token = self._resolve_access_token()
        self._ensure_project_id(token)
        payload = self._to_code_assist_request(messages, **kwargs)

        response = self._session.post(
            self._method_url("streamGenerateContent"),
            params={"alt": "sse"},
            headers={
                **self._build_headers(token),
                "Accept": "text/event-stream",
            },
            json=payload,
            timeout=self.timeout,
            stream=True,
        )
        response.raise_for_status()

        def gen() -> ChatResponseGen:
            accumulated = ""
            buffer: list[str] = []

            for line in response.iter_lines(decode_unicode=True):
                if line is None:
                    continue

                stripped = line.strip()
                if stripped.startswith("data:"):
                    buffer.append(stripped[5:].strip())
                    continue

                if stripped != "" or not buffer:
                    continue

                chunk_text = "\n".join(buffer)
                buffer = []
                try:
                    raw_chunk = json.loads(chunk_text)
                except json.JSONDecodeError:
                    continue

                delta = self._extract_text(raw_chunk)
                if not delta:
                    continue

                accumulated += delta
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=accumulated),
                    delta=delta,
                    raw=raw_chunk,
                    additional_kwargs={
                        "trace_id": raw_chunk.get("traceId"),
                        "usage": (raw_chunk.get("response") or {}).get("usageMetadata"),
                        "model_version": (raw_chunk.get("response") or {}).get("modelVersion"),
                    },
                )

            if buffer:
                chunk_text = "\n".join(buffer)
                try:
                    raw_chunk = json.loads(chunk_text)
                except json.JSONDecodeError:
                    raw_chunk = None

                if raw_chunk:
                    delta = self._extract_text(raw_chunk)
                    if delta:
                        accumulated += delta
                        yield ChatResponse(
                            message=ChatMessage(
                                role=MessageRole.ASSISTANT,
                                content=accumulated,
                            ),
                            delta=delta,
                            raw=raw_chunk,
                            additional_kwargs={
                                "trace_id": raw_chunk.get("traceId"),
                                "usage": (raw_chunk.get("response") or {}).get(
                                    "usageMetadata"
                                ),
                                "model_version": (raw_chunk.get("response") or {}).get(
                                    "modelVersion"
                                ),
                            },
                        )

        return gen()

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        def gen() -> CompletionResponseGen:
            for chat_chunk in self.stream_chat(
                [ChatMessage(role=MessageRole.USER, content=prompt)],
                **kwargs,
            ):
                yield CompletionResponse(
                    text=chat_chunk.message.content or "",
                    delta=chat_chunk.delta,
                    raw=chat_chunk.raw,
                    additional_kwargs=chat_chunk.additional_kwargs,
                )

        return gen()
