import base64
import hashlib
import json
import os
import queue
import secrets
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer as HTTPServer
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Sequence
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    ImageBlock,
    LLMMetadata,
    MessageRole,
    TextBlock,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM

from mobilerun.agent.utils.oauth.login_timeout import (
    OAuthLoginDeadline,
    open_browser_async,
)
from mobilerun.config_manager.auth_profile_store import AuthProfileStore
from mobilerun.config_manager.credential_paths import GEMINI_OAUTH_CREDENTIAL_PATH

DEFAULT_MODEL = "gemini-3.5-flash-low"
DEFAULT_CODE_ASSIST_ENDPOINT = "https://daily-cloudcode-pa.googleapis.com"
DEFAULT_CODE_ASSIST_API_VERSION = "v1internal"
DEFAULT_CODE_ASSIST_MODELS_METHOD = "fetchAvailableModels"
DEFAULT_TOKEN_URL = "https://oauth2.googleapis.com/token"
DEFAULT_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
DEFAULT_CREDENTIAL_PATH = str(GEMINI_OAUTH_CREDENTIAL_PATH)

# Antigravity CLI installed-app OAuth client.
DEFAULT_CLIENT_ID = (
    "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
)
DEFAULT_CLIENT_SECRET = "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"

DEFAULT_OAUTH_SCOPES = (
    "https://www.googleapis.com/auth/aicode",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
)

# Credential slot in the shared auth-profiles.json.
DEFAULT_CREDENTIAL_SLOT = "geminiAntigravityOauth"

# Kwargs that must never be forwarded to Google's API (LlamaIndex-internal, and
# project ids which the consumer entitlement does not use).
_IGNORED_REQUEST_KWARGS = {"formatted", "project", "project_id"}


def _b64_no_pad(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _pkce_pair() -> tuple[str, str]:
    verifier = _b64_no_pad(secrets.token_bytes(64))
    challenge = _b64_no_pad(hashlib.sha256(verifier.encode("utf-8")).digest())
    return verifier, challenge


# Keep in sync with _MAX_CODE_ATTEMPTS in anthropic_oauth_llm.py
_MAX_CODE_ATTEMPTS = 2


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


class GeminiOAuthCodeAssistLLM(CustomLLM):
    """Gemini OAuth LLM that talks to Google Code Assist endpoints.

    This class expects at least one of:
    - `access_token`
    - `refresh_token`
    - `credential_path` file containing cached OAuth credentials.

    Cached credentials are stored in mobilerun's config dir.
    """

    MODEL_PRESETS: ClassVar[Dict[str, str]] = {
        "flash": "gemini-3.5-flash-low",
        "pro": "gemini-pro-agent",
        "flash_lite": "gemini-3.5-flash-extra-low",
    }

    model: str = Field(default=DEFAULT_MODEL, description="Gemini model id.")
    model_preset: Optional[str] = Field(
        default=None,
        description="Quick model selector key from MODEL_PRESETS.",
    )
    custom_model: Optional[str] = Field(
        default=None,
        description="Optional custom model id; overrides model/model_preset.",
    )
    credential_slot: str = Field(
        default=DEFAULT_CREDENTIAL_SLOT,
        description="Nested key in the shared auth-profiles.json credential file.",
    )
    max_tokens: Optional[int] = Field(default=None, gt=0)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    timeout: float = Field(default=30.0, gt=0)

    access_token: Optional[str] = Field(default=None, description="OAuth access token.")
    refresh_token: Optional[str] = Field(
        default=None, description="OAuth refresh token."
    )
    client_id: str = Field(default=DEFAULT_CLIENT_ID)
    client_secret: str = Field(default=DEFAULT_CLIENT_SECRET)
    authorize_url: str = Field(default=DEFAULT_AUTHORIZE_URL)
    token_url: str = Field(default=DEFAULT_TOKEN_URL)
    refresh_buffer_seconds: int = Field(default=300, ge=0)

    code_assist_endpoint: str = Field(default=DEFAULT_CODE_ASSIST_ENDPOINT)
    code_assist_api_version: str = Field(default=DEFAULT_CODE_ASSIST_API_VERSION)
    scopes: tuple = Field(default=DEFAULT_OAUTH_SCOPES)

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
        model: Optional[str] = None,
        *,
        model_preset: Optional[str] = None,
        custom_model: Optional[str] = None,
        credential_slot: str = DEFAULT_CREDENTIAL_SLOT,
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
        scopes: Optional[Sequence[str]] = None,
        credential_path: Optional[str] = DEFAULT_CREDENTIAL_PATH,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        # custom_model wins; then an explicit model (honored verbatim, even if it
        # equals DEFAULT_MODEL); then a preset key; else the default model.
        selected_model = custom_model
        if not selected_model:
            if model in self.MODEL_PRESETS:
                selected_model = self.MODEL_PRESETS[model]
            elif model:
                selected_model = model
            elif model_preset in self.MODEL_PRESETS:
                selected_model = self.MODEL_PRESETS[model_preset]
            else:
                selected_model = DEFAULT_MODEL

        super().__init__(
            model=selected_model,
            model_preset=model_preset,
            custom_model=custom_model,
            credential_slot=credential_slot,
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
            scopes=tuple(scopes) if scopes else DEFAULT_OAUTH_SCOPES,
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

    def _load_credentials_from_file(self, credential_path: str) -> None:
        path = Path(credential_path).expanduser()
        if not path.exists():
            return

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return

        # Load only from the dedicated Antigravity slot. A bare/top-level or old
        # "geminiOauth" credential set has the wrong client/scopes and is ignored.
        payload = raw.get(self.credential_slot)
        if not isinstance(payload, dict):
            return

        file_access = payload.get("access_token")
        file_refresh = payload.get("refresh_token")
        expiry_ms = payload.get("expiry_date")

        if not self._cached_access_token and isinstance(file_access, str):
            self._cached_access_token = file_access
        if not self._cached_refresh_token and isinstance(file_refresh, str):
            self._cached_refresh_token = file_refresh
        if isinstance(expiry_ms, (int, float)):
            self._access_token_expiry = float(expiry_ms) / 1000.0

    def _persist_credentials(
        self,
        *,
        deadline: OAuthLoginDeadline | None = None,
    ) -> None:
        if not self.credential_path:
            return

        path = Path(self.credential_path).expanduser()
        AuthProfileStore(path).update_slot(
            self.credential_slot,
            {
                "access_token": self._cached_access_token,
                "refresh_token": self._cached_refresh_token,
                "token_type": "Bearer",
                "expiry_date": (
                    int(self._access_token_expiry * 1000)
                    if self._access_token_expiry
                    else None
                ),
            },
            lock_timeout=deadline.remaining() if deadline is not None else None,
            before_commit=deadline.check if deadline is not None else None,
        )

    def _metadata_payload(self) -> Dict[str, str]:
        return {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }

    def _build_headers(self, token: str) -> Dict[str, str]:
        # Identify as the Antigravity CLI. The gemini-cli/vscode User-Agent +
        # gl-node X-Goog-Api-Client make the backend 500 for an aicode token.
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "antigravity-cli",
            "Client-Metadata": json.dumps(self._metadata_payload()),
        }

    def fetch_available_models(
        self,
        *,
        deadline: OAuthLoginDeadline | None = None,
        access_token: str | None = None,
    ) -> list[Dict[str, Any]]:
        """Agent-usable Gemini models for the current entitlement.

        Calls Code Assist ``fetchAvailableModels`` and returns dicts with
        ``id``, ``display_name`` and ``supports_images``. Internal/tab/aux and
        deprecated ids are filtered out. Used to verify login and to optionally
        discover the live catalog. Requires the Antigravity client headers.
        """
        token = access_token or self._resolve_access_token()
        effective_timeout = (
            self.timeout if deadline is None else deadline.remaining(cap=self.timeout)
        )
        response = self._session.post(
            self._method_url(DEFAULT_CODE_ASSIST_MODELS_METHOD),
            headers=self._build_headers(token),
            json={},
            timeout=effective_timeout,
        )
        if deadline is not None:
            deadline.check()
        response.raise_for_status()
        data = response.json()
        if deadline is not None:
            deadline.check()
        models = data.get("models")
        if not isinstance(models, dict):
            return []
        deprecated = set((data.get("deprecatedModelIds") or {}).keys())
        out: list[Dict[str, Any]] = []
        for model_id, meta in models.items():
            if not isinstance(meta, dict) or model_id in deprecated:
                continue
            if meta.get("isInternal") or not meta.get("displayName"):
                continue
            if "GEMINI" not in str(meta.get("apiProvider") or ""):
                continue
            out.append(
                {
                    "id": model_id,
                    "display_name": meta.get("displayName"),
                    "supports_images": bool(meta.get("supportsImages")),
                }
            )
        if deadline is not None:
            deadline.check()
        return out

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
            raise RuntimeError(
                f"Token refresh succeeded but no access_token returned: {data}"
            )

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
        deadline: OAuthLoginDeadline | None = None,
        persist_credentials: bool = True,
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

        request_timeout = (
            self.timeout if deadline is None else deadline.remaining(cap=self.timeout)
        )
        response = self._session.post(
            self.token_url,
            data=payload,
            timeout=request_timeout,
        )
        if deadline is not None:
            deadline.check()
        response.raise_for_status()
        data = response.json()
        if deadline is not None:
            deadline.check()

        access_token = data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError(
                f"OAuth code exchange succeeded but no access_token returned: {data}"
            )

        refresh_token = data.get("refresh_token")
        cached_refresh_token = (
            refresh_token
            if isinstance(refresh_token, str) and refresh_token
            else self._cached_refresh_token
        )

        expires_in = data.get("expires_in", 3600)
        try:
            expires_in_s = int(expires_in)
        except (TypeError, ValueError):
            expires_in_s = 3600

        if deadline is not None:
            deadline.check()
        previous = (
            self._cached_access_token,
            self._cached_refresh_token,
            self._access_token_expiry,
        )
        self._cached_refresh_token = cached_refresh_token
        self._cached_access_token = access_token
        self._access_token_expiry = time.time() + expires_in_s
        try:
            if persist_credentials:
                self._persist_credentials(deadline=deadline)
        except BaseException:
            (
                self._cached_access_token,
                self._cached_refresh_token,
                self._access_token_expiry,
            ) = previous
            raise
        return access_token

    def _build_auth_url(
        self,
        redirect_uri: str,
        state: str,
        prompt_consent: bool,
        code_challenge: Optional[str] = None,
    ) -> str:
        scope = " ".join(self.scopes)
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
        deadline: OAuthLoginDeadline | None = None,
        persist_credentials: bool = True,
    ) -> str:
        active_deadline = deadline or OAuthLoginDeadline(timeout_seconds)
        active_deadline.check()

        # Headless environments: use authcode redirect flow (no local server)
        use_authcode = _is_headless_environment() or (
            os.environ.get("MOBILERUN_OAUTH_MANUAL")
            or os.environ.get("DROIDRUN_OAUTH_MANUAL")
            or ""
        ).lower() in ("1", "true", "yes")
        if use_authcode:
            return self.login_headless(
                open_browser=open_browser,
                timeout_seconds=timeout_seconds,
                prompt_consent=prompt_consent,
                deadline=active_deadline,
                persist_credentials=persist_credentials,
            )

        # Desktop: browser callback server
        result: Dict[str, Optional[str]] = {"code": None, "state": None, "error": None}
        done = threading.Event()
        callback_lock = threading.Lock()
        expected_state = secrets.token_hex(32)
        code_verifier, code_challenge = _pkce_pair()

        class _OAuthHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path != callback_path:
                    self.send_response(404)
                    self.end_headers()
                    return

                with callback_lock:
                    if done.is_set():
                        self.send_response(409)
                        self.end_headers()
                        return
                    params = parse_qs(parsed.query)
                    result["code"] = params.get("code", [None])[0]
                    result["state"] = params.get("state", [None])[0]
                    result["error"] = params.get("error", [None])[0]
                    ok = result["code"] is not None and result["error"] is None
                    done.set()
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

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

        try:
            httpd = HTTPServer((callback_host, callback_port), _OAuthHandler)
        except OSError as exc:
            print(
                f"Could not bind callback server on {callback_host}:{callback_port} ({exc}). "
                "Falling back to headless code entry."
            )
            return self.login_headless(
                open_browser=open_browser,
                timeout_seconds=timeout_seconds,
                prompt_consent=prompt_consent,
                deadline=active_deadline,
                persist_credentials=persist_credentials,
            )

        active_deadline.check()
        actual_port = httpd.server_address[1]
        redirect_uri = f"http://127.0.0.1:{actual_port}{callback_path}"
        auth_url = self._build_auth_url(
            redirect_uri=redirect_uri,
            state=expected_state,
            prompt_consent=prompt_consent,
            code_challenge=code_challenge,
        )

        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        httpd.daemon_threads = True
        server_thread.start()

        try:
            print(f"Open this URL to login:\n{auth_url}\n")
            if open_browser:
                open_browser_async(auth_url, webbrowser.open)

            if not done.wait(timeout=active_deadline.remaining()):
                raise TimeoutError(
                    "OAuth login timed out before callback was received."
                )
            active_deadline.check()

            if result["error"]:
                raise RuntimeError(f"OAuth callback returned error: {result['error']}")
            if result["state"] != expected_state:
                raise RuntimeError("OAuth callback state mismatch.")
            if not result["code"]:
                raise RuntimeError(
                    "OAuth callback did not include an authorization code."
                )

            access_token = self._exchange_authorization_code(
                result["code"],
                redirect_uri,
                code_verifier=code_verifier,
                deadline=active_deadline,
                persist_credentials=persist_credentials,
            )
        finally:
            httpd.shutdown()
            httpd.server_close()
        return access_token

    def login_headless(
        self,
        *,
        open_browser: bool = False,
        timeout_seconds: float = 300.0,
        input_fn: Any = input,
        prompt_consent: bool = True,
        deadline: OAuthLoginDeadline | None = None,
        persist_credentials: bool = True,
    ) -> str:
        """Headless OAuth flow for SSH/WSL environments."""
        active_deadline = deadline or OAuthLoginDeadline(timeout_seconds)
        active_deadline.check()

        code_verifier, code_challenge = _pkce_pair()
        expected_state = secrets.token_hex(32)
        redirect_uri = "https://codeassist.google.com/authcode"

        auth_url = self._build_auth_url(
            redirect_uri=redirect_uri,
            state=expected_state,
            prompt_consent=prompt_consent,
            code_challenge=code_challenge,
        )

        print(
            f"\nSign in with your Google account:\n"
            f"\n1. Open this link in your browser:\n   {auth_url}\n"
            f"\n2. Complete sign-in, then paste the authorization code shown on the page.\n"
        )
        if open_browser:
            open_browser_async(auth_url, webbrowser.open)

        input_queue: queue.Queue[Optional[str]] = queue.Queue()
        stop = threading.Event()
        need_more = threading.Event()

        def _reader() -> None:
            for _ in range(_MAX_CODE_ATTEMPTS):
                need_more.wait()
                need_more.clear()
                if stop.is_set():
                    return
                try:
                    input_queue.put(str(input_fn("Enter the authorization code: ")))
                except (EOFError, OSError):
                    input_queue.put(None)
                    return

        threading.Thread(target=_reader, daemon=True).start()

        try:
            for attempt in range(_MAX_CODE_ATTEMPTS):
                need_more.set()

                try:
                    raw = input_queue.get(timeout=active_deadline.remaining())
                except queue.Empty:
                    raise TimeoutError("OAuth login timed out.") from None
                active_deadline.check()

                if raw is None:
                    raise RuntimeError("Login failed — stdin closed.")
                if not raw.strip():
                    if attempt == 0:
                        print("No code entered. Try again.")
                        continue
                    raise RuntimeError("Login failed.") from None
                try:
                    code = _normalize_manual_code(raw, expected_state)
                except Exception:  # noqa: BLE001
                    if attempt == 0:
                        print("Invalid code. Try again.")
                        continue
                    raise RuntimeError("Login failed.") from None
                if code:
                    return self._exchange_authorization_code(
                        code,
                        redirect_uri,
                        code_verifier=code_verifier,
                        deadline=active_deadline,
                        persist_credentials=persist_credentials,
                    )
                if attempt == 0:
                    print("Invalid code. Try again.")
                    continue
                raise RuntimeError("Login failed.")
            raise RuntimeError("Login failed.")
        finally:
            # stop.set() prevents the reader from starting a new input() call.
            # It cannot interrupt an input() already in progress (Python limitation);
            # the daemon thread dies with the process.
            stop.set()
            need_more.set()

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

    @staticmethod
    def _image_mime_type(raw: bytes) -> str:
        if raw.startswith(b"\xff\xd8"):
            return "image/jpeg"
        if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
            return "image/webp"
        if raw.startswith(b"GIF8"):
            return "image/gif"
        return "image/png"

    @classmethod
    def _message_parts(cls, message: ChatMessage) -> list[Dict[str, Any]]:
        parts: list[Dict[str, Any]] = []
        for block in message.blocks or []:
            if isinstance(block, TextBlock):
                if block.text:
                    parts.append({"text": block.text})
            elif isinstance(block, ImageBlock):
                raw = block.resolve_image(as_base64=False).read()
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": cls._image_mime_type(raw),
                            "data": base64.b64encode(raw).decode("ascii"),
                        }
                    }
                )
        return parts

    def _to_code_assist_request(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # NOTE: deliberately not using ``self.convert_chat_messages`` — the
        # llama_index base helper is text-only and raises on ImageBlocks.
        contents: list[Dict[str, Any]] = []
        system_chunks: list[str] = []

        for msg in messages:
            parts = self._message_parts(msg)
            if msg.role == MessageRole.SYSTEM:
                system_chunks.extend(part["text"] for part in parts if part.get("text"))
                continue

            role = "model" if msg.role == MessageRole.ASSISTANT else "user"
            if any("inlineData" in part for part in parts):
                contents.append({"role": role, "parts": parts})
            else:
                # Text-only messages keep the single-text-part shape the API
                # already receives.
                contents.append(
                    {"role": role, "parts": [{"text": self._message_text(msg)}]}
                )

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
        for ignored in _IGNORED_REQUEST_KWARGS:
            request_extra.pop(ignored, None)
        request.update(request_extra)

        payload: Dict[str, Any] = {
            "model": self.model,
            "request": request,
            "userAgent": "droidrun",
            "requestId": f"droidrun-{int(time.time() * 1000)}-{secrets.token_hex(4)}",
        }

        # Strip internal kwargs Google rejects; the consumer entitlement is
        # project-less, so never send a project (also filtered by the ignore set).
        safe_kwargs = {
            k: v for k, v in kwargs.items() if k not in _IGNORED_REQUEST_KWARGS
        }
        payload.update(safe_kwargs)
        payload.pop("project", None)
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
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        token = self._resolve_access_token()
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
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT, content=accumulated
                    ),
                    delta=delta,
                    raw=raw_chunk,
                    additional_kwargs={
                        "trace_id": raw_chunk.get("traceId"),
                        "usage": (raw_chunk.get("response") or {}).get("usageMetadata"),
                        "model_version": (raw_chunk.get("response") or {}).get(
                            "modelVersion"
                        ),
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
