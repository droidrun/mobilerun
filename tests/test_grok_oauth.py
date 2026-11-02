from __future__ import annotations

import asyncio
import io
import json
import multiprocessing
import stat
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
    ToolCallBlock,
)

from mobilerun.agent.providers.grok import GROK_OAUTH_DEFAULT_MODEL
from mobilerun.agent.usage import get_usage_from_response
from mobilerun.agent.utils.oauth.grok_oauth_llm import (
    DEFAULT_GROK_CONTEXT_WINDOW,
    DEFAULT_GROK_MODEL,
    DEFAULT_GROK_OAUTH_CLIENT_ID,
    DEFAULT_GROK_OAUTH_ISSUER,
    DEFAULT_GROK_OAUTH_PROXY,
    DEFAULT_GROK_OAUTH_SCOPES,
    GROK_CLI_COMPAT_VERSION,
    GROK_CLI_COMPAT_VERSION_HEADER,
    GrokIDTokenValidator,
    GrokOAuth,
    GrokOAuthAuth,
    GrokOAuthCredentials,
    GrokOAuthCredentialStore,
    GrokOAuthError,
    GrokOAuthReloginRequired,
    GrokOAuthSessionManager,
    _parse_callback_query,
)
from mobilerun.agent.utils.oauth.login_timeout import OAuthLoginDeadline
from mobilerun.config_manager import auth_profile_store, env_keys
from mobilerun.config_manager.auth_profile_store import (
    AuthProfileFormatError,
    AuthProfileStore,
)


class _AcceptingIDTokenValidator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def validate(self, token: str, *, nonce: str | None):  # type: ignore[no-untyped-def]
        self.calls.append((token, nonce))
        return {"sub": "user"}


def test_oauth_default_uses_shared_xai_catalog_default() -> None:
    assert DEFAULT_GROK_MODEL == GROK_OAUTH_DEFAULT_MODEL == "grok-4.7"


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)


def _request_timeout(request: httpx.Request) -> float:
    return float(request.extensions["timeout"]["read"])


def _credentials(
    access_token: str = "access-old",
    refresh_token: str = "refresh-old",
    *,
    expires_at_ms: int | None = None,
) -> GrokOAuthCredentials:
    return GrokOAuthCredentials(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at_ms=expires_at_ms or int(time.time() * 1000) + 3_600_000,
    )


def _responses_payload(
    *,
    output: list[dict[str, object]] | None = None,
    input_tokens: int = 8,
    output_tokens: int = 3,
) -> dict[str, object]:
    return {
        "id": "resp_test",
        "created_at": int(time.time()),
        "model": DEFAULT_GROK_MODEL,
        "object": "response",
        "output": output or [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _responses_sse() -> bytes:
    completed = _responses_payload(
        output=[
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "hello", "annotations": []}
                ],
            }
        ],
    )
    events = (
        {
            "type": "response.output_text.delta",
            "content_index": 0,
            "delta": "hel",
            "item_id": "msg_test",
            "logprobs": [],
            "output_index": 0,
            "sequence_number": 1,
        },
        {
            "type": "response.completed",
            "response": completed,
            "sequence_number": 2,
        },
    )
    return (
        "".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
        )
        + "data: [DONE]\n\n"
    ).encode()


def _refresh_in_subprocess(path, calls, barrier, results):  # type: ignore[no-untyped-def]
    def handler(_: httpx.Request) -> httpx.Response:
        with calls.get_lock():
            calls.value += 1
        return httpx.Response(
            200,
            json={
                "access_token": "access-new",
                "refresh_token": "refresh-rotated",
                "expires_in": 3600,
            },
        )

    try:
        manager = GrokOAuthSessionManager(
            credential_store=GrokOAuthCredentialStore(path),
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            id_token_validator=_AcceptingIDTokenValidator(),  # type: ignore[arg-type]
        )
        barrier.wait(timeout=10)
        credentials = manager.get_valid_credentials()
        results.put(("ok", credentials.access_token))
    except BaseException as exc:
        results.put(("error", type(exc).__name__, str(exc)))


def test_auth_profile_store_preserves_siblings_and_writes_private_file(tmp_path: Path):
    path = tmp_path / "auth-profiles.json"
    path.write_text(json.dumps({"openaiOauth": {"access": "keep"}}))

    AuthProfileStore(path).update_slot("grokOauth", {"accessToken": "secret"})

    payload = json.loads(path.read_text())
    assert payload["openaiOauth"] == {"access": "keep"}
    assert payload["grokOauth"] == {"accessToken": "secret"}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".auth-profiles.json.*.tmp"))


def test_auth_profile_store_writes_when_fchmod_is_unavailable(
    monkeypatch, tmp_path: Path
):
    path = tmp_path / "auth-profiles.json"
    monkeypatch.setattr(auth_profile_store, "_FCHMOD", None)

    AuthProfileStore(path).update_slot("grokOauth", {"accessToken": "secret"})

    assert json.loads(path.read_text()) == {"grokOauth": {"accessToken": "secret"}}
    assert not list(tmp_path.glob(".auth-profiles.json.*.tmp"))


def test_auth_profile_store_rejects_malformed_existing_json(tmp_path: Path):
    path = tmp_path / "auth-profiles.json"
    path.write_text("not-json")

    with pytest.raises(AuthProfileFormatError, match="malformed"):
        AuthProfileStore(path).update_slot("grokOauth", {"accessToken": "new"})

    assert path.read_text() == "not-json"


def test_auth_profile_store_replace_failure_preserves_original_and_cleans_temp(
    monkeypatch, tmp_path: Path
):
    path = tmp_path / "auth-profiles.json"
    original = {"openaiOauth": {"access": "keep"}}
    path.write_text(json.dumps(original))

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(
        "mobilerun.config_manager.auth_profile_store.os.replace", fail_replace
    )

    with pytest.raises(OSError, match="simulated atomic replace failure"):
        AuthProfileStore(path).update_slot(
            "grokOauth", {"accessToken": "must-not-be-written"}
        )

    assert json.loads(path.read_text()) == original
    assert not list(tmp_path.glob(".auth-profiles.json.*.tmp"))


def test_saved_api_keys_use_shared_transaction_and_reject_malformed(
    monkeypatch, tmp_path: Path
):
    path = tmp_path / "auth-profiles.json"
    monkeypatch.setattr(env_keys, "AUTH_PROFILES_PATH", path)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    path.write_text(json.dumps({"grokOauth": {"accessToken": "keep"}}))

    env_keys.save_env_keys({"xai": "api-key"})
    payload = json.loads(path.read_text())
    assert payload["grokOauth"] == {"accessToken": "keep"}
    assert payload["apiKeys"]["xai"] == "api-key"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    path.write_text("broken")
    with pytest.raises(AuthProfileFormatError):
        env_keys.save_env_keys({"xai": "replacement"})
    assert path.read_text() == "broken"


def test_grok_credential_schema_round_trip_does_not_persist_id_token(tmp_path: Path):
    store = GrokOAuthCredentialStore(tmp_path / "auth-profiles.json")
    credentials = _credentials()
    store.save(credentials)

    assert store.load() == credentials
    raw = json.loads(store.path.read_text())["grokOauth"]
    assert raw == credentials.to_payload()
    assert raw["type"] == "oauth"
    assert raw["provider"] == "xai-grok"
    assert "idToken" not in raw
    assert raw["issuer"] == DEFAULT_GROK_OAUTH_ISSUER
    assert raw["clientId"] == DEFAULT_GROK_OAUTH_CLIENT_ID


def test_grok_credentials_reject_unpinned_issuer_and_client():
    payload = _credentials().to_payload()
    payload["issuer"] = "https://example.invalid"
    with pytest.raises(ValueError, match="issuer"):
        GrokOAuthCredentials.from_payload(payload)

    payload = _credentials().to_payload()
    payload["provider"] = "other"
    with pytest.raises(ValueError, match="credential type"):
        GrokOAuthCredentials.from_payload(payload)

    payload = _credentials().to_payload()
    payload["clientId"] = "other"
    with pytest.raises(ValueError, match="clientId"):
        GrokOAuthCredentials.from_payload(payload)


def test_id_token_validator_checks_es256_claims_and_nonce():
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()
    now = int(time.time())
    token = jwt.encode(
        {
            "sub": "user",
            "iss": DEFAULT_GROK_OAUTH_ISSUER,
            "aud": DEFAULT_GROK_OAUTH_CLIENT_ID,
            "iat": now,
            "exp": now + 300,
            "nonce": "expected",
        },
        private_key,
        algorithm="ES256",
        headers={"kid": "test"},
    )
    observed_timeouts: list[float] = []
    jwks_client = SimpleNamespace(timeout=30.0)

    def signing_key(_token: str):  # type: ignore[no-untyped-def]
        observed_timeouts.append(jwks_client.timeout)
        return SimpleNamespace(key=public_key)

    jwks_client.get_signing_key_from_jwt = signing_key
    validator = GrokIDTokenValidator(jwks_client)

    deadline = OAuthLoginDeadline(2)
    assert (
        validator.validate(token, nonce="expected", deadline=deadline)["sub"] == "user"
    )
    assert observed_timeouts[0] == pytest.approx(2, abs=0.1)
    assert jwks_client.timeout == 30.0
    with pytest.raises(jwt.InvalidTokenError, match="nonce"):
        validator.validate(token, nonce="wrong")


def test_id_token_validator_recomputes_timeout_for_each_jwks_fetch(monkeypatch):
    clock = _FakeClock()
    client = jwt.PyJWKClient("https://auth.x.ai/.well-known/jwks.json", timeout=30)
    observed_timeouts: list[float] = []

    def fetch_data():  # type: ignore[no-untyped-def]
        observed_timeouts.append(client.timeout)
        clock.advance(1)
        return {}

    def signing_key(_token: str):  # type: ignore[no-untyped-def]
        client.fetch_data()
        client.fetch_data()
        return SimpleNamespace(key="public-key")

    client.fetch_data = fetch_data
    client.get_signing_key_from_jwt = signing_key
    monkeypatch.setattr(jwt, "decode", lambda *args, **kwargs: {"nonce": "expected"})

    GrokIDTokenValidator(client).validate(
        "id-token",
        nonce="expected",
        deadline=OAuthLoginDeadline(3, clock=clock, sleeper=clock.sleep),
    )

    assert observed_timeouts == [3, 2]
    assert client.timeout == 30


def test_legacy_id_token_validator_subclass_keeps_prior_signature(tmp_path: Path):
    class LegacyValidator(GrokIDTokenValidator):
        def validate(self, token: str, *, nonce: str | None):  # type: ignore[no-untyped-def]
            return {"sub": "user", "nonce": nonce}

    manager = GrokOAuthSessionManager(
        credential_store=GrokOAuthCredentialStore(tmp_path / "auth.json"),
        id_token_validator=LegacyValidator(SimpleNamespace()),
    )

    credentials = manager._credentials_from_token_response(
        {"access_token": "access", "id_token": "id-token"},
        nonce="expected",
        deadline=OAuthLoginDeadline(1),
    )

    assert credentials.access_token == "access"


def test_id_token_validator_rejects_bad_signature():
    trusted_key = ec.generate_private_key(ec.SECP256R1())
    untrusted_key = ec.generate_private_key(ec.SECP256R1())
    now = int(time.time())
    token = jwt.encode(
        {
            "sub": "user",
            "iss": DEFAULT_GROK_OAUTH_ISSUER,
            "aud": DEFAULT_GROK_OAUTH_CLIENT_ID,
            "iat": now,
            "exp": now + 300,
        },
        untrusted_key,
        algorithm="ES256",
        headers={"kid": "test"},
    )
    validator = GrokIDTokenValidator(
        SimpleNamespace(
            get_signing_key_from_jwt=lambda _: SimpleNamespace(
                key=trusted_key.public_key()
            )
        )
    )

    with pytest.raises(jwt.InvalidSignatureError):
        validator.validate(token, nonce=None)


@pytest.mark.parametrize(
    ("claim_overrides", "expected_error"),
    (
        ({"iss": "https://issuer.invalid"}, jwt.InvalidIssuerError),
        ({"aud": "different-client"}, jwt.InvalidAudienceError),
        ({"iat": 1, "exp": 2}, jwt.ExpiredSignatureError),
    ),
)
def test_id_token_validator_rejects_invalid_registered_claims(
    claim_overrides: dict[str, object],
    expected_error: type[jwt.InvalidTokenError],
):
    private_key = ec.generate_private_key(ec.SECP256R1())
    now = int(time.time())
    claims: dict[str, object] = {
        "sub": "user",
        "iss": DEFAULT_GROK_OAUTH_ISSUER,
        "aud": DEFAULT_GROK_OAUTH_CLIENT_ID,
        "iat": now,
        "exp": now + 300,
    }
    claims.update(claim_overrides)
    token = jwt.encode(
        claims,
        private_key,
        algorithm="ES256",
        headers={"kid": "test"},
    )
    validator = GrokIDTokenValidator(
        SimpleNamespace(
            get_signing_key_from_jwt=lambda _: SimpleNamespace(
                key=private_key.public_key()
            )
        )
    )

    with pytest.raises(expected_error):
        validator.validate(token, nonce=None)


def test_authorization_url_is_pinned_pkce_and_complete():
    url = GrokOAuth._build_auth_url(
        redirect_uri="http://127.0.0.1:54321/callback",
        code_challenge="challenge",
        state="state",
        nonce="nonce",
    )
    expected_query = urlencode(
        [
            ("response_type", "code"),
            ("client_id", DEFAULT_GROK_OAUTH_CLIENT_ID),
            ("redirect_uri", "http://127.0.0.1:54321/callback"),
            ("scope", " ".join(DEFAULT_GROK_OAUTH_SCOPES)),
            ("code_challenge", "challenge"),
            ("code_challenge_method", "S256"),
            ("state", "state"),
            ("nonce", "nonce"),
            ("referrer", "grok-build"),
        ]
    )
    assert url == f"{DEFAULT_GROK_OAUTH_ISSUER}/oauth2/authorize?{expected_query}"

    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    assert f"{parsed.scheme}://{parsed.netloc}" == DEFAULT_GROK_OAUTH_ISSUER
    assert parsed.path == "/oauth2/authorize"
    assert query["client_id"] == [DEFAULT_GROK_OAUTH_CLIENT_ID]
    assert query["redirect_uri"] == ["http://127.0.0.1:54321/callback"]
    assert query["scope"] == [" ".join(DEFAULT_GROK_OAUTH_SCOPES)]
    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"] == ["challenge"]
    assert query["state"] == ["state"]
    assert query["nonce"] == ["nonce"]
    assert query["referrer"] == ["grok-build"]


def test_callback_query_rejects_duplicate_code_state_and_error_values():
    assert _parse_callback_query("code=one&state=expected") == {
        "code": "one",
        "state": "expected",
        "error": None,
    }
    for query in (
        "code=one&code=two&state=expected",
        "code=&code=one&state=expected",
        "code=one&state=a&state=b",
        "code=one&state=&state=expected",
        "error=denied&error=other",
    ):
        parsed = _parse_callback_query(query)
        assert parsed["error"] == "invalid_callback"


def test_browser_login_uses_random_loopback_callback_and_exchanges_code(
    monkeypatch, tmp_path: Path
):
    clock = _FakeClock()
    deadline = OAuthLoginDeadline(5, clock=clock, sleeper=clock.sleep)
    token_requests: list[httpx.Request] = []

    def token_handler(request: httpx.Request) -> httpx.Response:
        token_requests.append(request)
        return httpx.Response(
            200,
            json={
                "access_token": "access",
                "refresh_token": "refresh",
                "expires_in": 3600,
                "id_token": "signed-id-token",
            },
        )

    manager = GrokOAuthSessionManager(
        credential_store=GrokOAuthCredentialStore(tmp_path / "auth.json"),
        http_client=httpx.Client(transport=httpx.MockTransport(token_handler)),
        id_token_validator=_AcceptingIDTokenValidator(),  # type: ignore[arg-type]
    )
    llm = GrokOAuth(oauth_session_manager=manager)
    opened: list[str] = []
    callback_handler: list[type] = []

    class FakeRequest:
        def __init__(self, target: str) -> None:
            self.input = io.BytesIO(
                f"GET {target} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode()
            )
            self.output = io.BytesIO()

        def makefile(self, mode: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            return self.input if "r" in mode else self.output

        def sendall(self, data: bytes) -> None:
            self.output.write(data)

        def close(self) -> None:
            return

    class FakeHTTPServer:
        server_name = "127.0.0.1"
        server_port = 54321

        def __init__(self, address, handler):  # type: ignore[no-untyped-def]
            assert address == ("127.0.0.1", 0)
            self.server_address = ("127.0.0.1", self.server_port)
            callback_handler.append(handler)

        def serve_forever(self) -> None:
            return

        def shutdown(self) -> None:
            return

        def server_close(self) -> None:
            return

    def complete_in_browser(authorization_url: str) -> bool:
        opened.append(authorization_url)
        clock.advance(1)
        query = parse_qs(urlparse(authorization_url).query)
        redirect_uri = query["redirect_uri"][0]
        assert redirect_uri.startswith("http://127.0.0.1:")
        assert redirect_uri.endswith("/callback")
        assert ":0/" not in redirect_uri
        callback_url = f"{redirect_uri}?code=code&state={query['state'][0]}"
        callback_target = urlparse(callback_url)
        request = FakeRequest(f"{callback_target.path}?{callback_target.query}")
        callback_handler[0](request, ("127.0.0.1", 12345), fake_server[0])
        assert b"200" in request.output.getvalue().splitlines()[0]
        return True

    fake_server: list[FakeHTTPServer] = []

    def make_server(address, handler):  # type: ignore[no-untyped-def]
        server = FakeHTTPServer(address, handler)
        fake_server.append(server)
        return server

    monkeypatch.setattr(
        "mobilerun.agent.utils.oauth.grok_oauth_llm.HTTPServer", make_server
    )
    monkeypatch.setattr("webbrowser.open", complete_in_browser)
    credentials = llm.login(
        open_browser=True,
        timeout_seconds=5,
        deadline=deadline,
    )

    assert credentials.access_token == "access"
    assert len(opened) == 1
    token_form = parse_qs(token_requests[0].content.decode())
    assert token_form["code"] == ["code"]
    assert token_form["redirect_uri"][0].startswith("http://127.0.0.1:")
    assert token_form["code_verifier"][0]
    assert _request_timeout(token_requests[0]) == pytest.approx(4)


def test_code_exchange_uses_form_validates_id_and_saves_only_session(tmp_path: Path):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "access_token": "access-new",
                "refresh_token": "refresh-new",
                "expires_in": 3600,
                "token_type": "Bearer",
                "scope": "openid offline_access api:access",
                "id_token": "signed-id-token",
            },
        )

    validator = _AcceptingIDTokenValidator()
    manager = GrokOAuthSessionManager(
        credential_store=GrokOAuthCredentialStore(tmp_path / "auth.json"),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        id_token_validator=validator,  # type: ignore[arg-type]
    )
    credentials = manager.exchange_authorization_code(
        code="auth-code",
        redirect_uri="http://127.0.0.1:54321/callback",
        code_verifier="verifier",
        nonce="nonce",
    )

    form = parse_qs(requests[0].content.decode())
    assert form["grant_type"] == ["authorization_code"]
    assert form["client_id"] == [DEFAULT_GROK_OAUTH_CLIENT_ID]
    assert form["code_verifier"] == ["verifier"]
    assert requests[0].headers["Accept"] == "application/json"
    assert validator.calls == [("signed-id-token", "nonce")]
    assert credentials.refresh_token == "refresh-new"
    raw = json.loads((tmp_path / "auth.json").read_text())["grokOauth"]
    assert "signed-id-token" not in json.dumps(raw)


def test_non_browser_token_accepts_missing_id_token_and_uses_access_jwt_expiry(
    tmp_path: Path,
):
    now = int(time.time())
    # The access-token claim is used only as refresh scheduling metadata.
    unsigned_access = jwt.encode(
        {"exp": now + 900}, "not-a-provider-key-that-is-long-enough", algorithm="HS256"
    )

    validator = _AcceptingIDTokenValidator()
    manager = GrokOAuthSessionManager(
        credential_store=GrokOAuthCredentialStore(tmp_path / "auth.json"),
        id_token_validator=validator,  # type: ignore[arg-type]
    )

    credentials = manager._credentials_from_token_response(
        {
            "access_token": unsigned_access,
            "refresh_token": "refresh",
        },
        nonce=None,
    )

    assert credentials.expires_at_ms == (now + 900) * 1000
    assert validator.calls == []


def test_browser_code_exchange_rejects_missing_id_token(tmp_path: Path):
    manager = GrokOAuthSessionManager(
        credential_store=GrokOAuthCredentialStore(tmp_path / "auth.json"),
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={
                        "access_token": "access",
                        "refresh_token": "refresh",
                        "expires_in": 3600,
                    },
                )
            )
        ),
        id_token_validator=_AcceptingIDTokenValidator(),  # type: ignore[arg-type]
    )

    with pytest.raises(GrokOAuthError, match="did not contain an ID token"):
        manager.exchange_authorization_code(
            code="code",
            redirect_uri="http://127.0.0.1:54321/callback",
            code_verifier="verifier",
            nonce="nonce",
        )

    assert manager.credential_store.load() is None


@pytest.mark.parametrize(
    ("status_code", "body"),
    (
        (200, "secret-token-not-json"),
        (400, '{"error_description":"secret-token"}'),
    ),
)
def test_token_errors_never_expose_response_bodies(
    status_code: int, body: str, tmp_path: Path
):
    manager = GrokOAuthSessionManager(
        credential_store=GrokOAuthCredentialStore(tmp_path / "auth.json"),
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(status_code, text=body)
            )
        ),
        id_token_validator=_AcceptingIDTokenValidator(),  # type: ignore[arg-type]
    )

    with pytest.raises(GrokOAuthError) as captured:
        manager.exchange_authorization_code(
            code="code",
            redirect_uri="http://127.0.0.1:54321/callback",
            code_verifier="verifier",
            nonce="nonce",
        )

    assert "secret-token" not in str(captured.value)


def test_authorization_code_timeout_is_not_retried(tmp_path: Path):
    attempts = 0
    delays: list[float] = []
    request_timeouts: list[float] = []
    deadline = OAuthLoginDeadline(2)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        request_timeouts.append(_request_timeout(request))
        raise httpx.ReadTimeout("ambiguous token exchange timeout", request=request)

    manager = GrokOAuthSessionManager(
        credential_store=GrokOAuthCredentialStore(tmp_path / "auth.json"),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        id_token_validator=_AcceptingIDTokenValidator(),  # type: ignore[arg-type]
        request_timeout=20,
        retry_backoff_seconds=(0.01, 0.02),
        sleep=delays.append,
    )

    with pytest.raises(GrokOAuthError, match="transient network error") as captured:
        manager.exchange_authorization_code(
            code="one-time-code",
            redirect_uri="http://127.0.0.1:54321/callback",
            code_verifier="verifier",
            nonce="nonce",
            deadline=deadline,
        )

    assert attempts == 1
    assert request_timeouts == pytest.approx([2], abs=0.1)
    assert delays == []
    assert "one-time-code" not in str(captured.value)
    assert manager.credential_store.load() is None


def test_device_flow_matches_xai_surface_and_handles_pending(
    monkeypatch, capsys, tmp_path: Path
):
    requests: list[httpx.Request] = []
    verification_uri = "https://accounts.x.ai/device?code=ABCD-1234"
    opened: list[str] = []
    opened_event = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/device/code"):
            return httpx.Response(
                200,
                json={
                    "device_code": "device-secret",
                    "user_code": "ABCD-1234",
                    "verification_uri_complete": verification_uri,
                    "expires_in": 1800,
                    "interval": 1,
                },
            )
        if len([r for r in requests if r.url.path.endswith("/token")]) == 1:
            return httpx.Response(400, json={"error": "authorization_pending"})
        return httpx.Response(
            200,
            json={
                "access_token": "access",
                "refresh_token": "refresh",
                "expires_in": 3600,
            },
        )

    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(
        "webbrowser.open",
        lambda url: (opened.append(url), opened_event.set()),
    )
    manager = GrokOAuthSessionManager(
        credential_store=GrokOAuthCredentialStore(tmp_path / "auth.json"),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        id_token_validator=_AcceptingIDTokenValidator(),  # type: ignore[arg-type]
    )
    llm = GrokOAuth(oauth_session_manager=manager)

    credentials = llm.login(
        device_code=True,
        open_browser=True,
        timeout_seconds=10,
    )

    device_request = requests[0]
    assert device_request.headers["x-grok-client-surface"] == "grok-build"
    assert device_request.headers["Accept"] == "application/json"
    assert parse_qs(device_request.content.decode()) == {
        "client_id": [DEFAULT_GROK_OAUTH_CLIENT_ID],
        "scope": [" ".join(DEFAULT_GROK_OAUTH_SCOPES)],
    }
    token_form = parse_qs(requests[1].content.decode())
    assert token_form["device_code"] == ["device-secret"]
    assert "user_code" not in token_form
    assert credentials.access_token == "access"
    assert manager.credential_store.load() == credentials
    assert opened_event.wait(timeout=1)
    assert opened == [verification_uri]
    assert verification_uri in capsys.readouterr().out


def test_bind_fallback_preserves_deadline_and_no_browser(monkeypatch, tmp_path: Path):
    manager = GrokOAuthSessionManager(
        credential_store=GrokOAuthCredentialStore(tmp_path / "auth.json")
    )
    llm = GrokOAuth(oauth_session_manager=manager)
    deadline = OAuthLoginDeadline(2)
    expected = _credentials()
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "mobilerun.agent.utils.oauth.grok_oauth_llm._is_headless_environment",
        lambda: False,
    )
    monkeypatch.setattr(
        "mobilerun.agent.utils.oauth.grok_oauth_llm.HTTPServer",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("unavailable")),
    )
    monkeypatch.setattr(
        llm,
        "_login_device_code",
        lambda **kwargs: (calls.append(kwargs), expected)[1],
    )

    assert llm.login(open_browser=False, deadline=deadline) == expected
    assert calls == [{"deadline": deadline, "open_browser": False}]


def test_device_poll_retries_shrink_deadline_and_do_not_save_after_expiry(
    tmp_path: Path,
):
    clock = _FakeClock()
    deadline = OAuthLoginDeadline(2, clock=clock, sleeper=clock.sleep)
    poll_attempts = 0
    request_timeouts: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal poll_attempts
        request_timeouts.append(_request_timeout(request))
        if request.url.path.endswith("/device/code"):
            clock.advance(0.4)
            return httpx.Response(
                200,
                json={
                    "device_code": "device-secret",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://accounts.x.ai/device",
                    "expires_in": 1800,
                    "interval": 1,
                },
            )
        poll_attempts += 1
        if poll_attempts == 1:
            clock.advance(0.3)
            raise httpx.ConnectError("temporary connect failure", request=request)
        if poll_attempts == 2:
            clock.advance(0.2)
            return httpx.Response(503, text="sensitive-upstream-body")
        clock.advance(0.36)
        return httpx.Response(
            200,
            json={
                "access_token": "access",
                "refresh_token": "refresh",
                "expires_in": 3600,
            },
        )

    store = GrokOAuthCredentialStore(tmp_path / "auth.json")
    manager = GrokOAuthSessionManager(
        credential_store=store,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        id_token_validator=_AcceptingIDTokenValidator(),  # type: ignore[arg-type]
        request_timeout=20,
        retry_backoff_seconds=(0.25, 0.5),
    )

    with pytest.raises(TimeoutError, match="OAuth login timed out"):
        GrokOAuth(oauth_session_manager=manager).login(
            device_code=True,
            open_browser=False,
            deadline=deadline,
        )

    assert poll_attempts == 3
    assert request_timeouts == pytest.approx([2.0, 1.6, 1.05, 0.35])
    assert clock.sleeps == pytest.approx([0.25, 0.5])
    assert store.load() is None


def test_refresh_preserves_rotated_token_and_is_coordinated_across_managers(
    tmp_path: Path,
):
    path = tmp_path / "auth.json"
    store = GrokOAuthCredentialStore(path)
    store.save(_credentials(expires_at_ms=1))
    calls = 0
    calls_lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        with calls_lock:
            calls += 1
        return httpx.Response(
            200,
            json={
                "access_token": "access-new",
                "refresh_token": "refresh-rotated",
                "expires_in": 3600,
            },
        )

    managers = [
        GrokOAuthSessionManager(
            credential_store=GrokOAuthCredentialStore(path),
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            id_token_validator=_AcceptingIDTokenValidator(),  # type: ignore[arg-type]
        )
        for _ in range(2)
    ]
    results: list[GrokOAuthCredentials] = []
    threads = [
        threading.Thread(
            target=lambda manager=m: results.append(manager.get_valid_credentials())
        )
        for m in managers
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == 1
    assert len(results) == 2
    assert {result.access_token for result in results} == {"access-new"}
    assert store.load().refresh_token == "refresh-rotated"  # type: ignore[union-attr]


def test_refresh_retries_timeout_and_5xx_then_persists_rotation(tmp_path: Path):
    path = tmp_path / "auth.json"
    store = GrokOAuthCredentialStore(path)
    store.save(_credentials(expires_at_ms=1))
    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("temporary timeout", request=request)
        if attempts == 2:
            return httpx.Response(502, text="sensitive-upstream-body")
        return httpx.Response(
            200,
            json={
                "access_token": "access-new",
                "refresh_token": "refresh-rotated",
                "expires_in": 3600,
            },
        )

    manager = GrokOAuthSessionManager(
        credential_store=store,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        id_token_validator=_AcceptingIDTokenValidator(),  # type: ignore[arg-type]
        retry_backoff_seconds=(0.01, 0.02),
        sleep=delays.append,
    )

    credentials = manager.get_valid_credentials()

    assert attempts == 3
    assert delays == [0.01, 0.02]
    assert credentials.access_token == "access-new"
    assert store.load().refresh_token == "refresh-rotated"  # type: ignore[union-attr]


@pytest.mark.parametrize("oauth_error", ("invalid_grant", "invalid_client"))
def test_permanent_refresh_rejection_requires_relogin_without_changing_disk(
    oauth_error: str, tmp_path: Path
):
    path = tmp_path / "auth.json"
    store = GrokOAuthCredentialStore(path)
    store.save(_credentials(expires_at_ms=1))
    store.profile_store.update_slot("openaiOauth", {"access": "keep-sibling"})
    before = path.read_bytes()

    manager = GrokOAuthSessionManager(
        credential_store=store,
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    400,
                    json={
                        "error": oauth_error,
                        "error_description": "sensitive-refresh-response-body",
                    },
                )
            )
        ),
        id_token_validator=_AcceptingIDTokenValidator(),  # type: ignore[arg-type]
        sleep=lambda _: None,
    )
    manager._credentials = _credentials(expires_at_ms=1)

    with pytest.raises(
        GrokOAuthReloginRequired, match="re-login is required"
    ) as captured:
        manager.get_valid_credentials()

    assert "sensitive-refresh-response-body" not in str(captured.value)
    assert manager._credentials is None
    assert path.read_bytes() == before
    profile = json.loads(path.read_text())
    assert profile["openaiOauth"] == {"access": "keep-sibling"}
    assert profile["grokOauth"]["accessToken"] == "access-old"


def test_refresh_is_coordinated_across_processes(tmp_path: Path):
    path = tmp_path / "auth.json"
    store = GrokOAuthCredentialStore(path)
    store.save(_credentials(expires_at_ms=1))
    context = multiprocessing.get_context("spawn")
    calls = context.Value("i", 0)
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_refresh_in_subprocess,
            args=(str(path), calls, barrier, results),
        )
        for _ in range(2)
    ]

    try:
        for process in processes:
            process.start()
        received = [results.get(timeout=15) for _ in processes]
        for process in processes:
            process.join(timeout=15)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert received == [("ok", "access-new"), ("ok", "access-new")]
    assert [process.exitcode for process in processes] == [0, 0]
    assert calls.value == 1
    assert store.load().refresh_token == "refresh-rotated"  # type: ignore[union-attr]


class _StubSessionManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_valid_credentials(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        if kwargs.get("force_refresh"):
            return _credentials("access-new", "refresh-new")
        return _credentials()


def test_sync_auth_injects_headers_and_retries_exactly_one_401():
    manager = _StubSessionManager()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        assert request.headers["X-XAI-Token-Auth"] == "xai-grok-cli"
        assert request.headers["x-grok-model-override"] == DEFAULT_GROK_MODEL
        assert (
            request.headers[GROK_CLI_COMPAT_VERSION_HEADER] == GROK_CLI_COMPAT_VERSION
        )
        return httpx.Response(401 if len(seen) == 1 else 200, json={})

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        auth=GrokOAuthAuth(manager, model=DEFAULT_GROK_MODEL),  # type: ignore[arg-type]
    ) as client:
        response = client.post("https://example.test/responses", json={"input": "hi"})

    assert response.status_code == 200
    assert seen == ["Bearer access-old", "Bearer access-new"]
    assert manager.calls == [
        {},
        {"force_refresh": True, "rejected_access_token": "access-old"},
    ]


def test_sync_auth_replays_only_once_on_consecutive_401s():
    manager = _StubSessionManager()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        return httpx.Response(401, json={})

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        auth=GrokOAuthAuth(manager, model=DEFAULT_GROK_MODEL),  # type: ignore[arg-type]
    ) as client:
        response = client.post("https://example.test/responses", json={"input": "hi"})

    assert response.status_code == 401
    assert seen == ["Bearer access-old", "Bearer access-new"]
    assert manager.calls == [
        {},
        {"force_refresh": True, "rejected_access_token": "access-old"},
    ]


@pytest.mark.parametrize("status_code", (400, 403, 429))
def test_sync_auth_does_not_replay_non_401_responses(status_code: int):
    manager = _StubSessionManager()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        return httpx.Response(status_code, json={})

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        auth=GrokOAuthAuth(manager, model=DEFAULT_GROK_MODEL),  # type: ignore[arg-type]
    ) as client:
        response = client.post("https://example.test/responses", json={"input": "hi"})

    assert response.status_code == status_code
    assert seen == ["Bearer access-old"]
    assert manager.calls == [{}]


def test_streaming_request_hides_initial_401_and_replays_before_body_exposure():
    manager = _StubSessionManager()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        if len(seen) == 1:
            return httpx.Response(401, content=b"rejected-stream-body")
        return httpx.Response(200, content=b"data: response-event\n\n")

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        auth=GrokOAuthAuth(manager, model=DEFAULT_GROK_MODEL),  # type: ignore[arg-type]
    ) as client:
        with client.stream(
            "POST", "https://example.test/responses", json={"stream": True}
        ) as response:
            exposed_body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert exposed_body == b"data: response-event\n\n"
    assert b"rejected" not in exposed_body
    assert seen == ["Bearer access-old", "Bearer access-new"]


def test_async_auth_injects_headers_and_retries_exactly_one_401():
    manager = _StubSessionManager()
    seen: list[str] = []

    async def run() -> httpx.Response:
        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers["Authorization"])
            assert request.headers["X-XAI-Token-Auth"] == "xai-grok-cli"
            assert request.headers["x-grok-model-override"] == DEFAULT_GROK_MODEL
            assert (
                request.headers[GROK_CLI_COMPAT_VERSION_HEADER]
                == GROK_CLI_COMPAT_VERSION
            )
            return httpx.Response(401 if len(seen) == 1 else 200, json={})

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            auth=GrokOAuthAuth(manager, model=DEFAULT_GROK_MODEL),  # type: ignore[arg-type]
        ) as client:
            return await client.post(
                "https://example.test/responses", json={"input": "hi"}
            )

    response = asyncio.run(run())
    assert response.status_code == 200
    assert seen == ["Bearer access-old", "Bearer access-new"]
    assert manager.calls == [
        {},
        {"force_refresh": True, "rejected_access_token": "access-old"},
    ]


def test_oauth_adapter_sync_chat_serializes_image_and_tool_on_sanitized_wire(
    tmp_path: Path,
):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=_responses_payload(
                output=[
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "I will tap.",
                                "annotations": [],
                            }
                        ],
                    },
                    {
                        "id": "call_item",
                        "call_id": "call_test",
                        "type": "function_call",
                        "name": "tap",
                        "arguments": '{"x":1}',
                        "status": "completed",
                    },
                ]
            ),
        )

    llm = GrokOAuth(
        oauth_credential_path=str(tmp_path / "auth.json"),
        oauth_access_token="adapter-access",
        oauth_refresh_token="adapter-refresh",
        oauth_expires_at_ms=int(time.time() * 1000) + 3_600_000,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        response = llm.chat(
            [
                ChatMessage(
                    role=MessageRole.USER,
                    blocks=[
                        TextBlock(text="inspect this image"),
                        ImageBlock(image=b"\x89PNG\r\n\x1a\n"),
                    ],
                )
            ],
            tools=[
                {
                    "type": "function",
                    "name": "tap",
                    "description": "Tap a coordinate",
                    "parameters": {"type": "object"},
                }
            ],
            tool_choice={"type": "function", "name": "tap"},
            temperature=0.2,
            top_p=0.7,
            presence_penalty=0.3,
            frequency_penalty=0.4,
            stop="done",
            reasoning={"effort": "high"},
            store=True,
            model="caller-selected-model",
            extra_body={
                "model": "extra-body-model",
                "store": True,
                "temperature": 0.9,
                "top_p": 0.8,
                "presence_penalty": 0.7,
                "frequency_penalty": 0.6,
                "stop": "extra-stop",
                "reasoning": {"effort": "low"},
                "metadata": {"safe": "value"},
            },
        )
    finally:
        llm._client.close()
        asyncio.run(llm._aclient.close())

    assert response.message.content == "I will tap."
    tool_call = next(
        block for block in response.message.blocks if isinstance(block, ToolCallBlock)
    )
    assert tool_call.tool_name == "tap"
    assert json.loads(tool_call.tool_kwargs) == {"x": 1}
    assert response.additional_kwargs["usage"].total_tokens == 11

    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == f"{DEFAULT_GROK_OAUTH_PROXY}/responses"
    assert request.headers["Authorization"] == "Bearer adapter-access"
    assert request.headers["X-XAI-Token-Auth"] == "xai-grok-cli"
    assert request.headers["x-grok-model-override"] == DEFAULT_GROK_MODEL
    assert request.headers[GROK_CLI_COMPAT_VERSION_HEADER] == GROK_CLI_COMPAT_VERSION
    payload = json.loads(request.content)
    assert payload["model"] == DEFAULT_GROK_MODEL
    assert payload["store"] is False
    assert payload["metadata"] == {"safe": "value"}
    assert payload["stream"] is False
    assert payload["tool_choice"] == {"type": "function", "name": "tap"}
    assert payload["tools"][0]["name"] == "tap"
    content = payload["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "inspect this image"}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    for unsupported in (
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "reasoning",
    ):
        assert unsupported not in payload


def test_oauth_adapter_sync_and_async_stream_emit_completed_usage(
    tmp_path: Path,
):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=_responses_sse(),
        )

    llm = GrokOAuth(
        oauth_credential_path=str(tmp_path / "auth.json"),
        oauth_access_token="stream-access",
        oauth_refresh_token="stream-refresh",
        oauth_expires_at_ms=int(time.time() * 1000) + 3_600_000,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        async_http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    async def collect_async():  # type: ignore[no-untyped-def]
        stream = await llm.astream_chat(
            [ChatMessage(role=MessageRole.USER, content="hello")],
            temperature=0.3,
            top_p=0.6,
            reasoning={"effort": "low"},
            store=True,
        )
        return [response async for response in stream]

    try:
        sync_responses = list(
            llm.stream_chat(
                [ChatMessage(role=MessageRole.USER, content="hello")],
                temperature=0.3,
                top_p=0.6,
                reasoning={"effort": "low"},
                store=True,
            )
        )
        async_responses = asyncio.run(collect_async())
    finally:
        llm._client.close()
        asyncio.run(llm._aclient.close())

    for responses in (sync_responses, async_responses):
        assert [response.delta for response in responses] == ["hel", ""]
        assert responses[-1].message.content == "hello"
        assert responses[-1].raw.type == "response.completed"
        assert responses[-1].additional_kwargs["usage"].total_tokens == 11
        usage = get_usage_from_response("GrokOAuth", responses[-1])
        assert (usage.request_tokens, usage.response_tokens, usage.total_tokens) == (
            8,
            3,
            11,
        )

    assert len(requests) == 2
    for request in requests:
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["store"] is False
        assert request.headers["Authorization"] == "Bearer stream-access"
        assert request.headers["X-XAI-Token-Auth"] == "xai-grok-cli"
        assert request.headers["x-grok-model-override"] == DEFAULT_GROK_MODEL
        assert (
            request.headers[GROK_CLI_COMPAT_VERSION_HEADER] == GROK_CLI_COMPAT_VERSION
        )
        assert {"temperature", "top_p", "reasoning"}.isdisjoint(payload)


@pytest.mark.parametrize("model", ("grok-4.7", "grok-4.6", "grok-4.5"))
def test_oauth_adapter_async_responses_request_uses_exact_model_in_header_and_body(
    tmp_path: Path, model: str
):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "created_at": int(time.time()),
                "model": model,
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "status": "completed",
            },
        )

    async def run() -> str:
        async_http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        llm = GrokOAuth(
            model=model,
            oauth_credential_path=str(tmp_path / "auth.json"),
            oauth_access_token="adapter-access",
            oauth_refresh_token="adapter-refresh",
            oauth_expires_at_ms=int(time.time() * 1000) + 3_600_000,
            async_http_client=async_http_client,
        )
        try:
            response = await llm._aclient.responses.create(
                model=model,
                input="hello",
                store=False,
            )
            return response.id
        finally:
            await llm._aclient.close()

    assert asyncio.run(run()) == "resp_test"
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == f"{DEFAULT_GROK_OAUTH_PROXY}/responses"
    assert request.headers["Authorization"] == "Bearer adapter-access"
    assert request.headers["X-XAI-Token-Auth"] == "xai-grok-cli"
    assert request.headers["x-grok-model-override"] == model
    assert request.headers[GROK_CLI_COMPAT_VERSION_HEADER] == GROK_CLI_COMPAT_VERSION
    assert json.loads(request.content)["model"] == model


def test_oauth_rejects_grok_build_latest_without_rewriting_it(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="grok-build-latest"):
        GrokOAuth(
            model="grok-build-latest",
            oauth_credential_path=str(tmp_path / "auth.json"),
        )


def test_oauth_responses_adapter_pins_proxy_and_omits_controls(tmp_path: Path):
    llm = GrokOAuth(
        oauth_credential_path=str(tmp_path / "auth.json"),
        oauth_access_token="access",
        oauth_refresh_token="refresh",
        oauth_expires_at_ms=int(time.time() * 1000) + 3_600_000,
        store=True,
        track_previous_responses=True,
        reasoning_options={"effort": "low"},
        context_window=123,
        default_headers={GROK_CLI_COMPAT_VERSION_HEADER: "caller-version"},
        additional_kwargs={"presence_penalty": 0.5, "stop": ["done"]},
    )

    model_kwargs = llm._get_model_kwargs(
        model="caller-selected-model",
        top_p=0.5,
        temperature=0.2,
        frequency_penalty=0.3,
        reasoning={"effort": "high"},
        extra_body={
            "model": "extra-body-model",
            "store": True,
            "temperature": 0.9,
            "top_p": 0.8,
            "presence_penalty": 0.7,
            "frequency_penalty": 0.6,
            "stop": "extra-stop",
            "reasoning": {"effort": "low"},
            "metadata": {"safe": "value"},
        },
    )
    assert llm.api_base == DEFAULT_GROK_OAUTH_PROXY
    assert llm.metadata.context_window == DEFAULT_GROK_CONTEXT_WINDOW
    assert llm.metadata.is_function_calling_model
    assert llm._tokenizer is None
    assert (
        llm.default_headers[GROK_CLI_COMPAT_VERSION_HEADER] == GROK_CLI_COMPAT_VERSION
    )
    assert model_kwargs["store"] is False
    assert model_kwargs["model"] == DEFAULT_GROK_MODEL
    assert model_kwargs["extra_body"] == {"metadata": {"safe": "value"}}
    for key in (
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "reasoning",
    ):
        assert key not in model_kwargs

    with pytest.raises(ValueError, match="not supported with XAI OAuth"):
        GrokOAuth(
            model="other-model",
            oauth_credential_path=str(tmp_path / "other.json"),
        )

    no_fallback = GrokOAuth(
        api_key="xai-api-key-must-be-ignored",
        oauth_credential_path=str(tmp_path / "missing.json"),
    )
    assert no_fallback.api_key == "oauth"
    with pytest.raises(ValueError, match="No XAI OAuth credentials"):
        no_fallback._oauth_manager.get_valid_credentials()


def test_structured_predict_sanitizes_runtime_kwargs(monkeypatch, tmp_path: Path):
    llm = GrokOAuth(
        oauth_credential_path=str(tmp_path / "auth.json"),
        oauth_access_token="access",
        oauth_refresh_token="refresh",
        oauth_expires_at_ms=int(time.time() * 1000) + 3_600_000,
    )
    captured: dict[str, object] = {}

    def fake_parse(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return SimpleNamespace(output_parsed="parsed")

    monkeypatch.setattr(
        "mobilerun.agent.utils.oauth.grok_oauth_llm.to_openai_message_dicts",
        lambda messages, **kwargs: [{"role": "user", "content": "formatted"}],
    )
    llm._client = SimpleNamespace(responses=SimpleNamespace(parse=fake_parse))
    prompt = SimpleNamespace(format_messages=lambda **kwargs: ["message"])
    result = llm.structured_predict(
        object,
        prompt,
        llm_kwargs={
            "temperature": 0.2,
            "top_p": 0.5,
            "presence_penalty": 0.1,
            "reasoning": {"effort": "low"},
            "store": True,
            "tool_choice": "none",
            "metadata": {"safe": "value"},
            "model": "caller-selected-model",
            "extra_body": {
                "model": "extra-body-model",
                "store": True,
                "tool_choice": "required",
                "reasoning": {"effort": "low"},
                "temperature": 0.9,
                "metadata": {"nested": "safe"},
            },
        },
    )

    assert result == "parsed"
    assert captured == {
        "model": DEFAULT_GROK_MODEL,
        "input": [{"role": "user", "content": "formatted"}],
        "text_format": object,
        "store": False,
        "metadata": {"safe": "value"},
        "extra_body": {"metadata": {"nested": "safe"}},
    }
    assert "tool_choice" not in captured
    assert llm.store is False


def test_async_structured_predict_sanitizes_runtime_kwargs(monkeypatch, tmp_path: Path):
    llm = GrokOAuth(
        oauth_credential_path=str(tmp_path / "auth.json"),
        oauth_access_token="access",
        oauth_refresh_token="refresh",
        oauth_expires_at_ms=int(time.time() * 1000) + 3_600_000,
    )
    captured: dict[str, object] = {}

    async def fake_parse(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return SimpleNamespace(output_parsed="parsed")

    monkeypatch.setattr(
        "mobilerun.agent.utils.oauth.grok_oauth_llm.to_openai_message_dicts",
        lambda messages, **kwargs: [{"role": "user", "content": "formatted"}],
    )
    llm._aclient = SimpleNamespace(responses=SimpleNamespace(parse=fake_parse))
    prompt = SimpleNamespace(format_messages=lambda **kwargs: ["message"])
    result = asyncio.run(
        llm.astructured_predict(
            object,
            prompt,
            llm_kwargs={
                "temperature": 0.2,
                "top_p": 0.5,
                "reasoning": {"effort": "low"},
                "store": True,
                "tool_choice": "required",
                "metadata": {"safe": "value"},
                "model": "caller-selected-model",
                "extra_body": {
                    "model": "extra-body-model",
                    "store": True,
                    "tool_choice": "required",
                    "reasoning": {"effort": "low"},
                    "top_p": 0.8,
                    "metadata": {"nested": "safe"},
                },
            },
        )
    )

    assert result == "parsed"
    assert captured == {
        "model": DEFAULT_GROK_MODEL,
        "input": [{"role": "user", "content": "formatted"}],
        "text_format": object,
        "store": False,
        "metadata": {"safe": "value"},
        "extra_body": {"metadata": {"nested": "safe"}},
    }
    assert "tool_choice" not in captured


def test_grok_integration_source_does_not_bridge_to_external_credentials():
    source = Path("mobilerun/agent/utils/oauth/grok_oauth_llm.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "." + "grok" + "/" + "auth.json",
        "GROK" + "_HOME",
        "subprocess" + ".run",
    )
    assert not any(value in source for value in forbidden)


@pytest.mark.parametrize("model", ("grok-4.5", "grok-4.5-latest", "xai/grok-4.5"))
def test_oauth_saved_grok_4_5_profiles_still_load(model: str, tmp_path) -> None:
    llm = GrokOAuth(
        model=model,
        oauth_access_token="stub",
        credential_path=str(tmp_path / "auth-profiles.json"),
    )
    try:
        assert llm.model == "grok-4.5"
    finally:
        llm._client.close()
        asyncio.run(llm._aclient.close())


def test_oauth_rejects_unknown_grok_models(tmp_path) -> None:
    with pytest.raises(ValueError, match="not supported with XAI OAuth"):
        GrokOAuth(
            model="grok-4.7-build-fast",
            oauth_access_token="stub",
            credential_path=str(tmp_path / "auth-profiles.json"),
        )
