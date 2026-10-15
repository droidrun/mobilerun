import asyncio
import json
import logging
import traceback
from collections.abc import Callable

import httpx
import pytest

import mobilerun.agent.jev.client as client_module
from mobilerun.agent.jev.client import (
    DEFAULT_BASE_URL,
    JevClient,
    JevResponse,
    resolve_typesafe_api_key,
)
from mobilerun.agent.jev.errors import (
    JevAuthError,
    JevConfigError,
    JevError,
    JevRateLimitError,
    JevRequestError,
    JevResponseError,
    JevServiceError,
    JevTimeoutError,
    JevTransportError,
)
from mobilerun.config_manager import env_keys

KEY = "tsk_test_jevclient_7Qx9Zr2Lm4Np6Vb8"
REQUEST_ID = "req_0123456789abcdef0123456789abcdef"
ANSWERS = {
    "operation": {
        "type": "choice",
        "choice": "TAP",
        "confidence": 0.8,
        "probabilities": {"TAP": 0.9, "DONE": 0.1},
    }
}
SUCCESS = {
    "model": "jev-1.13.0",
    "answers": ANSWERS,
    "usage": {"input_tokens": 589, "output_tokens": 212},
}
STATE = {"goal": "Open settings", "visibleText": ["Settings"]}
QUESTIONS = {
    "operation": {
        "type": "choice",
        "instructions": {"goal": "Open settings"},
        "criteria": {"TAP": "Tap", "DONE": "Done"},
    }
}

Handler = Callable[[httpx.Request], httpx.Response]


def make_client(handler: Handler, **kwargs) -> JevClient:
    transport = httpx.MockTransport(handler)
    return JevClient(KEY, http_client=httpx.AsyncClient(transport=transport), **kwargs)


def ask(client: JevClient) -> JevResponse:
    return asyncio.run(client.systemone(STATE, QUESTIONS))


def json_response(status: int, body, request_id: str | None = REQUEST_ID, **headers):
    if request_id:
        headers["x-typesafe-request-id"] = request_id
    return httpx.Response(status, json=body, headers=headers)


def sequence(*responses) -> tuple[Handler, list[httpx.Request]]:
    """Serve responses (or raise exception factories) in order, recording requests."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        item = responses[min(len(seen), len(responses)) - 1]
        if callable(item):
            raise item(request)
        return httpx.Response(
            item.status_code, headers=item.headers, content=item.content
        )

    return handler, seen


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(client_module.asyncio, "sleep", fake_sleep)
    return delays


@pytest.fixture
def mobilerun_caplog(caplog):
    """caplog wired to the non-propagating "mobilerun" logger at DEBUG."""
    logger = logging.getLogger("mobilerun")
    previous = logger.propagate
    logger.propagate = True
    caplog.set_level(logging.DEBUG, logger="mobilerun")
    yield caplog
    logger.propagate = previous


@pytest.fixture
def isolated_keys(monkeypatch, tmp_path):
    path = tmp_path / "auth-profiles.json"
    monkeypatch.setattr(env_keys, "AUTH_PROFILES_PATH", path)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    return path


def test_systemone_request_shape(sleeps) -> None:
    handler, seen = sequence(json_response(200, SUCCESS))
    ask(make_client(handler, model="jev-1.14.0"))

    (request,) = seen
    assert request.method == "POST"
    assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.headers["content-type"] == "application/json"
    assert request.headers["accept"] == "application/json"
    assert request.headers["user-agent"].startswith("mobilerun/")
    assert "x-typesafe-retry-count" not in request.headers
    assert json.loads(request.content) == {
        "model": "jev-1.14.0",
        "state": STATE,
        "questions": QUESTIONS,
    }
    assert sleeps == []


def test_systemone_parses_success() -> None:
    handler, _ = sequence(json_response(200, SUCCESS))
    response = ask(make_client(handler))

    assert response.model == "jev-1.13.0"
    assert response.answers == ANSWERS
    assert response.usage == {"input_tokens": 589, "output_tokens": 212}
    assert response.request_id == REQUEST_ID
    assert isinstance(response.latency_ms, int) and response.latency_ms >= 0


def test_systemone_tolerates_missing_model_and_usage() -> None:
    handler, _ = sequence(json_response(200, {"answers": ANSWERS}, request_id=None))
    response = ask(make_client(handler))

    assert response == JevResponse(answers=ANSWERS, latency_ms=response.latency_ms)


def test_custom_base_url_is_used() -> None:
    handler, seen = sequence(json_response(200, SUCCESS))
    ask(make_client(handler, base_url="http://127.0.0.1:8080/"))

    assert str(seen[0].url) == "http://127.0.0.1:8080/v1/systemone"


def test_lone_surrogate_in_state_does_not_abort() -> None:
    handler, seen = sequence(json_response(200, SUCCESS))
    client = make_client(handler)
    asyncio.run(client.systemone({"text": "broken \ud83d emoji"}, QUESTIONS))

    assert json.loads(seen[0].content)["state"] == {"text": "broken ? emoji"}


def test_nan_in_state_is_rejected_locally() -> None:
    handler, seen = sequence(json_response(200, SUCCESS))
    with pytest.raises(JevRequestError):
        asyncio.run(make_client(handler).systemone({"x": float("nan")}, QUESTIONS))
    assert seen == []


@pytest.mark.parametrize(
    "status, body",
    [
        (
            401,
            {
                "detail": {
                    "error_type": "authentication_error",
                    "message": "Cannot authenticate with the server.",
                }
            },
        ),
        (
            403,
            {
                "detail": {
                    "error_type": "authentication_error",
                    "message": "Must supply an API key! Check your request.",
                }
            },
        ),
    ],
)
def test_auth_errors_fail_fast(status, body, sleeps) -> None:
    handler, seen = sequence(json_response(status, body))
    with pytest.raises(JevAuthError) as info:
        ask(make_client(handler))

    assert len(seen) == 1
    assert sleeps == []
    assert info.value.status == status
    assert info.value.request_id == REQUEST_ID
    assert str(info.value) == (
        f"TypeSafe rejected the API key (HTTP {status}, request {REQUEST_ID}). "
        "Check TYPESAFE_API_KEY or the key saved with 'mobilerun configure jev'."
    )


def test_403_access_denied_keeps_detail(sleeps) -> None:
    handler, _ = sequence(json_response(403, {"error": "Model not enabled for org"}))
    with pytest.raises(JevAuthError) as info:
        ask(make_client(handler))

    assert str(info.value) == (
        f"TypeSafe denied access (HTTP 403, request {REQUEST_ID}): "
        "Model not enabled for org. "
        "Check TYPESAFE_API_KEY or the key saved with 'mobilerun configure jev'."
    )


def test_422_detail_is_extracted_without_echoing_input(sleeps) -> None:
    body = {
        "detail": [
            {
                "type": "missing",
                "loc": ["body", "questions", "operation", "criteria"],
                "msg": "Field required",
                "input": {"goal": "PRIVATE_SCREEN_TEXT"},
                "ctx": {"hint": "PRIVATE_CTX"},
            },
            {"type": "json_invalid", "loc": ["body", 1], "msg": "JSON decode error"},
            {"loc": ["body"], "input": "PRIVATE_NO_MSG"},
        ]
    }
    handler, seen = sequence(json_response(422, body))
    with pytest.raises(JevRequestError) as info:
        ask(make_client(handler))

    message = str(info.value)
    assert len(seen) == 1 and sleeps == []
    assert info.value.status == 422
    assert message == (
        f"TypeSafe rejected the request (HTTP 422, request {REQUEST_ID}): "
        "questions.operation.criteria: Field required; 1: JSON decode error"
    )
    assert "PRIVATE" not in message


@pytest.mark.parametrize(
    "body, detail",
    [
        ({"error": "max_tokens_exceeded"}, "max_tokens_exceeded"),
        ({"error": {"message": "unknown model"}}, "unknown model"),
        ({"message": "bad request"}, "bad request"),
        ({"detail": "Not Found"}, "Not Found"),
        ({"detail": {"message": "too big"}}, "too big"),
    ],
)
def test_request_error_detail_order(body, detail, sleeps) -> None:
    handler, _ = sequence(json_response(400, body, request_id=None))
    with pytest.raises(JevRequestError) as info:
        ask(make_client(handler))

    assert str(info.value) == f"TypeSafe rejected the request (HTTP 400): {detail}"
    assert info.value.request_id is None


def test_request_error_detail_is_truncated(sleeps) -> None:
    handler, _ = sequence(json_response(400, {"error": "x" * 1000}))
    with pytest.raises(JevRequestError) as info:
        ask(make_client(handler))

    detail = str(info.value).split(": ", 1)[1]
    assert len(detail) == 300 and detail.endswith("…")


def test_html_error_page_is_not_echoed(sleeps) -> None:
    page = "<html><body>" + "noise " * 200 + "</body></html>"
    handler, _ = sequence(httpx.Response(404, text=page))
    with pytest.raises(JevRequestError) as info:
        ask(make_client(handler))

    assert str(info.value) == "TypeSafe rejected the request (HTTP 404)."


@pytest.mark.parametrize(
    "headers, expected",
    [
        ({"retry-after-ms": "250"}, 0.25),
        ({"Retry-After": "3"}, 3.0),
        ({"retry-after-ms": "1500", "Retry-After": "9"}, 1.5),
        ({"Retry-After": "60"}, 10.0),
        ({"Retry-After": "soon"}, 0.5),
        ({}, 0.5),
    ],
)
def test_429_is_retried_with_retry_after(headers, expected, sleeps) -> None:
    handler, seen = sequence(
        json_response(429, {"error": "rate_limited"}, **headers),
        json_response(200, SUCCESS),
    )
    response = ask(make_client(handler))

    assert response.answers == ANSWERS
    assert len(seen) == 2
    assert sleeps == [expected]
    assert seen[1].headers["x-typesafe-retry-count"] == "1"
    assert seen[1].headers["authorization"] == f"Bearer {KEY}"


def test_429_exhausted_raises_rate_limit_error(sleeps) -> None:
    handler, seen = sequence(json_response(429, {"error": "rate_limited"}))
    with pytest.raises(JevRateLimitError) as info:
        ask(make_client(handler))

    assert len(seen) == 3
    assert sleeps == [0.5, 1.0]
    assert info.value.status == 429
    assert str(info.value) == (
        f"TypeSafe rate limit reached (HTTP 429, request {REQUEST_ID}) after 3 attempts."
    )


def test_529_exhausted_raises_service_error(sleeps, mobilerun_caplog) -> None:
    handler, seen = sequence(json_response(529, {"error": "overloaded"}))
    with pytest.raises(JevServiceError) as info:
        ask(make_client(handler, max_retries=3, retry_delay=2.0))

    assert len(seen) == 4
    assert sleeps == [2.0, 4.0, 5.0]
    assert info.value.status == 529
    assert info.value.request_id == REQUEST_ID
    assert "after 4 attempts" in str(info.value)
    assert f"Jev request attempt 1/4 failed: HTTP 529 (request {REQUEST_ID})" in (
        mobilerun_caplog.text
    )


def test_500_then_success_is_retried(sleeps) -> None:
    handler, seen = sequence(
        httpx.Response(500, text="oops"), json_response(200, SUCCESS)
    )
    assert ask(make_client(handler)).model == "jev-1.13.0"
    assert len(seen) == 2 and sleeps == [0.5]


def test_max_retries_zero_makes_one_attempt(sleeps) -> None:
    handler, seen = sequence(json_response(503, {}))
    with pytest.raises(JevServiceError, match=r"after 1 attempt\."):
        ask(make_client(handler, max_retries=0))
    assert len(seen) == 1 and sleeps == []


def test_timeouts_exhausted_raise_timeout_error(sleeps) -> None:
    handler, seen = sequence(lambda request: httpx.ReadTimeout("slow", request=request))
    with pytest.raises(JevTimeoutError) as info:
        ask(make_client(handler))

    assert len(seen) == 3
    assert sleeps == [0.5, 1.0]
    assert str(info.value) == (
        "TypeSafe did not respond within 30s (3 attempts). No action was executed."
    )


def test_timeout_then_success_is_retried(sleeps) -> None:
    handler, seen = sequence(
        lambda request: httpx.ConnectTimeout("slow", request=request),
        json_response(200, SUCCESS),
    )
    assert ask(make_client(handler)).answers == ANSWERS
    assert len(seen) == 2


def test_connect_error_raises_transport_error(sleeps) -> None:
    handler, seen = sequence(
        lambda request: httpx.ConnectError("down", request=request)
    )
    with pytest.raises(JevTransportError) as info:
        ask(make_client(handler))

    assert len(seen) == 3
    assert str(info.value) == (
        "Could not reach TypeSafe at https://api.typesafe.ai (ConnectError). "
        "No action was executed."
    )


@pytest.mark.parametrize(
    "response, what",
    [
        (httpx.Response(200, text="not json"), "not valid JSON"),
        (httpx.Response(200, json=["answers"]), "not a JSON object"),
        (httpx.Response(200, json={"model": "jev-latest"}), "'answers'"),
        (httpx.Response(200, json={"answers": []}), "'answers'"),
        (httpx.Response(200, json={"answers": {"operation": "TAP"}}), "'answers'"),
    ],
)
def test_malformed_success_body_raises_response_error(response, what, sleeps) -> None:
    handler, seen = sequence(response)
    with pytest.raises(JevResponseError, match=what) as info:
        ask(make_client(handler))

    assert len(seen) == 1 and sleeps == []
    assert str(info.value).startswith("TypeSafe returned an unexpected response:")


def test_list_models_parses_names() -> None:
    body = {
        "models": [
            {
                "name": "jev-latest",
                "description": "Stable",
                "release_date": "2026-09-01",
            },
            {"name": "jev-preview"},
            {"description": "nameless"},
            "junk",
        ]
    }
    handler, seen = sequence(json_response(200, body))
    assert asyncio.run(make_client(handler).list_models()) == [
        "jev-latest",
        "jev-preview",
    ]
    assert seen[0].method == "GET"
    assert str(seen[0].url) == f"{DEFAULT_BASE_URL}/v1/models"
    assert seen[0].headers["authorization"] == f"Bearer {KEY}"
    assert seen[0].content == b""


def test_list_models_rejects_bad_shape() -> None:
    handler, _ = sequence(json_response(200, {"models": {"name": "jev-latest"}}))
    with pytest.raises(JevResponseError, match="'models'"):
        asyncio.run(make_client(handler).list_models())


def test_list_models_401_raises_auth_error(sleeps) -> None:
    handler, seen = sequence(json_response(401, {"detail": {"message": "bad key"}}))
    with pytest.raises(JevAuthError):
        asyncio.run(make_client(handler).list_models())
    assert len(seen) == 1 and sleeps == []


LEAK_CASES = [
    json_response(401, {"detail": {"message": f"bad key {KEY}"}}),
    json_response(403, {"error": f"denied for {KEY}"}),
    json_response(
        422, {"detail": [{"loc": ["body"], "msg": f"bad {KEY}", "input": KEY}]}
    ),
    json_response(400, {"error": f"Authorization: Bearer {KEY}"}),
    json_response(404, {"detail": "Not Found"}),
    json_response(429, {"error": KEY}),
    json_response(529, {"error": KEY}),
    httpx.Response(502, text=f"upstream said {KEY}"),
    httpx.Response(200, text=f"not json {KEY}"),
    httpx.Response(200, json={"answers": KEY}),
    lambda request: httpx.ReadTimeout(f"timeout {KEY}", request=request),
    lambda request: httpx.ConnectError(f"refused {KEY}", request=request),
]


@pytest.mark.parametrize("call", ["systemone", "list_models"])
@pytest.mark.parametrize("outcome", LEAK_CASES)
def test_key_never_leaks(outcome, call, sleeps, mobilerun_caplog) -> None:
    handler, _ = sequence(outcome)
    client = make_client(handler)

    async def go():
        if call == "systemone":
            return await client.systemone(STATE, QUESTIONS)
        return await client.list_models()

    with pytest.raises(JevError) as info:
        asyncio.run(go())

    error = info.value
    assert KEY not in str(error)
    assert KEY not in repr(error)
    assert KEY not in "".join(traceback.format_exception_only(error))
    assert KEY not in repr(client)
    assert KEY not in mobilerun_caplog.text


def test_server_echoed_key_is_redacted(sleeps) -> None:
    handler, _ = sequence(json_response(400, {"error": f"bad token {KEY}"}))
    with pytest.raises(JevRequestError) as info:
        ask(make_client(handler))
    assert str(info.value).endswith("bad token [redacted]")


@pytest.mark.parametrize("padding", [150, 165])
def test_echoed_key_in_long_text_body_is_redacted_before_truncation(
    padding, sleeps
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        echo = "x" * padding + " Authorization: " + request.headers["authorization"]
        return httpx.Response(400, text=echo)

    with pytest.raises(JevRequestError) as info:
        ask(make_client(handler, max_retries=0))

    message = str(info.value)
    assert "Bearer [redacted]" in message
    assert KEY[:5] not in message


def test_success_debug_log_has_no_body_or_key(mobilerun_caplog) -> None:
    handler, _ = sequence(json_response(200, SUCCESS))
    ask(make_client(handler))

    text = mobilerun_caplog.text
    assert "Jev response: model jev-1.13.0" in text
    assert KEY not in text
    assert "Open settings" not in text


def test_repr_hides_key() -> None:
    client = JevClient(KEY, base_url="https://user:pass@api.typesafe.ai")
    text = repr(client)
    assert KEY not in text and "pass" not in text
    assert text == "JevClient(model='jev-latest', base_url='https://api.typesafe.ai')"
    asyncio.run(client.aclose())


@pytest.mark.parametrize(
    "bad_key",
    ["tsk with space", "tsk\tkey\x00", "tsk_kéy_value", "tsk\nkey"],
)
def test_malformed_key_is_rejected(bad_key) -> None:
    with pytest.raises(JevConfigError, match="malformed") as info:
        JevClient(bad_key)
    assert bad_key not in str(info.value)
    assert bad_key.strip() not in str(info.value)


@pytest.mark.parametrize("empty", ["", "   ", None])
def test_empty_key_is_rejected(empty) -> None:
    with pytest.raises(JevConfigError, match="TYPESAFE_API_KEY"):
        JevClient(empty)


def test_key_is_stripped() -> None:
    handler, seen = sequence(json_response(200, SUCCESS))
    transport = httpx.MockTransport(handler)
    client = JevClient(f"  {KEY}\n", http_client=httpx.AsyncClient(transport=transport))
    ask(client)
    assert seen[0].headers["authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize(
    "base_url", ["http://api.typesafe.ai", "ftp://api.typesafe.ai", "not a url"]
)
def test_insecure_or_invalid_base_url_is_rejected(base_url) -> None:
    with pytest.raises(JevConfigError, match="base_url"):
        JevClient(KEY, base_url=base_url)


@pytest.mark.parametrize(
    "kwargs", [{"model": " "}, {"timeout": 0}, {"max_retries": -1}]
)
def test_invalid_settings_are_rejected(kwargs) -> None:
    with pytest.raises(JevConfigError):
        JevClient(KEY, **kwargs)


def test_aclose_keeps_injected_client_open() -> None:
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: json_response(200, SUCCESS))
    )
    client = JevClient(KEY, http_client=http_client)
    asyncio.run(client.aclose())
    assert not http_client.is_closed


def test_aclose_closes_owned_client() -> None:
    client = JevClient(KEY)

    async def go() -> None:
        async with client:
            pass

    asyncio.run(go())
    assert client._http.is_closed


def test_resolve_key_from_env(isolated_keys, monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", f"  {KEY}  ")
    assert resolve_typesafe_api_key() == KEY
    assert resolve_typesafe_api_key("env") == KEY


def test_resolve_key_from_saved_file(isolated_keys) -> None:
    isolated_keys.write_text(json.dumps({"apiKeys": {"typesafe": KEY}}))
    assert resolve_typesafe_api_key() == KEY
    assert resolve_typesafe_api_key("file") == KEY


def test_resolve_key_respects_source(isolated_keys, monkeypatch) -> None:
    isolated_keys.write_text(json.dumps({"apiKeys": {"typesafe": "saved_key"}}))
    monkeypatch.setenv("TYPESAFE_API_KEY", "shell_key")

    assert resolve_typesafe_api_key("auto") == "saved_key"
    assert resolve_typesafe_api_key("env") == "shell_key"
    assert resolve_typesafe_api_key("file") == "saved_key"


def test_resolve_key_missing(isolated_keys) -> None:
    with pytest.raises(JevConfigError) as info:
        resolve_typesafe_api_key()
    assert str(info.value) == (
        "No TypeSafe API key found. Set TYPESAFE_API_KEY or run "
        "'mobilerun configure jev'."
    )


@pytest.mark.parametrize("source", ["env", "file"])
def test_resolve_key_missing_from_named_source(
    source, isolated_keys, monkeypatch
) -> None:
    if source == "env":
        isolated_keys.write_text(json.dumps({"apiKeys": {"typesafe": "saved_key"}}))
    else:
        monkeypatch.setenv("TYPESAFE_API_KEY", "shell_key")
    with pytest.raises(JevConfigError, match=f"api_key_source is '{source}'"):
        resolve_typesafe_api_key(source)


def test_resolve_key_rejects_unknown_source(isolated_keys) -> None:
    with pytest.raises(JevConfigError, match="api_key_source"):
        resolve_typesafe_api_key("vault")
