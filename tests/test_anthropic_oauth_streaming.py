import asyncio
import copy
import gc
import io
import json

import pytest
import requests
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from mobilerun.agent.usage import get_usage_from_response
from mobilerun.agent.utils.errors import http_status_code
from mobilerun.agent.utils.inference import (
    _is_permanent_http_client_error,
    _stream_response,
    acall_with_retries,
    acomplete_with_retries,
)
from mobilerun.agent.utils.oauth.anthropic_oauth_llm import (
    DEFAULT_CC_VERSION,
    DEFAULT_USER_AGENT,
    AnthropicOAuthLLM,
    AnthropicOAuthStreamError,
)

API_URL = "https://api.anthropic.com/v1/messages"
_REASONS = {400: "Bad Request", 429: "Too Many Requests", 500: "Internal Server Error"}


@pytest.fixture(autouse=True)
def _no_env_token(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_OAUTH_TOKEN", raising=False)


class _Response(requests.Response):
    def __init__(self, body: bytes, status_code: int = 200):
        super().__init__()
        self.status_code = status_code
        self.reason = _REASONS.get(status_code, "OK")
        self.raw = io.BytesIO(body)
        self.url = API_URL
        self.closed = False
        self.lines_read = False

    def iter_lines(self, *args, **kwargs):
        self.lines_read = True
        return super().iter_lines(*args, **kwargs)

    def close(self):
        self.closed = True
        super().close()


class _Session:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **copy.deepcopy(kwargs)})
        return self.responses.pop(0)


def _sse(*events) -> bytes:
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        for event in events
    ).encode("utf-8")


def _message_start(input_tokens=25):
    return {
        "type": "message_start",
        "message": {
            "id": "msg_stream",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 10,
                "output_tokens": 1,
            },
        },
    }


def _block_start(index, block):
    return {"type": "content_block_start", "index": index, "content_block": block}


def _delta(index, **delta):
    return {"type": "content_block_delta", "index": index, "delta": delta}


def _block_stop(index):
    return {"type": "content_block_stop", "index": index}


def _message_delta(output_tokens=42, stop_reason="end_turn"):
    return {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }


_PING = {"type": "ping"}
_STOP = {"type": "message_stop"}


def _text_stream(*pieces) -> bytes:
    return _sse(
        _message_start(),
        _PING,
        _block_start(0, {"type": "text", "text": ""}),
        *(_delta(0, type="text_delta", text=piece) for piece in pieces),
        _block_stop(0),
        _message_delta(),
        _STOP,
    )


def _llm(*responses, **kwargs):
    llm = AnthropicOAuthLLM(access_token="test-token", credential_path=None, **kwargs)
    llm._session = _Session(*responses)
    return llm


def _user(text="hello"):
    return [ChatMessage(role=MessageRole.USER, content=text)]


def test_chat_request_shape_is_pinned():
    llm = _llm(
        _Response(json.dumps({"content": [{"type": "text", "text": "ok"}]}).encode()),
        model="claude-sonnet-4-6",
        temperature=0.3,
        additional_kwargs={"top_k": 5},
    )

    llm.chat(
        [
            ChatMessage(role=MessageRole.SYSTEM, content="be terse"),
            ChatMessage(role=MessageRole.USER, content="hello"),
        ],
        stop_sequences=["END"],
        formatted=True,
    )

    assert llm._session.calls == [
        {
            "url": API_URL,
            "headers": {
                "Authorization": "Bearer test-token",
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "oauth-2025-04-20",
                "User-Agent": DEFAULT_USER_AGENT,
            },
            "json": {
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.3,
                "max_tokens": 8192,
                "system": [
                    {
                        "type": "text",
                        "text": "x-anthropic-billing-header: "
                        f"cc_version={DEFAULT_CC_VERSION}; cc_entrypoint=cli; cch=00000;",
                    },
                    {"type": "text", "text": "be terse"},
                ],
                "top_k": 5,
                "stop_sequences": ["END"],
            },
            "timeout": 60.0,
        }
    ]


def test_stream_request_matches_chat_request_plus_stream_flag():
    kwargs = {
        "model": "claude-sonnet-4-6",
        "temperature": 0.3,
        "additional_kwargs": {"top_k": 5},
    }
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="be terse"),
        ChatMessage(role=MessageRole.USER, content="hello"),
    ]
    chat_llm = _llm(
        _Response(json.dumps({"content": [{"type": "text", "text": "ok"}]}).encode()),
        **kwargs,
    )
    stream_llm = _llm(_Response(_text_stream("ok")), **kwargs)

    chat_llm.chat(messages, stop_sequences=["END"], formatted=True)
    list(stream_llm.stream_chat(messages, stop_sequences=["END"], formatted=True))

    chat_call = chat_llm._session.calls[0]
    stream_call = stream_llm._session.calls[0]
    assert "stream" not in chat_call and "stream" not in chat_call["json"]
    assert stream_call.pop("stream") is True
    assert stream_call["json"].pop("stream") is True
    assert stream_call == chat_call


def test_text_stream_yields_deltas_and_running_content():
    llm = _llm(_Response(_text_stream("Hel", "lo", " world")))

    chunks = list(llm.stream_chat(_user()))

    assert [chunk.delta for chunk in chunks] == ["Hel", "lo", " world", ""]
    assert [chunk.message.content for chunk in chunks] == [
        "Hel",
        "Hello",
        "Hello world",
        "Hello world",
    ]
    final = chunks[-1]
    assert final.message.role == MessageRole.ASSISTANT
    assert final.message.additional_kwargs["content_blocks"] == [
        {"type": "text", "text": "Hello world"}
    ]


def test_thinking_signature_and_text_rebuild_the_chat_response():
    stream = _sse(
        _message_start(),
        _block_start(0, {"type": "thinking", "thinking": "", "signature": ""}),
        _delta(0, type="thinking_delta", thinking="Let me "),
        _delta(0, type="thinking_delta", thinking="think."),
        _delta(0, type="signature_delta", signature="sig-123"),
        _block_stop(0),
        _block_start(1, {"type": "redacted_thinking", "data": "opaque"}),
        _block_stop(1),
        _block_start(2, {"type": "text", "text": ""}),
        _delta(2, type="text_delta", text="Done"),
        _block_stop(2),
        _message_delta(),
        _STOP,
    )
    final = list(_llm(_Response(stream)).stream_chat(_user()))[-1]

    expected_message = {
        "id": "msg_stream",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [
            {"type": "thinking", "thinking": "Let me think.", "signature": "sig-123"},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "text", "text": "Done"},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 25,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 10,
            "output_tokens": 42,
        },
    }
    from_chat = _llm(_Response(json.dumps(expected_message).encode())).chat(_user())

    assert final.raw == expected_message
    assert final.message.content == from_chat.message.content == "Done"
    assert final.message.additional_kwargs == from_chat.message.additional_kwargs
    assert final.additional_kwargs == from_chat.additional_kwargs


def test_final_chunk_carries_id_merged_usage_and_stop_reason():
    final = list(_llm(_Response(_text_stream("ok"))).stream_chat(_user()))[-1]

    assert final.additional_kwargs == {
        "id": "msg_stream",
        "usage": {
            "input_tokens": 25,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 10,
            "output_tokens": 42,
        },
        "stop_reason": "end_turn",
    }
    usage = get_usage_from_response("AnthropicOAuthLLM", final)
    assert (usage.request_tokens, usage.response_tokens, usage.total_tokens) == (
        25,
        42,
        67,
    )


def test_deltas_concatenate_to_final_text_across_text_blocks():
    long_unicode = "naïve ✓ " * 200
    stream = _sse(
        _message_start(),
        _block_start(0, {"type": "text", "text": ""}),
        _delta(0, type="text_delta", text="first"),
        _block_stop(0),
        _block_start(1, {"type": "text", "text": ""}),
        _delta(1, type="text_delta", text=""),
        _delta(1, type="text_delta", text=long_unicode),
        _delta(1, type="text_delta", text="end"),
        _block_stop(1),
        _message_delta(),
        _STOP,
    )

    chunks = list(_llm(_Response(stream)).stream_chat(_user()))

    assert "".join(chunk.delta for chunk in chunks) == chunks[-1].message.content
    assert chunks[-1].message.content == f"first\n{long_unicode}end"


def test_tool_input_json_deltas_are_parsed_into_the_block():
    stream = _sse(
        _message_start(),
        _block_start(0, {"type": "tool_use", "id": "t1", "name": "tap", "input": {}}),
        _delta(0, type="input_json_delta", partial_json='{"x": 1'),
        _delta(0, type="input_json_delta", partial_json=', "y": 2}'),
        _block_stop(0),
        _message_delta(stop_reason="tool_use"),
        _STOP,
    )

    final = list(_llm(_Response(stream)).stream_chat(_user()))[-1]

    assert final.message.additional_kwargs["content_blocks"] == [
        {"type": "tool_use", "id": "t1", "name": "tap", "input": {"x": 1, "y": 2}}
    ]
    assert final.additional_kwargs["stop_reason"] == "tool_use"


@pytest.mark.parametrize(
    ("error_type", "status_code", "permanent"),
    [
        ("overloaded_error", 529, False),
        ("api_error", 500, False),
        ("rate_limit_error", 429, False),
        ("invalid_request_error", 400, True),
    ],
)
def test_error_event_raises_with_equivalent_http_status(
    error_type, status_code, permanent
):
    response = _Response(
        _sse(
            _message_start(),
            _block_start(0, {"type": "text", "text": ""}),
            _delta(0, type="text_delta", text="partial"),
            {"type": "error", "error": {"type": error_type, "message": "boom"}},
        )
    )
    stream = _llm(response).stream_chat(_user())

    assert next(stream).delta == "partial"
    with pytest.raises(AnthropicOAuthStreamError) as exc_info:
        next(stream)

    error = exc_info.value
    assert isinstance(error, requests.HTTPError)
    assert error.error_type == error_type
    assert http_status_code(error) == status_code
    assert _is_permanent_http_client_error(error) is permanent
    assert "boom" in str(error)
    assert response.closed


@pytest.mark.parametrize(
    ("error_type", "expected_calls"),
    [("overloaded_error", 3), ("invalid_request_error", 1)],
)
def test_stream_error_events_follow_http_retry_policy(error_type, expected_calls):
    error_stream = _sse(
        _message_start(),
        {"type": "error", "error": {"type": error_type, "message": "boom"}},
    )
    llm = _llm(*(_Response(error_stream) for _ in range(3)))

    with pytest.raises(AnthropicOAuthStreamError):
        asyncio.run(acall_with_retries(llm, _user(), retries=3, delay=0, stream=True))

    assert len(llm._session.calls) == expected_calls


@pytest.mark.parametrize("status_code", [400, 429, 500])
def test_http_error_raises_before_reading_the_stream(status_code):
    body = json.dumps({"type": "error", "error": {"type": "x", "message": "nope"}})
    chat_llm = _llm(_Response(body.encode(), status_code=status_code))
    response = _Response(body.encode(), status_code=status_code)
    stream_llm = _llm(response)

    with pytest.raises(requests.HTTPError) as chat_error:
        chat_llm.chat(_user())
    with pytest.raises(requests.HTTPError) as stream_error:
        stream_llm.stream_chat(_user())

    assert type(stream_error.value) is requests.HTTPError
    assert str(stream_error.value) == str(chat_error.value)
    assert stream_error.value.response is response
    assert response.text == body
    assert http_status_code(stream_error.value) == status_code
    assert not response.lines_read
    assert response.closed


def test_stream_without_message_stop_raises_retryable_error():
    response = _Response(
        _sse(
            _message_start(),
            _block_start(0, {"type": "text", "text": ""}),
            _delta(0, type="text_delta", text="cut"),
        )
    )

    with pytest.raises(AnthropicOAuthStreamError) as exc_info:
        list(_llm(response).stream_chat(_user()))

    assert not _is_permanent_http_client_error(exc_info.value)
    assert response.closed


def test_response_is_closed_when_the_consumer_stops_early():
    response = _Response(_text_stream("a", "b", "c"))
    stream = _llm(response).stream_chat(_user())

    next(stream)
    assert not response.closed
    stream.close()

    assert response.closed


def test_response_is_closed_when_the_stream_is_dropped_unread():
    response = _Response(_text_stream("ok"))
    stream = _llm(response).stream_chat(_user())

    assert not response.closed
    del stream
    gc.collect()

    assert response.closed
    assert not response.lines_read


def test_response_is_closed_after_a_full_stream():
    response = _Response(_text_stream("ok"))

    list(_llm(response).stream_chat(_user()))

    assert response.closed


def test_stream_complete_yields_completion_chunks():
    llm = _llm(_Response(_text_stream("Hi", " there")))

    chunks = list(llm.stream_complete("hello", formatted=True))

    assert [chunk.delta for chunk in chunks] == ["Hi", " there", ""]
    assert chunks[-1].text == "Hi there"
    assert chunks[-1].additional_kwargs["stop_reason"] == "end_turn"
    assert llm._session.calls[0]["json"]["messages"] == [
        {"role": "user", "content": "hello"}
    ]
    assert llm._session.calls[0]["json"]["stream"] is True


def test_inference_stream_response_returns_the_final_message():
    stream = _sse(
        _message_start(),
        _block_start(0, {"type": "thinking", "thinking": "", "signature": ""}),
        _delta(0, type="signature_delta", signature="sig"),
        _block_stop(0),
        _block_start(1, {"type": "text", "text": ""}),
        _delta(1, type="text_delta", text="<request_accomplished>"),
        _delta(1, type="text_delta", text="done</request_accomplished>"),
        _block_stop(1),
        _message_delta(),
        _STOP,
    )
    llm = _llm(_Response(stream))

    response = asyncio.run(_stream_response(llm, _user(), timeout=5))

    assert response.message.content == (
        "<request_accomplished>done</request_accomplished>"
    )
    assert response.message.additional_kwargs["content_blocks"][0] == {
        "type": "thinking",
        "thinking": "",
        "signature": "sig",
    }
    assert response.additional_kwargs["id"] == "msg_stream"
    assert get_usage_from_response("AnthropicOAuthLLM", response).total_tokens == 67


def test_inference_retries_return_streamed_chat_and_completion():
    chat_llm = _llm(_Response(_text_stream("chat ", "reply")))
    completion_llm = _llm(_Response(_text_stream("com.example.app")))

    chat = asyncio.run(acall_with_retries(chat_llm, _user(), retries=1, stream=True))
    completion = asyncio.run(
        acomplete_with_retries(completion_llm, "open app", retries=1, stream=True)
    )

    assert chat.message.content == "chat reply"
    assert completion.text == "com.example.app"
