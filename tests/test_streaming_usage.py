import asyncio

import anthropic
from anthropic.types import (
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    TextBlock,
    TextDelta,
    Usage,
)
from llama_index.core.base.llms.types import ChatMessage

from mobilerun.agent.usage import get_usage_from_response, track_usage
from mobilerun.agent.utils.inference import acall_with_retries
from mobilerun.agent.utils.llm_picker import load_llm


def _events():
    return [
        RawMessageStartEvent.model_construct(
            type="message_start",
            message=anthropic.types.Message.model_construct(
                id="msg_1",
                type="message",
                role="assistant",
                model="claude-sonnet-5",
                content=[],
                usage=Usage(input_tokens=120, output_tokens=1),
            ),
        ),
        RawContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=TextBlock(type="text", text=""),
        ),
        RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=TextDelta(type="text_delta", text="Hello"),
        ),
        RawContentBlockStopEvent(type="content_block_stop", index=0),
        RawMessageDeltaEvent.model_construct(
            type="message_delta",
            delta={"stop_reason": "end_turn"},
            usage=MessageDeltaUsage(output_tokens=42),
        ),
        RawMessageStopEvent(type="message_stop"),
    ]


class _AsyncStream:
    def __init__(self, events):
        self._events = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None


def _anthropic_llm(monkeypatch):
    async def acreate(self, **kwargs):
        assert kwargs["stream"] is True
        return _AsyncStream(_events())

    def create(self, **kwargs):
        assert kwargs["stream"] is True
        return iter(_events())

    monkeypatch.setattr(anthropic.resources.messages.AsyncMessages, "create", acreate)
    monkeypatch.setattr(anthropic.resources.messages.Messages, "create", create)
    return load_llm("Anthropic", model="claude-sonnet-5", api_key="stub")


def _assert_usage(llm, response):
    usage = get_usage_from_response(llm.class_name(), response)
    assert (usage.request_tokens, usage.response_tokens, usage.total_tokens) == (
        120,
        42,
        162,
    )


def test_anthropic_stream_reports_usage(monkeypatch):
    llm = _anthropic_llm(monkeypatch)

    response = asyncio.run(
        acall_with_retries(llm, [ChatMessage(role="user", content="hi")], stream=True)
    )

    assert response.message.content == "Hello"
    _assert_usage(llm, response)


def test_anthropic_sync_stream_reports_usage(monkeypatch):
    llm = _anthropic_llm(monkeypatch)

    chunks = list(llm.stream_chat([ChatMessage(role="user", content="hi")]))

    assert "".join(c.delta for c in chunks) == "Hello"
    assert chunks[-1].message.content == "Hello"
    _assert_usage(llm, chunks[-1])


def test_anthropic_stream_usage_reaches_callbacks(monkeypatch):
    llm = _anthropic_llm(monkeypatch)
    tracker = track_usage(llm)
    messages = [ChatMessage(role="user", content="hi")]

    async def consume():
        async for _ in await llm.astream_chat(messages):
            pass

    asyncio.run(consume())
    list(llm.stream_chat(messages))

    assert tracker.usage.model_dump() == {
        "request_tokens": 240,
        "response_tokens": 84,
        "total_tokens": 324,
        "requests": 2,
    }


def test_minimax_requests_usage_only_for_streams():
    llm = load_llm("MiniMax", model="MiniMax-M2.7", api_key="stub")

    assert llm._get_model_kwargs(stream=True)["stream_options"] == {
        "include_usage": True
    }
    assert "stream_options" not in llm._get_model_kwargs()
