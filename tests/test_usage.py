from types import SimpleNamespace

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole

from mobilerun.agent.usage import (
    TokenCountingHandler,
    create_tracker,
    get_usage_from_response,
    track_usage,
)
from mobilerun.agent.utils.llm_picker import load_llm


def _openai_responses_chat_response() -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="ok"),
        raw=SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=3,
                output_tokens=2,
                total_tokens=5,
            )
        ),
    )


def test_track_usage_supports_mobilerun_openai_responses_wrapper() -> None:
    llm = load_llm("OpenAIResponses", model="gpt-5.5", api_key="stub")

    tracker = track_usage(llm)

    assert isinstance(tracker, TokenCountingHandler)
    assert tracker.provider == "OpenAIResponses"


def test_create_tracker_supports_mobilerun_openai_responses_wrapper() -> None:
    llm = load_llm("OpenAIResponses", model="gpt-5.5", api_key="stub")

    tracker = create_tracker(llm)

    assert isinstance(tracker, TokenCountingHandler)
    assert tracker.provider == "OpenAIResponses"


def test_openai_responses_wrapper_name_extracts_usage_from_response() -> None:
    usage = get_usage_from_response(
        "MobilerunOpenAIResponses", _openai_responses_chat_response()
    )

    assert usage.request_tokens == 3
    assert usage.response_tokens == 2
    assert usage.total_tokens == 5
    assert usage.requests == 1


def test_openai_responses_class_name_extracts_usage_from_response() -> None:
    usage = get_usage_from_response(
        "openai_responses_llm", _openai_responses_chat_response()
    )

    assert usage.request_tokens == 3
    assert usage.response_tokens == 2
    assert usage.total_tokens == 5
    assert usage.requests == 1


def _litellm_chat_response() -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="ok"),
        raw=SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            )
        ),
    )


def test_litellm_extracts_usage_from_response() -> None:
    usage = get_usage_from_response("LiteLLM", _litellm_chat_response())

    assert usage.request_tokens == 10
    assert usage.response_tokens == 5
    assert usage.total_tokens == 15
    assert usage.requests == 1


def test_track_usage_supports_litellm() -> None:
    llm = load_llm("LiteLLM", model="openai/gpt-4o-mini", api_key="stub")

    tracker = track_usage(llm)

    assert isinstance(tracker, TokenCountingHandler)
    assert tracker.provider == "LiteLLM"


def test_litellm_usage_with_null_usage_returns_zeros() -> None:
    rsp = ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="ok"),
        raw=SimpleNamespace(),
    )

    usage = get_usage_from_response("LiteLLM", rsp)

    assert usage.request_tokens == 0
    assert usage.response_tokens == 0
    assert usage.total_tokens == 0
    assert usage.requests == 1


def test_litellm_usage_with_dict_raw_response() -> None:
    rsp = ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="ok"),
        raw={
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 8,
                "total_tokens": 23,
            }
        },
    )

    usage = get_usage_from_response("LiteLLM", rsp)

    assert usage.request_tokens == 15
    assert usage.response_tokens == 8
    assert usage.total_tokens == 23


def test_litellm_usage_computes_total_when_missing() -> None:
    rsp = ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="ok"),
        raw=SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        ),
    )

    usage = get_usage_from_response("LiteLLM", rsp)

    assert usage.request_tokens == 10
    assert usage.response_tokens == 5
    assert usage.total_tokens == 15


def test_litellm_no_raw_response_raises() -> None:
    rsp = ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="ok"),
        raw=None,
    )

    import pytest

    with pytest.raises(ValueError, match="No raw response"):
        get_usage_from_response("LiteLLM", rsp)


def test_track_usage_supports_mobilerun_anthropic_wrapper() -> None:
    llm = load_llm("Anthropic", model="claude-opus-4-8", api_key="stub")

    tracker = track_usage(llm)

    assert isinstance(tracker, TokenCountingHandler)
    assert tracker.provider == "Anthropic"
