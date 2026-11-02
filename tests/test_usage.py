import json
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


def _streamed_responses_chat_response(*, raw, additional_kwargs=None) -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="ok"),
        raw=raw,
        additional_kwargs=additional_kwargs or {},
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


def test_openai_responses_extracts_usage_from_completed_stream_object() -> None:
    chat_response = _streamed_responses_chat_response(
        raw=SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=7,
                    output_tokens=5,
                    total_tokens=12,
                )
            ),
        )
    )

    usage = get_usage_from_response("OpenAIResponses", chat_response)

    assert usage.request_tokens == 7
    assert usage.response_tokens == 5
    assert usage.total_tokens == 12
    assert usage.requests == 1


def test_openai_responses_extracts_usage_from_completed_stream_dict() -> None:
    chat_response = _streamed_responses_chat_response(
        raw={
            "type": "response.completed",
            "response": {
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 4,
                    "total_tokens": 15,
                }
            },
        }
    )

    usage = get_usage_from_response("OpenAIResponses", chat_response)

    assert usage.request_tokens == 11
    assert usage.response_tokens == 4
    assert usage.total_tokens == 15


def test_openai_responses_extracts_completed_stream_additional_usage_fallback() -> None:
    chat_response = _streamed_responses_chat_response(
        raw={"type": "response.completed"},
        additional_kwargs={
            "usage": {
                "input_tokens": 13,
                "output_tokens": 6,
                "total_tokens": 19,
            }
        },
    )

    usage = get_usage_from_response("xai_oauth", chat_response)

    assert usage.request_tokens == 13
    assert usage.response_tokens == 6
    assert usage.total_tokens == 19


def test_openai_responses_extracts_object_usage_without_raw_event() -> None:
    chat_response = _streamed_responses_chat_response(
        raw=None,
        additional_kwargs={
            "usage": SimpleNamespace(
                input_tokens=17,
                output_tokens=8,
                total_tokens=25,
            )
        },
    )

    usage = get_usage_from_response("OpenAIResponses", chat_response)

    assert usage.request_tokens == 17
    assert usage.response_tokens == 8
    assert usage.total_tokens == 25


def test_track_usage_supports_mobilerun_anthropic_wrapper() -> None:
    llm = load_llm("Anthropic", model="claude-opus-4-8", api_key="stub")

    tracker = track_usage(llm)

    assert isinstance(tracker, TokenCountingHandler)
    assert tracker.provider == "Anthropic"


def test_track_usage_supports_mobilerun_google_genai_wrapper() -> None:
    llm = load_llm(
        "GoogleGenAI",
        model="gemini-4-argon",
        api_key="stub",
        max_tokens=64,
        context_window=1_000_000,
    )

    assert type(llm).__name__ == "MobilerunGoogleGenAI"
    assert track_usage(llm).provider == "GoogleGenAI"
    assert create_tracker(llm).provider == "GoogleGenAI"


def _google_genai_chat_response(**usage_metadata) -> ChatResponse:
    from google.genai import types
    from llama_index.llms.google_genai.utils import chat_from_gemini_response

    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text="ok")]),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(**usage_metadata),
    )
    return chat_from_gemini_response(response, [])


def test_google_genai_response_tokens_include_thinking() -> None:
    chat_response = _google_genai_chat_response(
        prompt_token_count=2753,
        candidates_token_count=52,
        thoughts_token_count=76,
        total_token_count=2881,
    )

    for provider in ("GenAI", "GoogleGenAI", "MobilerunGoogleGenAI"):
        usage = get_usage_from_response(provider, chat_response)

        assert usage.request_tokens == 2753
        assert usage.response_tokens == 128
        assert usage.total_tokens == 2881
        assert usage.requests == 1


def test_google_genai_without_thinking_counts_candidates_only() -> None:
    chat_response = _google_genai_chat_response(
        prompt_token_count=10, candidates_token_count=4, total_token_count=14
    )

    usage = get_usage_from_response("GenAI", chat_response)

    assert (usage.request_tokens, usage.response_tokens, usage.total_tokens) == (
        10,
        4,
        14,
    )


def test_google_genai_missing_total_is_prompt_plus_output_and_thinking() -> None:
    chat_response = _google_genai_chat_response(
        prompt_token_count=20, candidates_token_count=5, thoughts_token_count=7
    )

    usage = get_usage_from_response("GenAI", chat_response)

    assert (usage.request_tokens, usage.response_tokens, usage.total_tokens) == (
        20,
        12,
        32,
    )


class _GeminiSSEResponse:
    ok = True
    status_code = 200
    text = ""

    def __init__(self, chunks):
        self.chunks = chunks

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode):
        for chunk in self.chunks:
            yield f"data: {json.dumps(chunk)}"
            yield ""


class _GeminiSSESession:
    def __init__(self, chunks):
        self.chunks = chunks

    def post(self, url, **kwargs):
        return _GeminiSSEResponse(self.chunks)


def _code_assist_chunk(text: str, usage_metadata=None) -> dict:
    response = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    if usage_metadata is not None:
        response["usageMetadata"] = usage_metadata
    return {"traceId": "trace", "response": response}


def _gemini_oauth_chat(usage_metadata: dict) -> tuple[str, ChatResponse]:
    from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
        GeminiOAuthCodeAssistLLM,
    )

    llm = GeminiOAuthCodeAssistLLM(access_token="test-token", credential_path=None)
    llm._session = _GeminiSSESession(
        [
            _code_assist_chunk("O", {"promptTokenCount": 30}),
            _code_assist_chunk("K", usage_metadata),
        ]
    )
    return llm.class_name(), llm.chat(
        [ChatMessage(role=MessageRole.USER, content="Reply OK")]
    )


def test_gemini_oauth_response_tokens_include_thinking() -> None:
    provider, chat_response = _gemini_oauth_chat(
        {
            "promptTokenCount": 30,
            "candidatesTokenCount": 2,
            "thoughtsTokenCount": 40,
            "totalTokenCount": 72,
        }
    )

    for name in (provider, "GeminiOAuthCodeAssistLLM"):
        usage = get_usage_from_response(name, chat_response)

        assert usage.request_tokens == 30
        assert usage.response_tokens == 42
        assert usage.total_tokens == 72


def test_gemini_oauth_missing_total_is_prompt_plus_output_and_thinking() -> None:
    provider, chat_response = _gemini_oauth_chat(
        {
            "promptTokenCount": 30,
            "candidatesTokenCount": 2,
            "thoughtsTokenCount": 40,
        }
    )

    usage = get_usage_from_response(provider, chat_response)

    assert (usage.request_tokens, usage.response_tokens, usage.total_tokens) == (
        30,
        42,
        72,
    )


def test_responses_adds_reasoning_reported_outside_output_tokens() -> None:
    chat_response = _streamed_responses_chat_response(
        raw={
            "type": "response.completed",
            "response": {
                "usage": {
                    "input_tokens": 199,
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 186},
                    "total_tokens": 386,
                }
            },
        }
    )

    usage = get_usage_from_response("xai_oauth", chat_response)

    assert usage.request_tokens == 199
    assert usage.response_tokens == 187
    assert usage.total_tokens == 386


def test_responses_does_not_double_count_reasoning_inside_output_tokens() -> None:
    chat_response = ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="ok"),
        raw=SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=50,
                output_tokens_details=SimpleNamespace(reasoning_tokens=40),
                total_tokens=60,
            )
        ),
    )

    usage = get_usage_from_response("OpenAIOAuth", chat_response)

    assert (usage.request_tokens, usage.response_tokens, usage.total_tokens) == (
        10,
        50,
        60,
    )


def _openai_like_chat_response(**usage) -> ChatResponse:
    from openai.types import CompletionUsage
    from openai.types.completion_usage import CompletionTokensDetails

    reasoning_tokens = usage.pop("reasoning_tokens")
    return ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="ok"),
        raw=SimpleNamespace(
            usage=CompletionUsage(
                **usage,
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=reasoning_tokens
                ),
            )
        ),
    )


def test_openai_like_adds_reasoning_reported_outside_completion_tokens() -> None:
    chat_response = _openai_like_chat_response(
        prompt_tokens=32, completion_tokens=9, reasoning_tokens=94, total_tokens=135
    )

    usage = get_usage_from_response("OpenAILike", chat_response)

    assert (usage.request_tokens, usage.response_tokens, usage.total_tokens) == (
        32,
        103,
        135,
    )


def test_openai_like_does_not_double_count_reasoning_inside_completion() -> None:
    chat_response = _openai_like_chat_response(
        prompt_tokens=32, completion_tokens=103, reasoning_tokens=94, total_tokens=135
    )

    usage = get_usage_from_response("OpenAILike", chat_response)

    assert (usage.request_tokens, usage.response_tokens, usage.total_tokens) == (
        32,
        103,
        135,
    )
