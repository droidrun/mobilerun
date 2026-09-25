import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ToolCallBlock,
)

from mobilerun.agent.providers import resolve_provider_variant
from mobilerun.agent.providers.setup_service import (
    SetupSelection,
    create_profile_for_variant,
)
from mobilerun.agent.utils.oauth.openai_oauth_llm import OpenAIOAuth


class _AsyncEvents:
    """Mirrors openai.AsyncStream: async iteration and an async close()."""

    def __init__(self, events):
        self._events = iter(events)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None

    async def close(self):
        self.closed = True


def _offline_oauth_llm(tmp_path, model: str = "gpt-5.6-sol") -> OpenAIOAuth:
    return OpenAIOAuth(
        model=model,
        oauth_access_token="stub-access-token",
        oauth_expires_at_ms=4_102_444_800_000,
        oauth_credential_path=str(tmp_path / "auth-profiles.json"),
    )


def test_openai_oauth_constructs_with_updated_openai_adapter(tmp_path) -> None:
    llm = _offline_oauth_llm(tmp_path)

    assert llm.class_name() == "OpenAIOAuth"
    assert llm.model == "gpt-5.6-sol"
    assert llm.metadata.model_name == "gpt-5.6-sol"
    assert llm.metadata.context_window == 272_000


@pytest.mark.parametrize(
    "model_alias",
    (
        "gpt-5.6",
        "openai/gpt-5.6",
        "openai-codex/gpt-5.6",
    ),
)
def test_openai_oauth_normalizes_gpt_5_6_aliases(tmp_path, model_alias: str) -> None:
    llm = OpenAIOAuth(
        model=model_alias,
        oauth_access_token="stub-access-token",
        oauth_expires_at_ms=4_102_444_800_000,
        oauth_credential_path=str(tmp_path / "auth-profiles.json"),
    )

    assert llm.model == "gpt-5.6-sol"
    assert llm.metadata.model_name == "gpt-5.6-sol"
    assert llm.metadata.context_window == 272_000


def test_openai_oauth_normalizes_auth_model_alias(tmp_path) -> None:
    llm = OpenAIOAuth(
        auth_model="openai-codex/gpt-5.6",
        oauth_access_token="stub-access-token",
        oauth_expires_at_ms=4_102_444_800_000,
        oauth_credential_path=str(tmp_path / "auth-profiles.json"),
    )

    assert llm.model == "gpt-5.6-sol"


def test_openai_oauth_preserves_explicit_custom_model(tmp_path) -> None:
    llm = OpenAIOAuth(
        custom_model="acme/custom-reasoning-model",
        oauth_access_token="stub-access-token",
        oauth_expires_at_ms=4_102_444_800_000,
        oauth_credential_path=str(tmp_path / "auth-profiles.json"),
    )

    assert llm.model == "acme/custom-reasoning-model"


@pytest.mark.parametrize(
    ("auth_mode", "model_alias"),
    (
        ("api_key", "gpt-5.6"),
        ("api_key", "openai/gpt-5.6"),
        ("oauth", "gpt-5.6"),
        ("oauth", "openai/gpt-5.6"),
        ("oauth", "openai-codex/gpt-5.6"),
    ),
)
def test_openai_setup_profiles_normalize_gpt_5_6_aliases(
    auth_mode: str, model_alias: str
) -> None:
    variant = resolve_provider_variant("openai", auth_mode)
    profile = create_profile_for_variant(
        variant,
        SetupSelection(
            family_id="openai",
            variant_id=variant.id,
            auth_mode=auth_mode,
            model=model_alias,
            api_key_source="env",
        ),
    )

    assert profile.model == "gpt-5.6-sol"
    assert profile.provider == variant.runtime_provider_name


def test_openai_setup_profile_preserves_unknown_custom_model() -> None:
    variant = resolve_provider_variant("openai", "api_key")
    profile = create_profile_for_variant(
        variant,
        SetupSelection(
            family_id="openai",
            variant_id=variant.id,
            auth_mode="api_key",
            model="acme/custom-reasoning-model",
            api_key_source="env",
        ),
    )

    assert profile.model == "acme/custom-reasoning-model"


def test_openai_oauth_preserves_text_and_serializes_tool_arguments(tmp_path) -> None:
    llm = _offline_oauth_llm(tmp_path)
    payload = llm._build_responses_payload(
        [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                blocks=[
                    TextBlock(text="I will open Settings."),
                    ToolCallBlock(
                        tool_call_id="call-1",
                        tool_name="start_app",
                        tool_kwargs={"package": "com.android.settings"},
                    ),
                ],
            )
        ]
    )

    text_item = next(item for item in payload if item.get("role") == "assistant")
    tool_item = next(item for item in payload if item.get("type") == "function_call")

    assert text_item["content"] == [
        {"type": "output_text", "text": "I will open Settings."}
    ]
    assert isinstance(tool_item["arguments"], str)
    assert json.loads(tool_item["arguments"]) == {"package": "com.android.settings"}
    assert tool_item["call_id"] == "call-1"
    assert tool_item["name"] == "start_app"


@pytest.mark.parametrize("effort", (None, "low", "medium", "high", "xhigh", "max"))
def test_gpt_6_astra_forwards_supported_reasoning_only(
    tmp_path, monkeypatch, effort: str | None
) -> None:
    llm = OpenAIOAuth(
        model="gpt-6-astra",
        reasoning_effort=effort,
        oauth_access_token="stub-access-token",
        oauth_expires_at_ms=4_102_444_800_000,
        oauth_credential_path=str(tmp_path / "auth-profiles.json"),
        max_tokens=256,
        additional_kwargs={
            "include": (
                "message.output_text.logprobs",
                "reasoning.encrypted_content",
            )
        },
    )
    create_response = Mock(
        return_value=[
            SimpleNamespace(type="response.output_text.delta", delta="OK"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(output_text="OK"),
            ),
        ]
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=create_response))
    monkeypatch.setattr(OpenAIOAuth, "_get_client", lambda _self: client)

    runtime_kwargs = {
        "temperature": 0.7,
        "top_p": 0.8,
        "logprobs": True,
        "top_logprobs": 5,
    }
    llm._chat(
        [ChatMessage(role=MessageRole.USER, content="Reply with OK.")],
        **runtime_kwargs,
    )

    request = create_response.call_args.kwargs
    assert request["model"] == "gpt-6-astra"
    assert llm.metadata.context_window == 272_000
    assert request["reasoning"] == {"effort": effort or "low"}
    assert request["include"] == ["reasoning.encrypted_content"]
    assert {"temperature", "top_p", "logprobs", "top_logprobs"}.isdisjoint(request)


@pytest.mark.parametrize("effort", ("none", "minimal"))
def test_gpt_6_astra_rejects_unsupported_reasoning_before_request(
    tmp_path, monkeypatch, effort: str
) -> None:
    llm = OpenAIOAuth(
        model="gpt-6-astra",
        reasoning_effort=effort,
        oauth_access_token="stub-access-token",
        oauth_expires_at_ms=4_102_444_800_000,
        oauth_credential_path=str(tmp_path / "auth-profiles.json"),
        max_tokens=256,
    )
    create_response = Mock()
    client = SimpleNamespace(responses=SimpleNamespace(create=create_response))
    monkeypatch.setattr(OpenAIOAuth, "_get_client", lambda _self: client)

    with pytest.raises(ValueError, match=rf"reasoning effort '{effort}'"):
        llm._chat(
            [ChatMessage(role=MessageRole.USER, content="Reply with OK.")],
        )

    create_response.assert_not_called()


@pytest.mark.parametrize("source", ("additional_kwargs", "runtime"))
@pytest.mark.parametrize("effort", ("none", "minimal", "unsupported"))
def test_gpt_6_astra_rejects_invalid_final_merged_reasoning(
    tmp_path, monkeypatch, source: str, effort: str
) -> None:
    additional_kwargs = (
        {"reasoning": {"effort": effort}} if source == "additional_kwargs" else None
    )
    runtime_kwargs = {"reasoning": {"effort": effort}} if source == "runtime" else {}
    llm = OpenAIOAuth(
        model="gpt-6-astra",
        reasoning_effort="low",
        oauth_access_token="stub-access-token",
        oauth_expires_at_ms=4_102_444_800_000,
        oauth_credential_path=str(tmp_path / "auth-profiles.json"),
        additional_kwargs=additional_kwargs,
    )
    create_response = Mock()
    client = SimpleNamespace(responses=SimpleNamespace(create=create_response))
    monkeypatch.setattr(OpenAIOAuth, "_get_client", lambda _self: client)

    with pytest.raises(ValueError, match=rf"reasoning effort '{effort}'"):
        llm._chat(
            [ChatMessage(role=MessageRole.USER, content="Reply with OK.")],
            **runtime_kwargs,
        )

    create_response.assert_not_called()


@pytest.mark.parametrize(
    ("source", "effort"),
    (("additional_kwargs", "xhigh"), ("runtime", "max")),
)
def test_gpt_6_astra_accepts_supported_final_merged_reasoning(
    tmp_path, source: str, effort: str
) -> None:
    additional_kwargs = (
        {"reasoning": {"effort": effort}} if source == "additional_kwargs" else None
    )
    runtime_kwargs = {"reasoning": {"effort": effort}} if source == "runtime" else {}
    llm = OpenAIOAuth(
        model="gpt-6-astra",
        reasoning_effort="low",
        oauth_access_token="stub-access-token",
        oauth_expires_at_ms=4_102_444_800_000,
        oauth_credential_path=str(tmp_path / "auth-profiles.json"),
        additional_kwargs=additional_kwargs,
    )

    assert llm._sanitize_reasoning_kwargs(runtime_kwargs)["reasoning"] == {
        "effort": effort
    }


def test_gpt_6_astra_async_request_uses_exact_model_and_reasoning(
    tmp_path, monkeypatch
) -> None:
    llm = OpenAIOAuth(
        model="gpt-6-astra",
        reasoning_effort="low",
        oauth_access_token="stub-access-token",
        oauth_expires_at_ms=4_102_444_800_000,
        oauth_credential_path=str(tmp_path / "auth-profiles.json"),
        max_tokens=256,
    )
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return _AsyncEvents(
            [
                SimpleNamespace(type="response.output_text.delta", delta="OK"),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(output_text="OK"),
                ),
            ]
        )

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    monkeypatch.setattr(OpenAIOAuth, "_get_aclient", lambda _self: client)

    response = asyncio.run(
        llm._achat(
            [ChatMessage(role=MessageRole.USER, content="Reply with OK.")],
            temperature=0.7,
            top_logprobs=5,
        )
    )

    assert response.message.content == "OK"
    assert calls == [
        {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Reply with OK."}],
                }
            ],
            "model": "gpt-6-astra",
            "instructions": "You are a helpful coding assistant.",
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "store": False,
            "stream": True,
            "reasoning": {"effort": "low"},
        }
    ]


def test_oauth_implicit_default_matches_catalog(tmp_path):
    llm = OpenAIOAuth(oauth_credential_path=str(tmp_path / "auth.json"))
    assert (
        llm.model
        == resolve_provider_variant("openai", "oauth").default_model
        == "gpt-6-astra"
    )


@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"])
@pytest.mark.parametrize("argument", ["model", "custom_model", "auth_model"])
@pytest.mark.parametrize("prefix", ["", "openai/", "openai-codex/"])
def test_unsupported_chatgpt_models_fail_locally(tmp_path, model, argument, prefix):
    with pytest.raises(ValueError, match="not supported.*gpt-6-astra"):
        OpenAIOAuth(
            **{argument: prefix + model},
            oauth_credential_path=str(tmp_path / "auth.json"),
        )


@pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-6-astra"])
@pytest.mark.parametrize("async_call", [False, True])
def test_oauth_structured_extraction_prompts_for_schema(
    tmp_path, monkeypatch, model, async_call
):
    from llama_index.core.base.llms.types import ChatResponse
    from llama_index.core.prompts import PromptTemplate
    from pydantic import BaseModel

    class Result(BaseModel):
        value: int

    captured = []
    llm = _offline_oauth_llm(tmp_path, model=model)

    def chat(_self, messages, **kwargs):
        captured.extend(messages)
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content='{"value":19}')
        )

    async def achat(_self, messages, **kwargs):
        return chat(_self, messages, **kwargs)

    monkeypatch.setattr(type(llm), "chat", chat)
    monkeypatch.setattr(type(llm), "achat", achat)
    prompt = PromptTemplate("Return 8 plus 11 as value.")
    result = (
        asyncio.run(llm.astructured_predict(Result, prompt))
        if async_call
        else llm.structured_predict(Result, prompt)
    )
    assert result.value == 19
    assert '"properties"' in "\n".join(str(m.content) for m in captured)
    assert '"value"' in "\n".join(str(m.content) for m in captured)


@pytest.mark.parametrize(
    ("model", "effort"),
    (("gpt-6-sol", "none"), ("gpt-6-luna", "xhigh"), ("gpt-5.6-sol", "high")),
)
def test_oauth_forwards_configured_reasoning_for_every_model(
    tmp_path, monkeypatch, model: str, effort: str
) -> None:
    llm = OpenAIOAuth(
        model=model,
        reasoning_effort=effort,
        oauth_access_token="stub-access-token",
        oauth_expires_at_ms=4_102_444_800_000,
        oauth_credential_path=str(tmp_path / "auth-profiles.json"),
    )
    create_response = Mock(
        return_value=[
            SimpleNamespace(type="response.output_text.delta", delta="OK"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(output_text="OK"),
            ),
        ]
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=create_response))
    monkeypatch.setattr(OpenAIOAuth, "_get_client", lambda _self: client)

    llm._chat([ChatMessage(role=MessageRole.USER, content="Reply with OK.")])

    request = create_response.call_args.kwargs
    assert request["model"] == model
    assert request["reasoning"] == {"effort": effort}


@pytest.mark.parametrize("model", ("gpt-6-sol", "gpt-6-luna"))
def test_oauth_gpt_6_sol_and_luna_have_no_reasoning_default(tmp_path, model) -> None:
    llm = _offline_oauth_llm(tmp_path, model=model)

    assert "reasoning" not in llm._sanitize_reasoning_kwargs({})


@pytest.mark.parametrize(
    "model", ("gpt-6-sol", "gpt-6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")
)
def test_oauth_models_reject_minimal_effort_locally(tmp_path, model) -> None:
    llm = OpenAIOAuth(
        model=model,
        reasoning_effort="minimal",
        oauth_access_token="stub-access-token",
        oauth_expires_at_ms=4_102_444_800_000,
        oauth_credential_path=str(tmp_path / "auth-profiles.json"),
    )

    with pytest.raises(ValueError, match=f"{model} does not support reasoning"):
        llm._sanitize_reasoning_kwargs({})


def test_oauth_saved_gpt_5_5_profile_still_loads(tmp_path) -> None:
    llm = _offline_oauth_llm(tmp_path, model="openai-codex/gpt-5.5")

    assert llm.model == "gpt-5.5"


def _stream_events(*deltas: str, usage: object | None = None) -> list:
    final = SimpleNamespace(output_text="".join(deltas), usage=usage)
    return [
        *(SimpleNamespace(type="response.output_text.delta", delta=d) for d in deltas),
        SimpleNamespace(type="response.completed", response=final),
    ]


def _streaming_oauth_llm(tmp_path, model: str, **kwargs) -> OpenAIOAuth:
    return OpenAIOAuth(
        model=model,
        oauth_access_token="stub-access-token",
        oauth_expires_at_ms=4_102_444_800_000,
        oauth_credential_path=str(tmp_path / "auth-profiles.json"),
        **kwargs,
    )


def _sync_client(events):
    return SimpleNamespace(
        responses=SimpleNamespace(create=Mock(return_value=events)),
        chat=SimpleNamespace(completions=SimpleNamespace(create=Mock())),
    )


@pytest.mark.parametrize(
    ("model", "reasoning_effort", "expected_reasoning"),
    (
        ("gpt-6-astra", None, {"effort": "low"}),
        ("gpt-6-sol", "high", {"effort": "high"}),
        ("gpt-5.6-sol", None, None),
    ),
)
def test_stream_chat_uses_codex_responses_stream(
    tmp_path, monkeypatch, model, reasoning_effort, expected_reasoning
) -> None:
    llm = _streaming_oauth_llm(tmp_path, model, reasoning_effort=reasoning_effort)
    client = _sync_client(_stream_events("Hel", "lo"))
    monkeypatch.setattr(OpenAIOAuth, "_get_client", lambda _self: client)

    chunks = list(
        llm.stream_chat(
            [ChatMessage(role=MessageRole.USER, content="Say hello.")],
            temperature=0.7,
        )
    )

    assert [chunk.delta for chunk in chunks] == ["Hel", "lo", ""]
    assert [chunk.message.content for chunk in chunks] == ["Hel", "Hello", "Hello"]
    assert chunks[-1].raw.output_text == "Hello"
    client.chat.completions.create.assert_not_called()
    request = client.responses.create.call_args.kwargs
    assert request["model"] == model
    assert request["stream"] is True
    assert request["tools"] == []
    assert request["store"] is False
    assert request.get("reasoning") == expected_reasoning
    assert "temperature" not in request


@pytest.mark.parametrize(
    ("model", "reasoning_effort", "expected_reasoning"),
    (
        ("gpt-6-astra", None, {"effort": "low"}),
        ("gpt-6-sol", "none", {"effort": "none"}),
    ),
)
def test_astream_chat_uses_codex_responses_stream(
    tmp_path, monkeypatch, model, reasoning_effort, expected_reasoning
) -> None:
    llm = _streaming_oauth_llm(tmp_path, model, reasoning_effort=reasoning_effort)
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return _AsyncEvents(_stream_events("O", "K"))

    completions_create = Mock()
    client = SimpleNamespace(
        responses=SimpleNamespace(create=create),
        chat=SimpleNamespace(completions=SimpleNamespace(create=completions_create)),
    )
    monkeypatch.setattr(OpenAIOAuth, "_get_aclient", lambda _self: client)

    async def run():
        stream = await llm.astream_chat(
            [ChatMessage(role=MessageRole.USER, content="Reply with OK.")]
        )
        return [chunk async for chunk in stream]

    chunks = asyncio.run(run())

    assert [chunk.delta for chunk in chunks] == ["O", "K", ""]
    assert chunks[-1].message.content == "OK"
    assert chunks[-1].raw.output_text == "OK"
    completions_create.assert_not_called()
    [request] = calls
    assert request["model"] == model
    assert request["stream"] is True
    assert request["reasoning"] == expected_reasoning


def test_stream_complete_routes_through_codex_responses(tmp_path, monkeypatch) -> None:
    llm = _streaming_oauth_llm(tmp_path, "gpt-6-astra")
    client = _sync_client(_stream_events("O", "K"))
    monkeypatch.setattr(OpenAIOAuth, "_get_client", lambda _self: client)

    chunks = list(llm.stream_complete("Reply with OK."))

    assert chunks[-1].text == "OK"
    client.chat.completions.create.assert_not_called()


def test_stream_chat_uses_final_text_when_no_deltas_arrive(
    tmp_path, monkeypatch
) -> None:
    llm = _streaming_oauth_llm(tmp_path, "gpt-6-astra")
    final = SimpleNamespace(output_text="OK")
    client = _sync_client([SimpleNamespace(type="response.completed", response=final)])
    monkeypatch.setattr(OpenAIOAuth, "_get_client", lambda _self: client)

    chunks = list(
        llm.stream_chat([ChatMessage(role=MessageRole.USER, content="Reply OK")])
    )

    assert [(chunk.delta, chunk.message.content) for chunk in chunks] == [("OK", "OK")]
    assert chunks[0].raw is final


def test_stream_chat_falls_back_to_backend_api_on_not_found(
    tmp_path, monkeypatch
) -> None:
    from mobilerun.agent.utils.oauth.openai_oauth_llm import DEFAULT_BACKEND_API_BASE

    llm = _streaming_oauth_llm(tmp_path, "gpt-6-astra")
    not_found = Exception("404 Not Found")
    not_found.status_code = 404
    create = Mock(side_effect=[not_found, _stream_events("OK")])
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    monkeypatch.setattr(OpenAIOAuth, "_get_client", lambda _self: client)

    chunks = list(
        llm.stream_chat([ChatMessage(role=MessageRole.USER, content="Reply OK")])
    )

    assert chunks[-1].message.content == "OK"
    assert create.call_count == 2
    assert llm._responses_api_base == DEFAULT_BACKEND_API_BASE


def test_stream_chat_rejects_unsupported_effort_before_request(
    tmp_path, monkeypatch
) -> None:
    llm = _streaming_oauth_llm(tmp_path, "gpt-6-sol", reasoning_effort="minimal")
    client = _sync_client(_stream_events("OK"))
    monkeypatch.setattr(OpenAIOAuth, "_get_client", lambda _self: client)

    with pytest.raises(ValueError, match="does not support reasoning effort"):
        llm.stream_chat([ChatMessage(role=MessageRole.USER, content="Reply OK")])

    client.responses.create.assert_not_called()


def test_streamed_response_keeps_text_and_usage(tmp_path, monkeypatch) -> None:
    from mobilerun.agent.usage import get_usage_from_response
    from mobilerun.agent.utils.inference import _stream_response

    usage = SimpleNamespace(input_tokens=11, output_tokens=2, total_tokens=13)
    llm = _streaming_oauth_llm(tmp_path, "gpt-6-astra")

    async def create(**kwargs):
        return _AsyncEvents(_stream_events("O", "K", usage=usage))

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    monkeypatch.setattr(OpenAIOAuth, "_get_aclient", lambda _self: client)

    response = asyncio.run(
        _stream_response(
            llm,
            [ChatMessage(role=MessageRole.USER, content="Reply with OK.")],
            timeout=5,
        )
    )

    assert response.message.content == "OK"
    result = get_usage_from_response("OpenAIOAuth", response)
    assert (result.request_tokens, result.response_tokens) == (11, 2)


def test_astream_chat_closes_the_stream_when_the_consumer_stops(
    tmp_path, monkeypatch
) -> None:
    llm = _streaming_oauth_llm(tmp_path, "gpt-6-astra")
    events = _AsyncEvents(_stream_events("O", "K"))

    async def create(**kwargs):
        return events

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    monkeypatch.setattr(OpenAIOAuth, "_get_aclient", lambda _self: client)

    async def run():
        stream = await llm._astream_chat(
            [ChatMessage(role=MessageRole.USER, content="Reply with OK.")]
        )
        first = await stream.__anext__()
        await stream.aclose()
        return first

    first = asyncio.run(run())

    assert first.delta == "O"
    assert events.closed is True


def test_achat_closes_the_async_stream(tmp_path, monkeypatch) -> None:
    llm = _streaming_oauth_llm(tmp_path, "gpt-6-astra")
    events = _AsyncEvents(_stream_events("OK"))

    async def create(**kwargs):
        return events

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    monkeypatch.setattr(OpenAIOAuth, "_get_aclient", lambda _self: client)

    asyncio.run(llm._achat([ChatMessage(role=MessageRole.USER, content="Reply OK")]))

    assert events.closed is True
