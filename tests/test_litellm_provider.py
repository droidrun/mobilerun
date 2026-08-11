"""Edge-case tests for the LiteLLM provider in mobilerun."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from mobilerun.agent.utils.llm_picker import load_llm, normalize_provider_name


# --- Provider alias resolution ---


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("litellm", "LiteLLM"),
        ("LiteLLM", "LiteLLM"),
        ("lite_llm", "LiteLLM"),
        ("lite-llm", "LiteLLM"),
    ],
)
def test_litellm_alias_resolution(alias: str, expected: str) -> None:
    assert normalize_provider_name(alias) == expected


# --- Constructor and kwargs ---


def test_litellm_loads_with_model_and_api_key() -> None:
    llm = load_llm("LiteLLM", model="anthropic/claude-sonnet-4-6", api_key="sk-test")
    assert llm.model == "anthropic/claude-sonnet-4-6"
    assert llm.api_key == "sk-test"


def test_litellm_loads_with_api_base() -> None:
    llm = load_llm(
        "LiteLLM",
        model="openai/gpt-4o",
        api_key="sk-test",
        api_base="http://myproxy:4000",
    )
    assert llm.api_base == "http://myproxy:4000"


def test_litellm_base_url_mapped_to_api_base() -> None:
    llm = load_llm(
        "LiteLLM",
        model="openai/gpt-4o",
        api_key="sk-test",
        base_url="http://myproxy:4000",
    )
    assert llm.api_base == "http://myproxy:4000"


def test_litellm_drop_params_always_true() -> None:
    llm = load_llm("LiteLLM", model="test/model", api_key="sk-test")
    kw = llm._get_litellm_kwargs()
    assert kw["drop_params"] is True


def test_litellm_kwargs_forwarding() -> None:
    llm = load_llm(
        "LiteLLM",
        model="anthropic/claude-sonnet-4-6",
        api_key="sk-key",
        api_base="http://localhost:4000",
        temperature=0.5,
        max_tokens=1024,
    )
    kw = llm._get_litellm_kwargs()
    assert kw["model"] == "anthropic/claude-sonnet-4-6"
    assert kw["temperature"] == 0.5
    assert kw["max_tokens"] == 1024
    assert kw["api_key"] == "sk-key"
    assert kw["api_base"] == "http://localhost:4000"
    assert kw["drop_params"] is True


def test_litellm_omits_none_api_key() -> None:
    llm = load_llm("LiteLLM", model="test/model")
    kw = llm._get_litellm_kwargs()
    assert "api_key" not in kw


def test_litellm_omits_none_api_base() -> None:
    llm = load_llm("LiteLLM", model="test/model", api_key="sk-test")
    kw = llm._get_litellm_kwargs()
    assert "api_base" not in kw


# --- Model string format ---


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-4o",
        "bedrock/anthropic.claude-sonnet-4-6-v1",
        "vertex_ai/gemini-2.5-flash",
        "groq/llama-4-scout-17b-16e-instruct",
        "mistral/mistral-large-latest",
        "deepseek/deepseek-chat",
    ],
)
def test_litellm_model_string_preserved(model: str) -> None:
    llm = load_llm("LiteLLM", model=model, api_key="sk-test")
    kw = llm._get_litellm_kwargs()
    assert kw["model"] == model


# --- Completion (mocked) ---


@patch("litellm.completion")
def test_litellm_complete(mock_completion: MagicMock) -> None:
    mock_completion.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="4"),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=1, total_tokens=11
        ),
    )

    llm = load_llm("LiteLLM", model="anthropic/claude-sonnet-4-6", api_key="sk-test")
    resp = llm.complete("What is 2+2?")

    assert resp.text == "4"
    mock_completion.assert_called_once()
    call_kwargs = mock_completion.call_args
    assert call_kwargs.kwargs["model"] == "anthropic/claude-sonnet-4-6"
    assert call_kwargs.kwargs["drop_params"] is True


@patch("litellm.completion")
def test_litellm_complete_empty_content(mock_completion: MagicMock) -> None:
    mock_completion.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
    )

    llm = load_llm("LiteLLM", model="test/model", api_key="sk-test")
    resp = llm.complete("test")
    assert resp.text == ""


@patch("litellm.completion")
def test_litellm_complete_auth_error(mock_completion: MagicMock) -> None:
    mock_completion.side_effect = Exception("AuthenticationError: Invalid API key")

    llm = load_llm("LiteLLM", model="test/model", api_key="sk-bad")
    with pytest.raises(Exception, match="AuthenticationError"):
        llm.complete("test")


# --- Chat (mocked) ---


@patch("litellm.completion")
def test_litellm_chat(mock_completion: MagicMock) -> None:
    mock_completion.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello!"))],
    )

    llm = load_llm("LiteLLM", model="openai/gpt-4o", api_key="sk-test")
    messages = [ChatMessage(role=MessageRole.USER, content="Hi")]
    resp = llm.chat(messages)

    assert resp.message.content == "Hello!"
    call_kwargs = mock_completion.call_args
    assert call_kwargs.kwargs["messages"][0]["role"] == "user"
    assert call_kwargs.kwargs["messages"][0]["content"] == "Hi"


# --- Streaming (mocked) ---


@patch("litellm.completion")
def test_litellm_stream_complete(mock_completion: MagicMock) -> None:
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hel"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="lo"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]),
    ]
    mock_completion.return_value = iter(chunks)

    llm = load_llm("LiteLLM", model="test/model", api_key="sk-test")
    result = list(llm.stream_complete("test"))

    assert len(result) == 3
    assert result[0].text == "Hel"
    assert result[1].text == "lo"
    assert result[2].text == ""
    call_kwargs = mock_completion.call_args
    assert call_kwargs.kwargs["stream"] is True


# --- Async chat (mocked) ---


def test_litellm_achat() -> None:
    import asyncio

    mock_resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="async ok"))],
    )
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_resp

        llm = load_llm("LiteLLM", model="openai/gpt-4o", api_key="sk-test")
        messages = [ChatMessage(role=MessageRole.USER, content="test")]
        resp = asyncio.get_event_loop().run_until_complete(llm.achat(messages))

        assert resp.message.content == "async ok"
        mock_acompletion.assert_called_once()


# --- Metadata ---


def test_litellm_metadata() -> None:
    llm = load_llm(
        "LiteLLM", model="anthropic/claude-sonnet-4-6", api_key="sk-test"
    )
    meta = llm.metadata
    assert meta.model_name == "anthropic/claude-sonnet-4-6"
    assert meta.is_chat_model is True


# --- Usage tracking ---


def test_litellm_usage_tracking() -> None:
    from mobilerun.agent.usage import create_tracker

    llm = load_llm("LiteLLM", model="anthropic/claude-sonnet-4-6", api_key="sk-test")
    tracker = create_tracker(llm)
    assert tracker.provider == "LiteLLM"
