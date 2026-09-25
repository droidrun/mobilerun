import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from mobilerun.agent.providers.anthropic import ANTHROPIC_HIGHRES_MODELS
from mobilerun.agent.utils.oauth.anthropic_oauth_llm import (
    DEFAULT_CC_VERSION,
    DEFAULT_CLAUDE_CODE_VERSION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_USER_AGENT,
    AnthropicOAuthLLM,
)


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "content": [{"type": "text", "text": "ok"}],
            "id": "msg_test",
            "usage": {},
            "stop_reason": "end_turn",
        }


class _CapturingSession:
    def __init__(self):
        self.payload = None
        self.headers = None

    def post(self, url, headers, json, timeout):
        self.payload = dict(json)
        self.headers = dict(headers)
        return _FakeResponse()


def _session_for(**kwargs):
    llm = AnthropicOAuthLLM(
        access_token="test-token",
        credential_path=None,
        **kwargs,
    )
    session = _CapturingSession()
    llm._session = session
    llm.chat([ChatMessage(role=MessageRole.USER, content="hello")])
    return session


def _payload_for(**kwargs):
    return _session_for(**kwargs).payload


def test_default_max_tokens_is_8192():
    assert DEFAULT_MAX_TOKENS == 8192
    assert AnthropicOAuthLLM(credential_path=None).metadata.num_output == 8192


def test_default_sonnet_payload_sends_max_tokens_without_temperature():
    payload = _payload_for()

    assert payload["model"] == "claude-sonnet-5"
    assert payload["max_tokens"] == 8192
    assert "temperature" not in payload


def test_opus_4_8_payload_sends_max_tokens_without_temperature():
    payload = _payload_for(model="claude-opus-4-8")

    assert payload["model"] == "claude-opus-4-8"
    assert payload["max_tokens"] == 8192
    assert "temperature" not in payload


@pytest.mark.parametrize(
    ("model", "context_window"),
    [
        ("claude-opus-5-5", 1_000_000),
        ("claude-opus-5", 1_000_000),
        ("claude-sonnet-5", 1_000_000),
        ("claude-fable-5-1", 1_000_000),
        ("claude-fable-5", 1_000_000),
        ("claude-opus-4-8", 1_000_000),
        ("claude-opus-4-7", 1_000_000),
        ("claude-opus-4-6", 1_000_000),
        ("claude-sonnet-4-6", 1_000_000),
        ("claude-haiku-4-5", 200_000),
    ],
)
def test_current_model_metadata_has_verified_context_window(model, context_window):
    metadata = AnthropicOAuthLLM(model=model, credential_path=None).metadata

    assert metadata.model_name == model
    assert metadata.context_window == context_window


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-fable-5-1",
        "claude-fable-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
    ],
)
def test_models_without_sampling_strip_all_final_payload_overrides(model):
    llm = AnthropicOAuthLLM(
        model=model,
        access_token="test-token",
        credential_path=None,
        temperature=0.7,
        additional_kwargs={"temperature": 0.6, "top_p": 0.5, "top_k": 10},
    )
    session = _CapturingSession()
    llm._session = session

    llm.chat(
        [ChatMessage(role=MessageRole.USER, content="hello")],
        temperature=0.4,
        top_p=0.3,
        top_k=5,
    )

    assert session.payload["model"] == model
    assert {"temperature", "top_p", "top_k"}.isdisjoint(session.payload)


@pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-opus-5-5"])
def test_current_models_use_high_resolution_vision_budget(model):
    assert model in ANTHROPIC_HIGHRES_MODELS


def test_fable_5_1_uses_current_claude_code_identity_defaults():
    session = _session_for(model="claude-fable-5-1")

    assert DEFAULT_CLAUDE_CODE_VERSION == "2.1.281"
    assert DEFAULT_USER_AGENT == "claude-cli/2.1.281"
    assert DEFAULT_CC_VERSION == "2.1.281.000"
    assert session.headers["User-Agent"] == DEFAULT_USER_AGENT
    assert f"cc_version={DEFAULT_CC_VERSION};" in session.payload["system"][0]["text"]


@pytest.mark.parametrize(
    "model",
    ["claude-fable-5-1", "claude-opus-5-5", "claude-opus-4-7", "claude-haiku-4-5"],
)
def test_oauth_structured_predict_uses_text_pydantic_extraction(monkeypatch, model):
    from llama_index.core.base.llms.types import ChatResponse
    from llama_index.core.prompts import PromptTemplate
    from pydantic import BaseModel

    class StructuredResult(BaseModel):
        value: str

    llm = AnthropicOAuthLLM(
        model=model,
        access_token="test-token",
        credential_path=None,
    )
    monkeypatch.setattr(
        type(llm),
        "chat",
        lambda _self, _messages, **_kwargs: ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content='{"value":"OK"}',
            )
        ),
    )

    result = llm.structured_predict(
        StructuredResult,
        PromptTemplate("Return {value}."),
        value="OK",
    )

    assert llm.metadata.is_function_calling_model is False
    assert result == StructuredResult(value="OK")
    assert (
        AnthropicOAuthLLM(credential_path=None).metadata.is_function_calling_model
        is False
    )


@pytest.mark.parametrize("model", ["claude-opus-4-6", "claude-sonnet-4-6"])
def test_4_6_models_keep_supported_sampling_fields(model):
    payload = _payload_for(
        model=model,
        temperature=0.7,
        additional_kwargs={"top_p": 0.5, "top_k": 10},
    )

    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.5
    assert payload["top_k"] == 10


def _chat_payload(messages):
    llm = AnthropicOAuthLLM(access_token="test-token", credential_path=None)
    session = _CapturingSession()
    llm._session = session
    llm.chat(messages)
    return session.payload


def _tiny_png() -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def test_image_block_is_sent_as_base64_image_content():
    import base64

    from llama_index.core.base.llms.types import ImageBlock, TextBlock

    png = _tiny_png()
    payload = _chat_payload(
        [
            ChatMessage(
                role=MessageRole.USER,
                blocks=[TextBlock(text="what is this?"), ImageBlock(image=png)],
            )
        ]
    )

    content = payload["messages"][0]["content"]
    assert isinstance(content, list)
    assert [block["type"] for block in content] == ["text", "image"]
    assert content[0]["text"] == "what is this?"
    source = content[1]["source"]
    assert source["type"] == "base64"
    assert source["media_type"] == "image/png"
    assert base64.b64decode(source["data"]) == png


def test_image_only_message_is_sent_as_image_content():
    from llama_index.core.base.llms.types import ImageBlock

    payload = _chat_payload(
        [ChatMessage(role=MessageRole.USER, blocks=[ImageBlock(image=_tiny_png())])]
    )

    content = payload["messages"][0]["content"]
    assert isinstance(content, list)
    assert [block["type"] for block in content] == ["image"]


def test_jpeg_media_type_is_detected():
    from io import BytesIO

    from llama_index.core.base.llms.types import ImageBlock
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (2, 2)).save(buf, format="JPEG")
    payload = _chat_payload(
        [
            ChatMessage(
                role=MessageRole.USER,
                blocks=[ImageBlock(image=buf.getvalue(), image_mimetype="image/jpeg")],
            )
        ]
    )

    assert payload["messages"][0]["content"][0]["source"]["media_type"] == "image/jpeg"


def test_text_only_messages_keep_plain_string_content():
    payload = _chat_payload(
        [
            ChatMessage(role=MessageRole.USER, content="hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="hi"),
            ChatMessage(role=MessageRole.USER, content="bye"),
        ]
    )

    assert [m["content"] for m in payload["messages"]] == ["hello", "hi", "bye"]
    assert all(isinstance(m["content"], str) for m in payload["messages"])


def test_system_message_text_reaches_system_blocks():
    payload = _chat_payload(
        [
            ChatMessage(role=MessageRole.SYSTEM, content="be terse"),
            ChatMessage(role=MessageRole.USER, content="hello"),
        ]
    )

    system_texts = [block["text"] for block in payload["system"]]
    assert "be terse" in system_texts
    assert all(m["role"] != "system" for m in payload["messages"])


def test_default_timeout_covers_unstreamed_thinking_replies():
    assert AnthropicOAuthLLM(credential_path=None).timeout == 60.0
