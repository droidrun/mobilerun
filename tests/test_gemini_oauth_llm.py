import json

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from mobilerun.agent.providers.registry import resolve_provider_variant
from mobilerun.agent.providers.setup_service import (
    SetupSelection,
    create_profile_for_variant,
)
from mobilerun.agent.utils.llm_picker import load_llm


@pytest.mark.parametrize(
    "model",
    (
        "gemini-3.8-flash-tiered",
        "gemini-3.7-flash-tiered",
        "gemini-3.6-flash-low",
        "gemini-3.6-flash-medium",
        "gemini-3.6-flash-high",
    ),
)
def test_gemini_oauth_profile_sends_supported_models_verbatim(
    tmp_path, model: str
) -> None:
    variant = resolve_provider_variant("gemini", "oauth")
    profile = create_profile_for_variant(
        variant,
        SetupSelection(
            family_id="gemini",
            variant_id=variant.id,
            auth_mode="oauth",
            model=model,
            credential_path=str(tmp_path / "missing-auth-profiles.json"),
        ),
    )

    llm = load_llm(
        profile.provider,
        model=profile.model,
        credential_path=profile.credential_path,
        **profile.kwargs,
    )
    payload = llm._to_code_assist_request(
        [ChatMessage(role=MessageRole.USER, content="hello")]
    )

    assert profile.provider == "gemini_oauth_code_assist"
    assert profile.model == model
    assert payload["model"] == model


def _gemini_llm(model: str | None = None):
    from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
        GeminiOAuthCodeAssistLLM,
    )

    kwargs = {"model": model} if model is not None else {}
    return GeminiOAuthCodeAssistLLM(
        access_token="test-token", credential_path=None, **kwargs
    )


class _SSEStreamResponse:
    ok = True
    status_code = 200
    text = ""

    def __init__(self, chunks):
        self.chunks = chunks

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode):
        assert decode_unicode is True
        for chunk in self.chunks:
            yield f"data: {json.dumps(chunk)}"
            yield ""


class _SSESession:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _SSEStreamResponse(self.chunks)


def _text_chunk(text: str, *, final: bool = False):
    response = {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
    }
    if final:
        response.update(
            {
                "usageMetadata": {"totalTokenCount": 4},
                "modelVersion": "tiered-test-version",
            }
        )
    return {"traceId": f"trace-{text}", "response": response}


@pytest.mark.parametrize(
    "model", ("gemini-3.8-flash-tiered", "gemini-3.7-flash-tiered")
)
def test_tiered_complete_routes_exact_model_over_sse(model: str) -> None:
    llm = _gemini_llm(model)
    session = _SSESession([_text_chunk("O"), _text_chunk("K", final=True)])
    llm._session = session

    response = llm.complete("Reply OK", generation_config={"maxOutputTokens": 32})

    assert response.text == "OK"
    assert response.additional_kwargs == {
        "trace_id": "trace-K",
        "usage": {"totalTokenCount": 4},
        "model_version": "tiered-test-version",
    }
    [(url, call)] = session.calls
    assert url.endswith("/v1internal:streamGenerateContent")
    assert call["params"] == {"alt": "sse"}
    assert call["headers"]["Accept"] == "text/event-stream"
    assert call["stream"] is True
    assert call["json"]["model"] == model
    request = call["json"]["request"]
    assert request["generationConfig"] == {"maxOutputTokens": 32}
    assert request["contents"] == [{"role": "user", "parts": [{"text": "Reply OK"}]}]


@pytest.mark.parametrize(
    "model", ("gemini-3.8-flash-tiered", "gemini-3.7-flash-tiered")
)
def test_tiered_stream_chat_routes_exact_model_and_image_payload(model: str) -> None:
    import base64

    from llama_index.core.base.llms.types import ImageBlock, TextBlock

    png = _tiny_image("PNG")
    llm = _gemini_llm(model)
    session = _SSESession([_text_chunk("first"), _text_chunk(" second")])
    llm._session = session

    chunks = list(
        llm.stream_chat(
            [
                ChatMessage(
                    role=MessageRole.USER,
                    blocks=[TextBlock(text="Inspect"), ImageBlock(image=png)],
                )
            ],
            generation_config={"maxOutputTokens": 32},
        )
    )

    assert [chunk.delta for chunk in chunks] == ["first", " second"]
    assert [chunk.message.content for chunk in chunks] == ["first", "first second"]
    [(url, call)] = session.calls
    assert url.endswith("/v1internal:streamGenerateContent")
    assert call["params"] == {"alt": "sse"}
    assert call["stream"] is True
    assert call["json"]["model"] == model
    request = call["json"]["request"]
    assert request["generationConfig"] == {"maxOutputTokens": 32}
    parts = request["contents"][0]["parts"]
    assert parts[0] == {"text": "Inspect"}
    assert parts[1]["inlineData"]["mimeType"] == "image/png"
    assert base64.b64decode(parts[1]["inlineData"]["data"]) == png


class _CatalogResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _CatalogSession:
    def __init__(self, payload):
        self.payload = payload

    def post(self, url, headers, json, timeout):
        return _CatalogResponse(self.payload)


def _models_from_catalog(payload):
    llm = _gemini_llm()
    llm._session = _CatalogSession(payload)
    return llm.fetch_available_models(access_token="catalog-token")


def test_fetch_available_models_uses_friendly_names_for_tiered_gemini_entries():
    models = _models_from_catalog(
        {
            "models": {
                "gemini-3.8-flash-tiered": {
                    "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
                    "supportsImages": True,
                },
                "gemini-3.7-flash-tiered": {
                    "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
                    "supportsImages": True,
                },
                "gemini-3.6-flash-high": {
                    "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
                    "displayName": "Gemini 3.6 Flash (High)",
                    "supportsImages": True,
                },
            }
        }
    )

    assert models == [
        {
            "id": "gemini-3.8-flash-tiered",
            "display_name": "gemini-3.8-flash",
            "supports_images": True,
        },
        {
            "id": "gemini-3.7-flash-tiered",
            "display_name": "gemini-3.7-flash",
            "supports_images": True,
        },
        {
            "id": "gemini-3.6-flash-high",
            "display_name": "Gemini 3.6 Flash (High)",
            "supports_images": True,
        },
    ]


@pytest.mark.parametrize(
    ("model_id", "expected_display_name"),
    [
        ("gemini-3.8-flash-tiered", "gemini-3.8-flash"),
        ("gemini-3.7-flash-tiered", "gemini-3.7-flash"),
    ],
)
def test_fetch_available_models_overrides_tiered_provider_display_name(
    model_id: str, expected_display_name: str
) -> None:
    models = _models_from_catalog(
        {
            "models": {
                model_id: {
                    "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
                    "displayName": f"Google label for {model_id}",
                    "supportsImages": True,
                }
            }
        }
    )

    assert models == [
        {
            "id": model_id,
            "display_name": expected_display_name,
            "supports_images": True,
        }
    ]


def test_fetch_available_models_rejects_unsafe_nameless_or_hidden_entries():
    models = _models_from_catalog(
        {
            "models": {
                "gemini-command-aux": {
                    "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
                },
                "gemini-internal-tiered": {
                    "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
                    "isInternal": True,
                },
                "gemini-deprecated-tiered": {
                    "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
                },
                "gemini-wrong-provider-tiered": {
                    "apiProvider": "API_PROVIDER_OTHER",
                },
                "gemini-malformed-tiered": "not metadata",
                "gemini-visible": {
                    "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
                    "displayName": "Gemini Visible",
                },
            },
            "deprecatedModelIds": {"gemini-deprecated-tiered": {}},
        }
    )

    assert models == [
        {
            "id": "gemini-visible",
            "display_name": "Gemini Visible",
            "supports_images": False,
        }
    ]


def _tiny_image(fmt: str) -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(buf, format=fmt)
    return buf.getvalue()


def test_image_block_is_sent_as_inline_data_part():
    import base64

    from llama_index.core.base.llms.types import ImageBlock, TextBlock

    png = _tiny_image("PNG")
    payload = _gemini_llm()._to_code_assist_request(
        [
            ChatMessage(
                role=MessageRole.USER,
                blocks=[TextBlock(text="what is this?"), ImageBlock(image=png)],
            )
        ]
    )

    parts = payload["request"]["contents"][0]["parts"]
    assert parts[0] == {"text": "what is this?"}
    inline = parts[1]["inlineData"]
    assert inline["mimeType"] == "image/png"
    assert base64.b64decode(inline["data"]) == png


def test_image_only_message_is_sent_as_inline_data():
    from llama_index.core.base.llms.types import ImageBlock

    payload = _gemini_llm()._to_code_assist_request(
        [
            ChatMessage(
                role=MessageRole.USER, blocks=[ImageBlock(image=_tiny_image("PNG"))]
            )
        ]
    )

    parts = payload["request"]["contents"][0]["parts"]
    assert len(parts) == 1 and "inlineData" in parts[0]


def test_jpeg_mime_type_is_detected():
    from llama_index.core.base.llms.types import ImageBlock

    payload = _gemini_llm()._to_code_assist_request(
        [
            ChatMessage(
                role=MessageRole.USER,
                blocks=[
                    ImageBlock(image=_tiny_image("JPEG"), image_mimetype="image/jpeg")
                ],
            )
        ]
    )

    inline = payload["request"]["contents"][0]["parts"][0]["inlineData"]
    assert inline["mimeType"] == "image/jpeg"


def test_text_only_messages_keep_single_text_part_shape():
    payload = _gemini_llm()._to_code_assist_request(
        [
            ChatMessage(role=MessageRole.USER, content="hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="hi"),
        ]
    )

    contents = payload["request"]["contents"]
    assert contents[0] == {"role": "user", "parts": [{"text": "hello"}]}
    assert contents[1] == {"role": "model", "parts": [{"text": "hi"}]}


def test_system_message_text_reaches_system_instruction():
    payload = _gemini_llm()._to_code_assist_request(
        [
            ChatMessage(role=MessageRole.SYSTEM, content="be terse"),
            ChatMessage(role=MessageRole.USER, content="hello"),
        ]
    )

    assert "be terse" in payload["request"]["systemInstruction"]["parts"][0]["text"]
    assert all(c["role"] != "system" for c in payload["request"]["contents"])


def test_oauth_default_model_resolves_to_antigravity_flash():
    from mobilerun.agent.providers.registry import resolve_provider_variant
    from mobilerun.agent.utils.llm_picker import load_llm
    from mobilerun.cli.oauth_actions import GEMINI_OAUTH_DEFAULT_MODEL

    # no model arg -> the Antigravity consumer default
    assert load_llm("gemini_oauth_code_assist").model == "gemini-3.8-flash-tiered"
    variant = resolve_provider_variant("gemini", "oauth")
    assert variant.models[0] == variant.default_model == GEMINI_OAUTH_DEFAULT_MODEL


@pytest.mark.parametrize(
    "model",
    ["gemini-3.5-flash-low", "gemini-3.5-flash-extra-low", "gemini-3-flash-agent"],
)
@pytest.mark.parametrize("argument", ["model", "custom_model"])
def test_retired_oauth_models_fail_before_loading_credentials(
    model, argument, monkeypatch
):
    from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
        GeminiOAuthCodeAssistLLM,
    )

    def unexpected_load(*args):
        pytest.fail("Retired selection must fail before credentials or network access")

    monkeypatch.setattr(
        GeminiOAuthCodeAssistLLM, "_load_credentials_from_file", unexpected_load
    )
    with pytest.raises(ValueError, match="retired.*gemini-3.8-flash-tiered.*configure"):
        GeminiOAuthCodeAssistLLM(**{argument: model})


def test_retired_oauth_models_are_hidden_even_when_provider_advertises_them():
    from mobilerun.agent.providers.registry import GEMINI_OAUTH_RETIRED_MODELS

    advertised = [
        *GEMINI_OAUTH_RETIRED_MODELS,
        "gemini-3.5-flash-lite",
        "gemini-future-custom",
    ]
    models = _models_from_catalog(
        {
            "models": {
                model: {
                    "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
                    "displayName": model,
                }
                for model in advertised
            }
        }
    )
    assert [model["id"] for model in models] == [
        "gemini-3.5-flash-lite",
        "gemini-future-custom",
    ]


def test_flash_lite_and_unknown_custom_oauth_models_load():
    from mobilerun.agent.utils.llm_picker import load_llm

    for model in ("gemini-3.5-flash-lite", "gemini-future-custom"):
        assert load_llm("gemini_oauth_code_assist", model=model).model == model


def test_oauth_explicit_default_model_is_honored_not_preset():
    from mobilerun.agent.utils.llm_picker import load_llm

    # explicit model equal to DEFAULT_MODEL must NOT fall through to a preset
    assert (
        load_llm("gemini_oauth_code_assist", model="gemini-3.7-flash-tiered").model
        == "gemini-3.7-flash-tiered"
    )


def test_oauth_explicit_agy_models_are_honored():
    from mobilerun.agent.utils.llm_picker import load_llm

    for m in (
        "gemini-3-flash",
        "gemini-pro-agent",
        "gemini-3.1-pro-low",
        "gemini-3.6-flash-low",
        "gemini-3.6-flash-medium",
        "gemini-3.6-flash-high",
    ):
        assert load_llm("gemini_oauth_code_assist", model=m).model == m


def test_oauth_preset_key_still_resolves():
    from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
        GeminiOAuthCodeAssistLLM,
    )

    assert (
        GeminiOAuthCodeAssistLLM(model_preset="flash").model
        == "gemini-3.8-flash-tiered"
    )


def test_public_api_model_ids_are_rejected_with_reconfigure_error():
    from mobilerun.agent.utils.llm_picker import load_llm

    for m in (
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
    ):
        with pytest.raises(ValueError):
            load_llm("gemini_oauth_code_assist", model=m)


def test_live_private_gemini_2_5_aliases_are_not_rejected():
    from mobilerun.agent.utils.llm_picker import load_llm

    for model in ("gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"):
        assert load_llm("gemini_oauth_code_assist", model=model).model == model


def test_never_sends_project_even_if_passed_as_kwarg():
    from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
        GeminiOAuthCodeAssistLLM,
    )

    llm = GeminiOAuthCodeAssistLLM(access_token="t", credential_path=None)
    payload = llm._to_code_assist_request(
        [ChatMessage(role=MessageRole.USER, content="hi")], project="proj-123"
    )
    assert "project" not in payload


def test_consumer_default_credential_slot_is_antigravity():
    from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
        DEFAULT_CREDENTIAL_SLOT,
        GeminiOAuthCodeAssistLLM,
    )

    llm = GeminiOAuthCodeAssistLLM(access_token="t", credential_path=None)
    assert llm.credential_slot == DEFAULT_CREDENTIAL_SLOT == "geminiAntigravityOauth"


def test_consumer_mode_uses_antigravity_client_and_aicode_scope():
    from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
        DEFAULT_CLIENT_ID,
        GeminiOAuthCodeAssistLLM,
    )

    llm = GeminiOAuthCodeAssistLLM(access_token="t", credential_path=None)
    assert llm.client_id == DEFAULT_CLIENT_ID
    assert any("aicode" in s for s in llm.scopes)
    assert "daily-cloudcode-pa" in llm.code_assist_endpoint


def test_flash_lite_preset_resolves_to_picker_model():
    from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
        GeminiOAuthCodeAssistLLM,
    )

    assert (
        GeminiOAuthCodeAssistLLM(model_preset="flash_lite").model
        == "gemini-3.5-flash-lite"
    )


def test_consumer_mode_ignores_bare_credential_file(tmp_path):
    import json

    from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
        GeminiOAuthCodeAssistLLM,
    )

    p = tmp_path / "bare.json"
    p.write_text(json.dumps({"access_token": "raw-old", "refresh_token": "raw-r"}))
    llm = GeminiOAuthCodeAssistLLM(credential_path=str(p))  # consumer mode default
    assert llm._cached_access_token is None
    assert llm._cached_refresh_token is None


def test_wizard_oauth_detection_handles_provider_field_names(tmp_path):
    import json

    from mobilerun.cli.configure_wizard import _oauth_credentials_present

    p = tmp_path / "auth.json"
    p.write_text(
        json.dumps(
            {
                "geminiAntigravityOauth": {"access_token": "a", "refresh_token": "b"},
                "claudeAiOauth": {"accessToken": "a", "refreshToken": "b"},
                "openaiOauth": {"access": "a", "refresh": "b"},
            }
        )
    )
    assert _oauth_credentials_present(str(p), "gemini_oauth_code_assist")
    assert _oauth_credentials_present(str(p), "anthropic_oauth")
    assert _oauth_credentials_present(str(p), "openai_oauth")

    p2 = tmp_path / "auth2.json"
    p2.write_text(json.dumps({"openaiOauth": {"access": "a"}}))
    # gemini slot absent -> not detected as present
    assert not _oauth_credentials_present(str(p2), "gemini_oauth_code_assist")


@pytest.mark.parametrize(
    ("model", "expected_config"),
    (
        ("gemini-3.8-flash-high", {}),
        ("gemini-pro-agent", {}),
        ("gemini-2.5-flash", {"temperature": 0.2}),
    ),
)
def test_profile_temperature_is_left_to_google_for_gemini_3(
    model: str, expected_config: dict
) -> None:
    from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
        GeminiOAuthCodeAssistLLM,
    )

    llm = GeminiOAuthCodeAssistLLM(
        model=model, access_token="test-token", credential_path=None, temperature=0.2
    )
    session = _SSESession([_text_chunk("OK", final=True)])
    llm._session = session

    llm.complete("Reply OK")

    [(_url, call)] = session.calls
    assert call["json"]["request"]["generationConfig"] == expected_config


def test_headers_use_a_versioned_antigravity_user_agent() -> None:
    import re

    headers = _gemini_llm()._build_headers("test-token")

    assert re.fullmatch(
        r"antigravity/2\.14\.0 [a-z0-9]+/[a-z0-9_]+", headers["User-Agent"]
    )


def test_fetch_available_models_skips_image_generation_models():
    models = _models_from_catalog(
        {
            "models": {
                "gemini-3.8-flash-tiered": {
                    "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
                    "supportsImages": True,
                },
                "gemini-3.1-flash-image": {
                    "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
                    "displayName": "Gemini 3.1 Flash Image",
                },
            },
            "imageGenerationModelIds": ["gemini-3.1-flash-image"],
        }
    )

    assert [model["id"] for model in models] == ["gemini-3.8-flash-tiered"]


@pytest.mark.parametrize(
    ("model", "expected_config"),
    (
        ("gemini-3.8-flash-tiered", {"maxOutputTokens": 32}),
        ("gemini-2.5-flash", {"maxOutputTokens": 32, "temperature": 0.9, "topP": 0.5}),
    ),
)
def test_call_time_sampling_is_left_to_google_for_gemini_3(
    model: str, expected_config: dict
) -> None:
    llm = _gemini_llm(model)
    session = _SSESession([_text_chunk("OK", final=True)])
    llm._session = session

    llm.complete(
        "Reply OK",
        generation_config={"maxOutputTokens": 32, "temperature": 0.9, "topP": 0.5},
    )

    [(_url, call)] = session.calls
    assert call["json"]["request"]["generationConfig"] == expected_config
