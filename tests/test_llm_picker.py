import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from mobilerun.agent.utils.llm_picker import (
    load_llm,
    load_llms_from_profiles,
    normalize_provider_name,
)
from mobilerun.config_manager.config_manager import LLMProfile


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("OpenAI", "OpenAIResponses"),
        ("openai", "OpenAIResponses"),
        ("GPT", "OpenAIResponses"),
        ("Gemini", "GoogleGenAI"),
        ("Google", "GoogleGenAI"),
        ("Claude", "Anthropic"),
        ("OpenAI Compatible", "OpenAILike"),
        ("OpenAI-like", "OpenAILike"),
        ("ZAI", "ZAI"),
        ("Z.AI", "ZAI"),
        ("litellm", "LiteLLM"),
        ("lite_llm", "LiteLLM"),
        ("lite-llm", "LiteLLM"),
    ],
)
def test_normalize_provider_name_accepts_user_facing_aliases(
    alias: str, expected: str
) -> None:
    assert normalize_provider_name(alias) == expected


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.5",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    ],
)
def test_openai_responses_current_reasoning_models_omit_sampling_params(
    model: str,
) -> None:
    llm = load_llm(
        "OpenAIResponses",
        model=model,
        api_key="stub",
        temperature=0.4,
    )

    kwargs = llm._get_model_kwargs()

    assert kwargs["model"] == model
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.5",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    ],
)
def test_openai_structured_predict_omits_per_call_sampling_params(
    model: str,
) -> None:
    from llama_index.core.prompts import PromptTemplate
    from pydantic import BaseModel

    class StructuredResult(BaseModel):
        value: str

    sync_payload: dict[str, Any] = {}
    async_payload: dict[str, Any] = {}

    def parse_sync(**kwargs: Any) -> Any:
        sync_payload.update(kwargs)
        return SimpleNamespace(output_parsed=StructuredResult(value="OK"))

    async def parse_async(**kwargs: Any) -> Any:
        async_payload.update(kwargs)
        return SimpleNamespace(output_parsed=StructuredResult(value="OK"))

    llm = load_llm("OpenAIResponses", model=model, api_key="stub")
    llm._client = SimpleNamespace(responses=SimpleNamespace(parse=parse_sync))
    llm._aclient = SimpleNamespace(responses=SimpleNamespace(parse=parse_async))
    prompt = PromptTemplate("Return {value}")
    call_kwargs = {
        "temperature": 0.4,
        "top_p": 0.6,
        "max_output_tokens": 32,
    }

    sync_result = llm.structured_predict(
        StructuredResult,
        prompt,
        llm_kwargs=dict(call_kwargs),
        value="OK",
    )
    async_result = asyncio.run(
        llm.astructured_predict(
            StructuredResult,
            prompt,
            llm_kwargs=dict(call_kwargs),
            value="OK",
        )
    )

    assert sync_result.value == "OK"
    assert async_result.value == "OK"
    for payload in (sync_payload, async_payload):
        assert {"temperature", "top_p"}.isdisjoint(payload)
        assert payload["max_output_tokens"] == 32


@pytest.mark.parametrize(
    ("model", "context_window"),
    [
        ("gpt-5.5", 1_050_000),
        ("gpt-5.6-sol", 1_050_000),
        ("gpt-5.6-terra", 1_050_000),
        ("gpt-5.6-luna", 1_050_000),
        ("gpt-5.4", 1_050_000),
        ("gpt-5.4-mini", 400_000),
        ("gpt-5.4-nano", 400_000),
    ],
)
def test_openai_responses_current_catalog_models_have_metadata(
    model: str, context_window: int
) -> None:
    llm = load_llm(
        "OpenAIResponses",
        model=model,
        api_key="stub",
    )

    metadata = llm.metadata

    assert metadata.model_name == model
    assert metadata.context_window == context_window
    assert metadata.is_function_calling_model is True


def test_openai_responses_preserves_explicit_context_window_override() -> None:
    llm = load_llm(
        "OpenAIResponses",
        model="gpt-5.5",
        api_key="stub",
        context_window=123_456,
    )

    assert llm.metadata.context_window == 123_456


def test_openai_alias_loads_openai_responses_without_temperature_for_gpt_5_5() -> None:
    llm = load_llm(
        "OpenAI",
        model="gpt-5.5",
        api_key="stub",
        temperature=0.4,
    )

    assert type(llm).__name__ == "MobilerunOpenAIResponses"
    assert llm.metadata.context_window == 1_050_000
    assert "temperature" not in llm._get_model_kwargs()
    assert "top_p" not in llm._get_model_kwargs()


def test_openai_responses_profile_loads_with_current_default_metadata() -> None:
    llm = load_llms_from_profiles(
        {
            "manager": LLMProfile(
                provider="OpenAIResponses",
                model="gpt-5.5",
                kwargs={"api_key": "stub"},
            )
        }
    )["manager"]

    assert type(llm).__name__ == "MobilerunOpenAIResponses"
    assert llm.metadata.model_name == "gpt-5.5"
    assert llm.metadata.context_window == 1_050_000


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("OpenAIResponses", "gpt-5.6"),
        ("OpenAIResponses", "openai/gpt-5.6"),
        ("OpenAI", "gpt-5.6"),
    ],
)
def test_openai_responses_aliases_load_canonical_sol_model(
    provider: str, model: str
) -> None:
    llm = load_llm(provider, model=model, api_key="stub")

    assert llm.model == "gpt-5.6-sol"
    assert llm.metadata.model_name == "gpt-5.6-sol"
    assert llm.metadata.context_window == 1_050_000


def test_zai_alias_uses_openai_like_transport_defaults() -> None:
    llm = load_llm(
        "ZAI",
        model="glm-5",
        api_key="stub",
    )

    assert type(llm).__name__ == "OpenAILike"
    assert llm.api_base == "https://api.z.ai/api/paas/v4"


def test_openai_oauth_rejects_unsupported_codex_model() -> None:
    with pytest.raises(ValueError, match="not supported with OpenAI OAuth"):
        load_llm("openai_oauth", model="gpt-5.3-codex")


@pytest.mark.parametrize(
    "model",
    [
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
    ],
)
def test_gemini_oauth_rejects_public_api_model_ids(model: str) -> None:
    with pytest.raises(ValueError, match="Gemini Developer API id"):
        load_llm("gemini_oauth_code_assist", model=model)


@pytest.mark.parametrize(
    "model",
    ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
)
def test_gemini_oauth_allows_live_unadvertised_2_5_ids(model: str) -> None:
    llm = load_llm(
        "gemini_oauth_code_assist",
        model=model,
        access_token="stub",
        credential_path=None,
    )

    assert llm.model == model


@pytest.mark.parametrize("model", ["gemini-3.6-flash", "gemini-3.5-flash-lite"])
def test_new_google_genai_models_omit_sampling_configuration(model: str) -> None:
    from google.genai import types

    llm = load_llm(
        "GoogleGenAI",
        model=model,
        api_key="stub",
        max_tokens=64,
        context_window=1_000_000,
        temperature=0.4,
        generation_config=types.GenerateContentConfig(
            temperature=0.6,
            top_p=0.8,
            top_k=20,
            max_output_tokens=64,
        ),
    )

    assert type(llm).__name__ == "MobilerunGoogleGenAI"
    assert llm._generation_config["max_output_tokens"] == 64
    assert {"temperature", "top_p", "top_k"}.isdisjoint(llm._generation_config)

    call_kwargs = llm._sanitize_call_kwargs(
        {
            "temperature": 0.5,
            "top_p": 0.7,
            "top_k": 10,
            "generation_config": {
                "temperature": 0.4,
                "top_p": 0.6,
                "top_k": 5,
                "max_output_tokens": 32,
            },
        }
    )
    assert {"temperature", "top_p", "top_k"}.isdisjoint(call_kwargs)
    assert call_kwargs["generation_config"] == {"max_output_tokens": 32}


class _AsyncChunks:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> "_AsyncChunks":
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


class _StructuredModelsCapture:
    def __init__(self, output: Any, calls: list[dict[str, Any]]) -> None:
        self._output = output
        self._calls = calls

    def generate_content(self, **kwargs: Any) -> Any:
        self._calls.append(kwargs)
        return SimpleNamespace(
            parsed=self._output,
            text=self._output.model_dump_json(),
        )

    def generate_content_stream(self, **kwargs: Any) -> Any:
        self._calls.append(kwargs)
        return iter([SimpleNamespace(parsed=self._output, candidates=[])])


class _AsyncStructuredModelsCapture:
    def __init__(self, output: Any, calls: list[dict[str, Any]]) -> None:
        self._output = output
        self._calls = calls

    async def generate_content(self, **kwargs: Any) -> Any:
        self._calls.append(kwargs)
        return SimpleNamespace(
            parsed=self._output,
            text=self._output.model_dump_json(),
        )

    async def generate_content_stream(self, **kwargs: Any) -> Any:
        self._calls.append(kwargs)
        return _AsyncChunks([SimpleNamespace(parsed=self._output, candidates=[])])


def _google_structured_llm_with_capture() -> tuple[Any, Any, Any, list[dict[str, Any]]]:
    from llama_index.core.prompts import PromptTemplate
    from pydantic import BaseModel

    class StructuredResult(BaseModel):
        value: str

    output = StructuredResult(value="OK")
    calls: list[dict[str, Any]] = []
    llm = load_llm(
        "GoogleGenAI",
        model="gemini-3.5-flash-lite",
        api_key="stub",
        max_tokens=64,
        context_window=1_000_000,
        temperature=0.4,
        file_mode="inline",
    )
    llm._client = SimpleNamespace(
        models=_StructuredModelsCapture(output, calls),
        aio=SimpleNamespace(models=_AsyncStructuredModelsCapture(output, calls)),
    )
    return llm, StructuredResult, PromptTemplate("Return {value}"), calls


def _sampling_llm_kwargs() -> dict[str, Any]:
    return {
        "temperature": 0.5,
        "top_p": 0.7,
        "top_k": 10,
        "generation_config": {
            "temperature": 0.4,
            "top_p": 0.6,
            "top_k": 5,
            "max_output_tokens": 32,
        },
    }


def _assert_google_request_omits_sampling(request: dict[str, Any]) -> None:
    sampling_params = {"temperature", "top_p", "top_k"}
    assert sampling_params.isdisjoint(request)
    assert sampling_params.isdisjoint(request["config"])


def test_google_direct_structured_path_strips_sampling_payload() -> None:
    llm, output_cls, prompt, calls = _google_structured_llm_with_capture()

    result = llm.structured_predict_without_function_calling(
        output_cls,
        prompt,
        llm_kwargs={"temperature": 0.5, "top_p": 0.7, "top_k": 10},
        value="OK",
    )

    assert result.value == "OK"
    _assert_google_request_omits_sampling(calls[-1])


@pytest.mark.parametrize(
    ("method_name", "streaming"),
    [
        ("structured_predict", False),
        ("stream_structured_predict", True),
    ],
)
def test_google_sync_structured_paths_strip_sampling_payload(
    method_name: str, streaming: bool
) -> None:
    llm, output_cls, prompt, calls = _google_structured_llm_with_capture()

    result = getattr(llm, method_name)(
        output_cls,
        prompt,
        llm_kwargs=_sampling_llm_kwargs(),
        value="OK",
    )
    if streaming:
        result = list(result)[-1]

    assert result.value == "OK"
    _assert_google_request_omits_sampling(calls[-1])
    assert calls[-1]["config"]["max_output_tokens"] == 32


def test_google_async_structured_paths_strip_sampling_payload() -> None:
    async def run() -> None:
        llm, output_cls, prompt, calls = _google_structured_llm_with_capture()

        result = await llm.astructured_predict(
            output_cls,
            prompt,
            llm_kwargs=_sampling_llm_kwargs(),
            value="OK",
        )

        assert result.value == "OK"
        _assert_google_request_omits_sampling(calls[-1])
        assert calls[-1]["config"]["max_output_tokens"] == 32

        result_stream = await llm.astream_structured_predict(
            output_cls,
            prompt,
            llm_kwargs=_sampling_llm_kwargs(),
            value="OK",
        )
        streamed = [result async for result in result_stream]

        assert streamed[-1].value == "OK"
        _assert_google_request_omits_sampling(calls[-1])
        assert calls[-1]["config"]["max_output_tokens"] == 32

    asyncio.run(run())


def test_existing_google_genai_model_keeps_sampling_configuration() -> None:
    llm = load_llm(
        "GoogleGenAI",
        model="gemini-3.5-flash",
        api_key="stub",
        max_tokens=64,
        context_window=1_000_000,
        temperature=0.4,
    )

    assert llm._generation_config["temperature"] == 0.4


def test_explicit_retired_default_google_profile_remains_loadable() -> None:
    llm = load_llms_from_profiles(
        {
            "manager": LLMProfile(
                provider="GoogleGenAI",
                model="gemini-3.1-flash-lite",
                temperature=0.2,
                kwargs={
                    "api_key": "stub",
                    "max_tokens": 64,
                    "context_window": 1_000_000,
                },
            )
        }
    )["manager"]

    assert llm.model == "gemini-3.1-flash-lite"


def test_gemini_oauth_supported_choices_come_from_registry() -> None:
    with pytest.raises(ValueError, match="gemini-3.6-flash-high"):
        load_llm("gemini_oauth_code_assist", model="gemini-3.5-flash")


@pytest.mark.parametrize("model", ["claude-opus-4-8"])
def test_anthropic_opus_4_omits_default_temperature(model: str) -> None:
    llm = load_llm(
        "Anthropic",
        model=model,
        api_key="stub",
        temperature=0.2,
    )

    kwargs = llm._get_all_kwargs()

    assert type(llm).__name__ == "MobilerunAnthropic"
    assert kwargs["model"] == model
    assert "temperature" not in kwargs


def test_anthropic_opus_4_strips_explicit_additional_sampling() -> None:
    llm = load_llm(
        "Anthropic",
        model="claude-opus-4-8",
        api_key="stub",
        temperature=0.2,
        additional_kwargs={"temperature": 0.0, "top_p": 0.5, "top_k": 10},
    )

    assert {"temperature", "top_p", "top_k"}.isdisjoint(llm._get_all_kwargs())


def test_anthropic_opus_4_strips_per_call_sampling() -> None:
    llm = load_llm(
        "Anthropic",
        model="claude-opus-4-8",
        api_key="stub",
        temperature=0.2,
    )

    kwargs = llm._get_all_kwargs(temperature=0.0, top_p=0.5, top_k=10)

    assert {"temperature", "top_p", "top_k"}.isdisjoint(kwargs)


def test_anthropic_opus_4_6_keeps_supported_sampling() -> None:
    llm = load_llm(
        "Anthropic",
        model="claude-opus-4-6",
        api_key="stub",
        temperature=0.2,
        additional_kwargs={"top_p": 0.6},
    )

    kwargs = llm._get_all_kwargs(top_k=10)

    assert kwargs["temperature"] == 0.2
    assert kwargs["top_p"] == 0.6
    assert kwargs["top_k"] == 10


def test_anthropic_sonnet_keeps_temperature() -> None:
    llm = load_llm(
        "Anthropic",
        model="claude-sonnet-4-6",
        api_key="stub",
        temperature=0.2,
    )

    kwargs = llm._get_all_kwargs()

    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["temperature"] == 0.2


def test_anthropic_uses_a_2048_token_default() -> None:
    llm = load_llm(
        "Anthropic",
        model="claude-haiku-4-5",
        api_key="stub",
    )

    assert llm.max_tokens == 2048
    assert llm.metadata.num_output == 2048


def test_anthropic_profile_uses_the_shared_2048_token_default() -> None:
    llm = load_llms_from_profiles(
        {
            "manager": LLMProfile(
                provider="Anthropic",
                model="claude-haiku-4-5",
                kwargs={"api_key": "stub"},
            )
        }
    )["manager"]

    assert llm.max_tokens == 2048


@pytest.mark.parametrize("max_tokens", [512, 4096])
def test_anthropic_preserves_explicit_max_tokens(max_tokens: int) -> None:
    llm = load_llm(
        "Anthropic",
        model="claude-haiku-4-5",
        api_key="stub",
        max_tokens=max_tokens,
    )

    assert llm.max_tokens == max_tokens
    assert llm.metadata.num_output == max_tokens


@pytest.mark.parametrize(
    ("model", "context_window"),
    [
        ("claude-opus-5", 1_000_000),
        ("claude-sonnet-5", 1_000_000),
        ("claude-fable-5", 1_000_000),
        ("claude-opus-4-8", 1_000_000),
        ("claude-sonnet-4-6", 1_000_000),
        ("claude-opus-4-6", 1_000_000),
        ("claude-haiku-4-5", 200_000),
    ],
)
def test_anthropic_current_catalog_models_have_metadata(
    model: str, context_window: int
) -> None:
    llm = load_llm(
        "Anthropic",
        model=model,
        api_key="stub",
    )

    metadata = llm.metadata

    assert metadata.model_name == model
    assert metadata.context_window == context_window


@pytest.mark.parametrize(
    "model",
    ["claude-opus-5", "claude-sonnet-5", "claude-fable-5"],
)
def test_anthropic_claude_5_models_strip_all_sampling_overrides(model: str) -> None:
    llm = load_llm(
        "Anthropic",
        model=model,
        api_key="stub",
        temperature=0.8,
        additional_kwargs={"temperature": 0.7, "top_p": 0.6, "top_k": 20},
    )

    kwargs = llm._get_all_kwargs(temperature=0.5, top_p=0.4, top_k=10)

    assert kwargs["model"] == model
    assert {"temperature", "top_p", "top_k"}.isdisjoint(kwargs)


def test_anthropic_sampling_filter_uses_effective_per_call_model() -> None:
    llm = load_llm(
        "Anthropic",
        model="claude-sonnet-4-6",
        api_key="stub",
        temperature=0.2,
    )

    kwargs = llm._get_all_kwargs(
        model="claude-sonnet-5",
        temperature=0.5,
        top_p=0.4,
        top_k=10,
    )

    assert kwargs["model"] == "claude-sonnet-5"
    assert {"temperature", "top_p", "top_k"}.isdisjoint(kwargs)


# --- Ollama kwarg translation (max_tokens / context_window) ------------------


@pytest.fixture
def mobilerun_caplog(caplog):
    """caplog wired to the non-propagating "mobilerun" logger."""
    logger = logging.getLogger("mobilerun")
    previous = logger.propagate
    logger.propagate = True
    caplog.set_level(logging.WARNING, logger="mobilerun")
    yield caplog
    logger.propagate = previous


def _ollama_class():
    from llama_index.llms.ollama import Ollama

    return Ollama


def _prepare(kwargs):
    from mobilerun.agent.utils.llm_picker import _prepare_ollama_kwargs

    return _prepare_ollama_kwargs(kwargs, _ollama_class())


def test_ollama_max_tokens_translates_to_num_predict() -> None:
    out = _prepare({"model": "qwen3:0.6b", "max_tokens": 2048})

    assert "max_tokens" not in out
    assert out["additional_kwargs"]["num_predict"] == 2048


def test_ollama_explicit_num_predict_wins_over_max_tokens(mobilerun_caplog) -> None:
    out = _prepare(
        {
            "model": "qwen3:0.6b",
            "max_tokens": 2048,
            "additional_kwargs": {"num_predict": 512},
        }
    )

    assert out["additional_kwargs"]["num_predict"] == 512
    assert any("num_predict wins" in r.message for r in mobilerun_caplog.records)


def test_ollama_equal_num_predict_and_max_tokens_no_warning(mobilerun_caplog) -> None:
    out = _prepare(
        {
            "model": "qwen3:0.6b",
            "max_tokens": 512,
            "additional_kwargs": {"num_predict": 512},
        }
    )

    assert out["additional_kwargs"]["num_predict"] == 512
    assert not any("num_predict wins" in r.message for r in mobilerun_caplog.records)


def test_ollama_numeric_string_max_tokens_is_converted() -> None:
    out = _prepare({"model": "qwen3:0.6b", "max_tokens": "1024"})

    assert out["additional_kwargs"]["num_predict"] == 1024


@pytest.mark.parametrize("bad", ["lots", True, None])
def test_ollama_invalid_max_tokens_warns_and_skips(bad, mobilerun_caplog) -> None:
    out = _prepare({"model": "qwen3:0.6b", "max_tokens": bad})

    assert "max_tokens" not in out
    assert "num_predict" not in out.get("additional_kwargs", {})
    assert any(
        "Ignoring non-integer max_tokens" in r.message for r in mobilerun_caplog.records
    )


def test_ollama_context_window_defaults_to_32k() -> None:
    out = _prepare({"model": "qwen3:0.6b"})

    assert out["context_window"] == 32768


@pytest.mark.parametrize("explicit", [8192, -1])
def test_ollama_explicit_context_window_is_preserved(explicit) -> None:
    out = _prepare({"model": "qwen3:0.6b", "context_window": explicit})

    assert out["context_window"] == explicit


def test_ollama_num_ctx_mirrors_into_context_window() -> None:
    out = _prepare({"model": "qwen3:0.6b", "additional_kwargs": {"num_ctx": 16384}})

    assert out["context_window"] == 16384
    assert out["additional_kwargs"]["num_ctx"] == 16384


def test_ollama_non_numeric_num_ctx_falls_back_to_default() -> None:
    out = _prepare({"model": "qwen3:0.6b", "additional_kwargs": {"num_ctx": "max"}})

    assert out["context_window"] == 32768


def test_ollama_unknown_kwarg_warns_once(mobilerun_caplog) -> None:
    from mobilerun.agent.utils import llm_picker

    llm_picker._warned_ollama_kwargs.discard("frobnicate")
    _prepare({"model": "qwen3:0.6b", "frobnicate": 1})
    _prepare({"model": "qwen3:0.6b", "frobnicate": 1})

    warnings = [r for r in mobilerun_caplog.records if "'frobnicate'" in r.message]
    assert len(warnings) == 1


def test_ollama_translation_disabled_if_class_grows_max_tokens_field() -> None:
    from mobilerun.agent.utils.llm_picker import _prepare_ollama_kwargs

    class FakeOllama:
        model_fields = {"model": None, "max_tokens": None, "context_window": None}

    out = _prepare_ollama_kwargs({"model": "m", "max_tokens": 99}, FakeOllama)

    assert out["max_tokens"] == 99
    assert "additional_kwargs" not in out


def test_load_llm_ollama_end_to_end_applies_translation() -> None:
    llm = load_llm("Ollama", model="qwen3:0.6b", max_tokens=256)

    assert llm.context_window == 32768
    assert llm.additional_kwargs["num_predict"] == 256
    assert not hasattr(llm, "max_tokens") or "max_tokens" not in type(llm).model_fields


def test_ollama_wizard_default_includes_context_window() -> None:
    from mobilerun.agent.providers.setup_service import DEFAULT_KWARGS_BY_VARIANT

    assert DEFAULT_KWARGS_BY_VARIANT["Ollama"] == {"context_window": 32768}


def test_litellm_loads_correct_class() -> None:
    llm = load_llm(
        "LiteLLM",
        model="anthropic/claude-haiku-4-5",
        api_key="stub",
    )

    assert type(llm).__name__ == "MobilerunLiteLLM"


def test_litellm_alias_resolves() -> None:
    llm = load_llm(
        "litellm",
        model="openai/gpt-4o-mini",
        api_key="stub",
    )

    assert type(llm).__name__ == "MobilerunLiteLLM"


def test_litellm_base_url_translates_to_api_base() -> None:
    llm = load_llm(
        "LiteLLM",
        model="openai/gpt-4o-mini",
        api_key="stub",
        base_url="http://localhost:4000",
    )

    assert type(llm).__name__ == "MobilerunLiteLLM"
    assert llm.api_base == "http://localhost:4000"


def test_litellm_provider_slash_model_format() -> None:
    llm = load_llm("LiteLLM", model="anthropic/claude-sonnet-4-6", api_key="stub")

    assert llm.model == "anthropic/claude-sonnet-4-6"


def test_litellm_bare_model_name_preserved() -> None:
    llm = load_llm("LiteLLM", model="gpt-4o-mini", api_key="stub")

    assert llm.model == "gpt-4o-mini"


def test_litellm_api_key_forwarded() -> None:
    llm = load_llm("LiteLLM", model="openai/gpt-4o-mini", api_key="sk-test-key-123")

    assert llm.api_key == "sk-test-key-123"


def test_litellm_temperature_forwarded() -> None:
    llm = load_llm(
        "LiteLLM", model="openai/gpt-4o-mini", api_key="stub", temperature=0.7
    )

    assert llm.temperature == 0.7


def test_litellm_max_tokens_forwarded() -> None:
    llm = load_llm(
        "LiteLLM", model="openai/gpt-4o-mini", api_key="stub", max_tokens=512
    )

    assert llm.max_tokens == 512


def test_litellm_unsupported_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported provider"):
        load_llm("NotARealProvider", model="some-model", api_key="stub")
