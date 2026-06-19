import pytest

from mobilerun.agent.utils.llm_picker import load_llm, normalize_provider_name


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
    ],
)
def test_normalize_provider_name_accepts_user_facing_aliases(
    alias: str, expected: str
) -> None:
    assert normalize_provider_name(alias) == expected


def test_openai_responses_omits_temperature_for_gpt_5_5() -> None:
    llm = load_llm(
        "OpenAIResponses",
        model="gpt-5.5",
        api_key="stub",
        temperature=0.4,
    )

    kwargs = llm._get_model_kwargs()

    assert kwargs["model"] == "gpt-5.5"
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


def test_openai_responses_keeps_temperature_for_gpt_5_4() -> None:
    llm = load_llm(
        "OpenAIResponses",
        model="gpt-5.4",
        api_key="stub",
        temperature=0.4,
    )

    kwargs = llm._get_model_kwargs()

    assert kwargs["model"] == "gpt-5.4"
    assert kwargs["temperature"] == 0.4
    assert kwargs["top_p"] == 1.0


def test_openai_alias_loads_openai_responses_without_temperature_for_gpt_5_5() -> None:
    llm = load_llm(
        "OpenAI",
        model="gpt-5.5",
        api_key="stub",
        temperature=0.4,
    )

    assert type(llm).__name__ == "MobilerunOpenAIResponses"
    assert "temperature" not in llm._get_model_kwargs()
    assert "top_p" not in llm._get_model_kwargs()


def test_zai_alias_uses_openai_like_transport_defaults() -> None:
    llm = load_llm(
        "ZAI",
        model="glm-5",
        api_key="stub",
    )

    assert type(llm).__name__ == "OpenAILike"
    assert llm.api_base == "https://api.z.ai/api/paas/v4"


@pytest.mark.parametrize("model", ["claude-opus-4-8", "claude-opus-4-6"])
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


def test_anthropic_opus_4_keeps_explicit_additional_temperature() -> None:
    llm = load_llm(
        "Anthropic",
        model="claude-opus-4-8",
        api_key="stub",
        temperature=0.2,
        additional_kwargs={"temperature": 0.0},
    )

    assert llm._get_all_kwargs()["temperature"] == 0.0


def test_anthropic_opus_4_keeps_per_call_temperature() -> None:
    llm = load_llm(
        "Anthropic",
        model="claude-opus-4-8",
        api_key="stub",
        temperature=0.2,
    )

    assert llm._get_all_kwargs(temperature=0.0)["temperature"] == 0.0


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


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-haiku-4-5",
    ],
)
def test_anthropic_current_catalog_models_have_metadata(model: str) -> None:
    llm = load_llm(
        "Anthropic",
        model=model,
        api_key="stub",
    )

    metadata = llm.metadata

    assert metadata.model_name == model
    assert metadata.context_window > 0


# --- Ollama kwarg translation (max_tokens / context_window) ------------------


import logging


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
    assert any("Ignoring non-integer max_tokens" in r.message for r in mobilerun_caplog.records)


def test_ollama_context_window_defaults_to_32k() -> None:
    out = _prepare({"model": "qwen3:0.6b"})

    assert out["context_window"] == 32768


@pytest.mark.parametrize("explicit", [8192, -1])
def test_ollama_explicit_context_window_is_preserved(explicit) -> None:
    out = _prepare({"model": "qwen3:0.6b", "context_window": explicit})

    assert out["context_window"] == explicit


def test_ollama_num_ctx_mirrors_into_context_window() -> None:
    out = _prepare(
        {"model": "qwen3:0.6b", "additional_kwargs": {"num_ctx": 16384}}
    )

    assert out["context_window"] == 16384
    assert out["additional_kwargs"]["num_ctx"] == 16384


def test_ollama_non_numeric_num_ctx_falls_back_to_default() -> None:
    out = _prepare(
        {"model": "qwen3:0.6b", "additional_kwargs": {"num_ctx": "max"}}
    )

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
