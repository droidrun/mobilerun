"""Default selection and reasoning at the provider SDK request boundary."""

import asyncio
import json
from dataclasses import asdict

import pytest

try:
    from openai._base_client import httpx2 as httpx
except ImportError:
    import httpx
from llama_index.core.llms import ChatMessage

from mobilerun.agent.providers.registry import resolve_provider_variant
from mobilerun.agent.providers.setup_service import (
    SetupSelection,
    create_profile_for_variant,
)
from mobilerun.agent.utils.llm_picker import load_llm


def _credentials(family, auth, tmp_path):
    if auth == "api_key":
        if family == "gemini":
            # Avoid the adapter's model-catalog request in an offline test.
            return {"api_key": "stub", "context_window": 1_000_000, "max_tokens": 1024}
        return {"api_key": "stub"}
    path = str(tmp_path / "credentials.json")
    if family == "openai":
        return {
            "oauth_access_token": "stub",
            "oauth_expires_at_ms": 4_102_444_800_000,
            "oauth_credential_path": path,
        }
    return {"credential_path": path}


@pytest.mark.parametrize(
    "family,auth,expected",
    [
        ("openai", "api_key", "gpt-6-astra"),
        ("openai", "oauth", "gpt-6-astra"),
        ("gemini", "api_key", "gemini-3.8-flash"),
        ("gemini", "oauth", "gemini-3.8-flash-tiered"),
        ("anthropic", "api_key", "claude-sonnet-5"),
        ("anthropic", "oauth", "claude-sonnet-5"),
    ],
)
def test_defaults_agree_across_menu_loader_and_generated_profile(
    family, auth, expected, tmp_path
):
    variant = resolve_provider_variant(family, auth)
    assert variant.default_model == variant.models[0] == expected
    assert len(variant.models) == len(set(variant.models))
    credentials = _credentials(family, auth, tmp_path)
    implicit = load_llm(variant.runtime_provider_name, **credentials)
    assert implicit.model == expected
    profile = create_profile_for_variant(
        variant,
        SetupSelection(
            family_id=family,
            variant_id=variant.id,
            auth_mode=auth,
            model=variant.default_model,
            credential_path=str(tmp_path / "credentials.json"),
        ),
    )
    restored = type(profile)(**asdict(profile))
    assert restored.model == expected
    llm = load_llm(
        restored.provider, model=restored.model, **{**restored.kwargs, **credentials}
    )
    assert llm.model == expected
    if family == "openai":
        payload = (
            llm._get_model_kwargs()
            if auth == "api_key"
            else llm._sanitize_reasoning_kwargs({})
        )
        assert payload["reasoning"] == {"effort": "low"}


@pytest.mark.parametrize("auth", ["api_key", "oauth"])
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    "source,expected",
    [
        ("default", {"effort": "low"}),
        ("constructor", {"effort": "high"}),
        ("additional", {"effort": "xhigh"}),
        ("runtime", {"effort": "max"}),
        ("runtime_null", None),
        ("runtime_summary", {"summary": "auto"}),
    ],
)
def test_openai_default_and_explicit_reasoning_in_sdk_body(
    auth, async_call, source, expected, tmp_path
):
    bodies = []

    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        response = {
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "model": "gpt-6-astra",
            "status": "completed",
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
            "output": [
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "READY", "annotations": []}
                    ],
                }
            ],
        }
        if body.get("stream"):
            events = [
                {"type": "response.output_text.delta", "delta": "READY"},
                {"type": "response.completed", "response": response},
            ]
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
            )
        return httpx.Response(200, json=response)

    options = _credentials("openai", auth, tmp_path)
    if source != "default":
        if auth == "api_key":
            options["reasoning_options"] = {"effort": "high"}
        else:
            options["reasoning_effort"] = "high"
    if source not in {"default", "constructor"}:
        options["additional_kwargs"] = {"reasoning": {"effort": "xhigh"}}
    runtime = {"reasoning": expected} if source.startswith("runtime") else {}
    sync_client = httpx.Client(transport=httpx.MockTransport(respond))
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    llm = load_llm(
        resolve_provider_variant("openai", auth).runtime_provider_name,
        http_client=sync_client,
        async_http_client=async_client,
        **options,
    )
    messages = [ChatMessage(role="user", content="Reply READY")]

    async def run():
        try:
            if async_call:
                return await llm.achat(messages, **runtime)
            return llm.chat(messages, **runtime)
        finally:
            sync_client.close()
            await async_client.aclose()

    result = asyncio.run(run())
    assert result.message.content == "READY"
    assert len(bodies) == 1
    assert bodies[0]["model"] == "gpt-6-astra"
    assert bodies[0].get("reasoning") == expected
    assert {"temperature", "top_p", "logprobs", "top_logprobs"}.isdisjoint(bodies[0])


@pytest.mark.parametrize("auth", ["api_key", "oauth"])
@pytest.mark.parametrize("model", ["gpt-5.5", "gpt-5.6-luna"])
def test_previous_openai_models_have_no_new_reasoning_default(auth, model, tmp_path):
    llm = load_llm(
        resolve_provider_variant("openai", auth).runtime_provider_name,
        model=model,
        **_credentials("openai", auth, tmp_path),
    )
    payload = (
        llm._get_model_kwargs()
        if auth == "api_key"
        else llm._sanitize_reasoning_kwargs({})
    )
    assert llm.model == model
    assert "reasoning" not in payload


@pytest.mark.parametrize("family", ["gemini", "anthropic", "openai"])
def test_login_helpers_use_current_oauth_defaults(family, tmp_path, monkeypatch):
    from mobilerun.cli import oauth_actions

    classes = {
        "gemini": oauth_actions.GeminiOAuthCodeAssistLLM,
        "anthropic": oauth_actions.AnthropicOAuthLLM,
        "openai": oauth_actions.OpenAIOAuth,
    }
    selected = []

    def login(self, **kwargs):
        selected.append(self.model)
        return "stub-token"

    monkeypatch.setattr(classes[family], "login", login)
    path = str(tmp_path / "credentials.json")
    if family == "gemini":
        monkeypatch.setattr(
            classes[family], "fetch_available_models", lambda *a, **kw: [{}]
        )
        monkeypatch.setattr(
            classes[family], "_persist_credentials", lambda *a, **kw: None
        )
        oauth_actions.run_gemini_oauth_login(path, None)
    elif family == "openai":
        oauth_actions.run_openai_oauth_login(path, None)
    else:
        oauth_actions.run_anthropic_oauth_setup(path)
    assert selected == [resolve_provider_variant(family, "oauth").default_model]
