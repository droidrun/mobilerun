import inspect
import logging
from functools import wraps
from typing import TYPE_CHECKING, Any

from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.llms.llm import LLM

from mobilerun.agent.providers.anthropic import (
    ANTHROPIC_API_DEFAULT_MODEL,
    ANTHROPIC_FABLE_5_1_MODEL,
    ANTHROPIC_UNSUPPORTED_SAMPLING_PARAMS,
    anthropic_model_context_window,
    anthropic_model_omits_sampling_params,
)
from mobilerun.agent.providers.grok import (
    GROK_CONTEXT_WINDOW,
    GROK_DEFAULT_MODEL,
    XAI_API_BASE,
    normalize_grok_model_id,
    sanitize_grok_responses_kwargs,
)
from mobilerun.agent.providers.minimax import (
    MINIMAX_GLOBAL_BASE_URL,
    warn_if_legacy_minimax_endpoint,
)
from mobilerun.agent.providers.registry import (
    GEMINI_API_DEFAULT_MODEL,
    OPENAI_API_DEFAULT_MODEL,
    OPENAI_ASTRA_DEFAULT_REASONING_EFFORT,
    OPENAI_OAUTH_UNSUPPORTED_MODELS,
    list_models_for_variant,
    normalize_model_id_for_variant,
)
from mobilerun.agent.providers.requesty import (
    REQUESTY_API_KEY_ENV_VAR,
    REQUESTY_DEFAULT_MODEL,
    resolve_requesty_base_url,
)
from mobilerun.agent.usage import track_usage

if TYPE_CHECKING:
    from mobilerun.config_manager.config_manager import LLMProfile

# Configure logging
logger = logging.getLogger("mobilerun")


SUPPORTED_PROVIDERS = [
    "OpenAIResponses",
    "OpenAILike",
    "GoogleGenAI",
    "Ollama",
    "Anthropic",
    "DeepSeek",
    "OpenRouter",
    "Requesty",
    "MiniMax",
    "XAI",
]

PROVIDER_ALIASES = {
    "openai": "OpenAIResponses",
    "gpt": "OpenAIResponses",
    "gemini": "GoogleGenAI",
    "google": "GoogleGenAI",
    "claude": "Anthropic",
    "openai compatible": "OpenAILike",
    "openai-like": "OpenAILike",
    "openai like": "OpenAILike",
    "openai_compatible": "OpenAILike",
    "openai_like": "OpenAILike",
    "zai": "ZAI",
    "z.ai": "ZAI",
    "xai": "XAI",
    "requesty": "Requesty",
}

ZAI_GLOBAL_API_BASE = "https://api.z.ai/api/paas/v4"
# Public Gemini Developer API ids that are not valid on the private
# Antigravity OAuth catalog. Catch these locally so users get the applicable
# OAuth choices instead of a remote model-not-found response.
GEMINI_OAUTH_UNSUPPORTED_MODELS = {
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
}
OPENAI_RESPONSES_MODELS_WITHOUT_SAMPLING_PARAMS = {
    "gpt-6-astra",
    "gpt-5.5",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
}
OPENAI_RESPONSES_UNSUPPORTED_SAMPLING_PARAMS = {"temperature", "top_p"}
OPENAI_ASTRA_MODEL = "gpt-6-astra"
OPENAI_ASTRA_CONTEXT_WINDOW = 1_050_000
OPENAI_ASTRA_UNSUPPORTED_PARAMS = {"logprobs", "top_logprobs"}
OPENAI_ASTRA_UNSUPPORTED_INCLUDE = "message.output_text.logprobs"
OPENAI_ASTRA_REASONING_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
GOOGLE_GENAI_MODELS_WITHOUT_SAMPLING_PARAMS = {
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
}
GOOGLE_GENAI_UNSUPPORTED_SAMPLING_PARAMS = {"temperature", "top_p", "top_k"}


def normalize_provider_name(provider_name: str) -> str:
    """Map user-facing provider names to Mobilerun runtime providers."""
    stripped = provider_name.strip()
    key = stripped.lower()
    return PROVIDER_ALIASES.get(key, stripped)


def _openai_responses_model_omits_sampling_params(model: object) -> bool:
    return str(model or "").strip() in OPENAI_RESPONSES_MODELS_WITHOUT_SAMPLING_PARAMS


def _validate_openai_oauth_model(model: object) -> None:
    model_id = str(model or "").strip()
    if model_id in OPENAI_OAUTH_UNSUPPORTED_MODELS:
        supported = ", ".join(list_models_for_variant("openai", "oauth"))
        raise ValueError(
            f"Model '{model_id}' is not supported with OpenAI OAuth "
            f"ChatGPT-account credentials. Use {supported}."
        )


def _validate_gemini_oauth_model(model: object) -> None:
    model_id = str(model or "").strip()
    if model_id in GEMINI_OAUTH_UNSUPPORTED_MODELS:
        supported = ", ".join(list_models_for_variant("gemini", "oauth"))
        raise ValueError(
            f"Model '{model_id}' is a Gemini Developer API id and is not valid "
            f"for the Antigravity OAuth catalog. Re-run "
            f"`mobilerun configure gemini` and pick one of: {supported}."
        )


# Default Ollama context size. llama-index's own default (-1) resolves to the
# model's maximum context, which allocates the full KV cache up front (e.g. a
# 256K-context model -> ~19 GB, spilling to CPU) — and because mobilerun sends
# num_ctx per request, it overrides every Ollama-side setting
# (OLLAMA_CONTEXT_LENGTH, Modelfile, /set parameter), so users cannot fix it
# server-side. ``context_window: -1`` in profile kwargs restores model-max.
_OLLAMA_DEFAULT_CONTEXT_WINDOW = 32768

_warned_ollama_kwargs: set[str] = set()


def _prepare_ollama_kwargs(kwargs: dict[str, Any], llm_class: Any) -> dict[str, Any]:
    """Translate provider-portable kwargs for llama-index's Ollama class.

    ``max_tokens`` is not an Ollama constructor field and pydantic silently
    drops it; translate it to ``additional_kwargs.num_predict`` (an explicit
    ``num_predict`` wins). Also default ``context_window`` (see
    ``_OLLAMA_DEFAULT_CONTEXT_WINDOW``) and keep it aligned with an explicit
    ``additional_kwargs.num_ctx`` so the -1 path's hidden ``client.show()``
    lookup is never triggered by mobilerun defaults.
    """
    kwargs = dict(kwargs)
    additional_kwargs = dict(kwargs.get("additional_kwargs") or {})

    if "max_tokens" in kwargs and "max_tokens" not in llm_class.model_fields:
        max_tokens = kwargs.pop("max_tokens")
        if isinstance(max_tokens, bool) or max_tokens is None:
            valid_max_tokens = None
        else:
            try:
                valid_max_tokens = int(max_tokens)
            except (TypeError, ValueError):
                valid_max_tokens = None
        if valid_max_tokens is None:
            logger.warning(
                f"Ignoring non-integer max_tokens={max_tokens!r} for Ollama."
            )
        elif "num_predict" in additional_kwargs:
            if additional_kwargs["num_predict"] != valid_max_tokens:
                logger.warning(
                    f"Both max_tokens={valid_max_tokens} and "
                    f"additional_kwargs.num_predict="
                    f"{additional_kwargs['num_predict']} are set for Ollama; "
                    f"num_predict wins."
                )
        else:
            additional_kwargs["num_predict"] = valid_max_tokens

    if kwargs.get("context_window") is None:
        context_window = _OLLAMA_DEFAULT_CONTEXT_WINDOW
        if "num_ctx" in additional_kwargs:
            try:
                context_window = int(additional_kwargs["num_ctx"])
            except (TypeError, ValueError):
                pass
        kwargs["context_window"] = context_window

    if additional_kwargs:
        kwargs["additional_kwargs"] = additional_kwargs

    for key, value in kwargs.items():
        if value is None or key in llm_class.model_fields:
            continue
        if key not in _warned_ollama_kwargs:
            _warned_ollama_kwargs.add(key)
            logger.warning(
                f"Ollama does not accept the {key!r} option; it will be ignored."
            )

    return kwargs


def _load_openai_responses(*, grok: bool = False, **kwargs: Any) -> LLM:
    from llama_index.llms.openai.responses import OpenAIResponses
    from llama_index.llms.openai.utils import to_openai_message_dicts

    class MobilerunOpenAIResponses(OpenAIResponses):
        @staticmethod
        def _validate_astra_reasoning(reasoning: object) -> None:
            if reasoning is None:
                return
            if not isinstance(reasoning, dict):
                raise ValueError(
                    "gpt-6-astra reasoning must be a mapping with an optional "
                    "'effort' value."
                )
            effort = reasoning.get("effort")
            if effort is None:
                return
            if effort not in OPENAI_ASTRA_REASONING_EFFORTS:
                supported = ", ".join(sorted(OPENAI_ASTRA_REASONING_EFFORTS))
                raise ValueError(
                    f"gpt-6-astra does not support reasoning effort {effort!r}; "
                    f"use one of: {supported}."
                )

        @classmethod
        def _sanitize_astra_payload_fields(
            cls,
            payload: dict[str, Any],
            *,
            drop_model: bool = False,
        ) -> dict[str, Any]:
            sanitized = dict(payload)
            unsupported = (
                OPENAI_RESPONSES_UNSUPPORTED_SAMPLING_PARAMS
                | OPENAI_ASTRA_UNSUPPORTED_PARAMS
            )
            for param in unsupported:
                sanitized.pop(param, None)

            include = sanitized.get("include")
            if isinstance(include, (list, tuple)):
                filtered_include = [
                    value
                    for value in include
                    if value != OPENAI_ASTRA_UNSUPPORTED_INCLUDE
                ]
                sanitized["include"] = filtered_include or None

            cls._validate_astra_reasoning(sanitized.get("reasoning"))
            if drop_model:
                sanitized.pop("model", None)
            return sanitized

        def _sanitize_call_kwargs(
            self,
            call_kwargs: dict[str, Any],
            *,
            omit_tool_choice: bool = False,
        ) -> dict[str, Any]:
            sanitized = dict(call_kwargs)
            if grok:
                sanitized = dict(
                    sanitize_grok_responses_kwargs(
                        sanitized,
                        omit_tool_choice=omit_tool_choice,
                    )
                )
                # Runtime and additional kwargs are merged after constructor
                # defaults. Re-pin the canonical model after that final merge.
                sanitized["model"] = self.model
                return sanitized
            effective_model = sanitized.get("model", self.model)
            if _openai_responses_model_omits_sampling_params(effective_model):
                for param in OPENAI_RESPONSES_UNSUPPORTED_SAMPLING_PARAMS:
                    sanitized.pop(param, None)
            if (
                self.model == OPENAI_ASTRA_MODEL
                or effective_model == OPENAI_ASTRA_MODEL
            ):
                sanitized = self._sanitize_astra_payload_fields(sanitized)
                # Apply the default after configured and per-call overrides.
                sanitized.setdefault(
                    "reasoning", {"effort": OPENAI_ASTRA_DEFAULT_REASONING_EFFORT}
                )
                extra_body = sanitized.get("extra_body")
                if isinstance(extra_body, dict):
                    sanitized["extra_body"] = self._sanitize_astra_payload_fields(
                        extra_body,
                        drop_model=True,
                    )
                if self.model == OPENAI_ASTRA_MODEL:
                    sanitized["model"] = self.model
            return sanitized

        def _get_model_kwargs(self, **kwargs: Any) -> dict[str, Any]:
            model_kwargs = super()._get_model_kwargs(**kwargs)
            effective_model = model_kwargs.get("model", self.model)
            if (
                (
                    self.model == OPENAI_ASTRA_MODEL
                    or effective_model == OPENAI_ASTRA_MODEL
                )
                and "reasoning" not in model_kwargs
                and self.reasoning_options is not None
            ):
                model_kwargs["reasoning"] = self.reasoning_options
            return self._sanitize_call_kwargs(model_kwargs)

        @property
        def metadata(self) -> LLMMetadata:
            if self.model != OPENAI_ASTRA_MODEL:
                return super().metadata
            return LLMMetadata(
                context_window=self.context_window or OPENAI_ASTRA_CONTEXT_WINDOW,
                num_output=self.max_output_tokens or -1,
                is_chat_model=True,
                is_function_calling_model=True,
                model_name=self.model,
            )

        def _sanitize_structured_call_kwargs(
            self, call_kwargs: dict[str, Any]
        ) -> dict[str, Any]:
            merged = dict(call_kwargs)
            if not grok and self.model == OPENAI_ASTRA_MODEL:
                configured: dict[str, Any] = {}
                if self.include is not None:
                    configured["include"] = self.include
                if self.reasoning_options is not None:
                    configured["reasoning"] = self.reasoning_options
                configured.update(self.additional_kwargs or {})
                configured.update(merged)
                merged = configured

            sanitized = self._sanitize_call_kwargs(merged, omit_tool_choice=grok)
            if grok:
                # The upstream structured adapter passes ``store=self.store``
                # explicitly. The constructor already pins that field false.
                # xAI also rejects tool_choice when no tools are supplied, so
                # structured parsing must not allow either the upstream default
                # or a generic caller override to restore it.
                sanitized.pop("store", None)
                sanitized.pop("model", None)
                sanitized.pop("tool_choice", None)
            elif self.model == OPENAI_ASTRA_MODEL:
                # The upstream structured adapter supplies these explicitly.
                sanitized.pop("model", None)
                sanitized.pop("store", None)
                sanitized.pop("tool_choice", None)
            return sanitized

        def structured_predict(
            self,
            output_cls: Any,
            prompt: Any,
            llm_kwargs: dict[str, Any] | None = None,
            **prompt_args: Any,
        ) -> Any:
            if grok:
                messages = prompt.format_messages(**prompt_args)
                message_dicts = to_openai_message_dicts(
                    messages, model=self.model, is_responses_api=True
                )
                response = self._client.responses.parse(
                    model=self._responses_model,
                    input=message_dicts,
                    text_format=output_cls,
                    store=self.store,
                    **self._sanitize_structured_call_kwargs(dict(llm_kwargs or {})),
                )
                if response.output_parsed is not None:
                    return response.output_parsed
                raise ValueError(
                    "Failed to produce a structured response from the model."
                )
            return super().structured_predict(
                output_cls,
                prompt,
                llm_kwargs=self._sanitize_structured_call_kwargs(
                    dict(llm_kwargs or {})
                ),
                **prompt_args,
            )

        async def astructured_predict(
            self,
            output_cls: Any,
            prompt: Any,
            llm_kwargs: dict[str, Any] | None = None,
            **prompt_args: Any,
        ) -> Any:
            if grok:
                messages = prompt.format_messages(**prompt_args)
                message_dicts = to_openai_message_dicts(
                    messages, model=self.model, is_responses_api=True
                )
                response = await self._aclient.responses.parse(
                    model=self._responses_model,
                    input=message_dicts,
                    text_format=output_cls,
                    store=self.store,
                    **self._sanitize_structured_call_kwargs(dict(llm_kwargs or {})),
                )
                if response.output_parsed is not None:
                    return response.output_parsed
                raise ValueError(
                    "Failed to produce a structured response from the model."
                )
            return await super().astructured_predict(
                output_cls,
                prompt,
                llm_kwargs=self._sanitize_structured_call_kwargs(
                    dict(llm_kwargs or {})
                ),
                **prompt_args,
            )

    if grok:
        kwargs = dict(sanitize_grok_responses_kwargs(kwargs))
        additional_kwargs = dict(kwargs.get("additional_kwargs") or {})
        sanitize_grok_responses_kwargs(additional_kwargs)
        # ``store`` is a first-class adapter field; keeping a second copy in
        # additional_kwargs is unnecessary and makes intent harder to inspect.
        additional_kwargs.pop("store", None)
        kwargs["additional_kwargs"] = additional_kwargs

    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    logger.debug(
        "Initializing MobilerunOpenAIResponses with kwargs: "
        f"{list(filtered_kwargs.keys())}"
    )
    return MobilerunOpenAIResponses(**filtered_kwargs)


def _load_google_genai(**kwargs: Any) -> LLM:
    from llama_index.llms.google_genai import GoogleGenAI

    class MobilerunGoogleGenAI(GoogleGenAI):
        def __init__(self, **init_kwargs: Any) -> None:
            super().__init__(**init_kwargs)
            if self.model in GOOGLE_GENAI_MODELS_WITHOUT_SAMPLING_PARAMS:
                for param in GOOGLE_GENAI_UNSUPPORTED_SAMPLING_PARAMS:
                    self._generation_config.pop(param, None)

        def _sanitize_call_kwargs(self, call_kwargs: dict[str, Any]) -> dict[str, Any]:
            if self.model not in GOOGLE_GENAI_MODELS_WITHOUT_SAMPLING_PARAMS:
                return call_kwargs

            sanitized = dict(call_kwargs)
            for param in GOOGLE_GENAI_UNSUPPORTED_SAMPLING_PARAMS:
                sanitized.pop(param, None)

            generation_config = sanitized.get("generation_config")
            if generation_config is not None:
                if hasattr(generation_config, "model_dump"):
                    generation_config = generation_config.model_dump()
                elif isinstance(generation_config, dict):
                    generation_config = dict(generation_config)
                else:
                    return sanitized
                for param in GOOGLE_GENAI_UNSUPPORTED_SAMPLING_PARAMS:
                    generation_config.pop(param, None)
                sanitized["generation_config"] = generation_config
            return sanitized

        def _chat(self, messages: Any, **kwargs: Any) -> Any:
            return super()._chat(messages, **self._sanitize_call_kwargs(dict(kwargs)))

        async def _achat(self, messages: Any, **kwargs: Any) -> Any:
            return await super()._achat(
                messages, **self._sanitize_call_kwargs(dict(kwargs))
            )

        def _stream_chat(self, messages: Any, **kwargs: Any) -> Any:
            return super()._stream_chat(
                messages, **self._sanitize_call_kwargs(dict(kwargs))
            )

        async def _astream_chat(self, messages: Any, **kwargs: Any) -> Any:
            return await super()._astream_chat(
                messages, **self._sanitize_call_kwargs(dict(kwargs))
            )

        def structured_predict_without_function_calling(
            self,
            output_cls: Any,
            prompt: Any,
            llm_kwargs: dict[str, Any] | None = None,
            **prompt_args: Any,
        ) -> Any:
            return super().structured_predict_without_function_calling(
                output_cls,
                prompt,
                llm_kwargs=self._sanitize_call_kwargs(dict(llm_kwargs or {})),
                **prompt_args,
            )

        def structured_predict(
            self,
            output_cls: Any,
            prompt: Any,
            llm_kwargs: dict[str, Any] | None = None,
            **prompt_args: Any,
        ) -> Any:
            return super().structured_predict(
                output_cls,
                prompt,
                llm_kwargs=self._sanitize_call_kwargs(dict(llm_kwargs or {})),
                **prompt_args,
            )

        async def astructured_predict(
            self,
            output_cls: Any,
            prompt: Any,
            llm_kwargs: dict[str, Any] | None = None,
            **prompt_args: Any,
        ) -> Any:
            return await super().astructured_predict(
                output_cls,
                prompt,
                llm_kwargs=self._sanitize_call_kwargs(dict(llm_kwargs or {})),
                **prompt_args,
            )

        def stream_structured_predict(
            self,
            output_cls: Any,
            prompt: Any,
            llm_kwargs: dict[str, Any] | None = None,
            **prompt_args: Any,
        ) -> Any:
            return super().stream_structured_predict(
                output_cls,
                prompt,
                llm_kwargs=self._sanitize_call_kwargs(dict(llm_kwargs or {})),
                **prompt_args,
            )

        async def astream_structured_predict(
            self,
            output_cls: Any,
            prompt: Any,
            llm_kwargs: dict[str, Any] | None = None,
            **prompt_args: Any,
        ) -> Any:
            return await super().astream_structured_predict(
                output_cls,
                prompt,
                llm_kwargs=self._sanitize_call_kwargs(dict(llm_kwargs or {})),
                **prompt_args,
            )

    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    logger.debug(
        "Initializing MobilerunGoogleGenAI with kwargs: "
        f"{list(filtered_kwargs.keys())}"
    )
    return MobilerunGoogleGenAI(**filtered_kwargs)


def _load_anthropic(**kwargs: Any) -> LLM:
    from llama_index.core.types import PydanticProgramMode
    from llama_index.llms.anthropic import Anthropic

    class MobilerunAnthropic(Anthropic):
        @wraps(Anthropic._prepare_chat_with_tools)
        def _prepare_chat_with_tools(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            # FunctionCallingProgram passes None by default, overwriting the
            # valid tool-choice object built by the upstream adapter.
            if kwargs.get("tool_choice") is None:
                kwargs.pop("tool_choice", None)
            return super()._prepare_chat_with_tools(*args, **kwargs)

        @property
        def _model_kwargs(self) -> dict[str, Any]:
            model_kwargs = super()._model_kwargs
            if anthropic_model_omits_sampling_params(
                model_kwargs.get("model", self.model)
            ) and "temperature" not in (self.additional_kwargs or {}):
                model_kwargs.pop("temperature", None)
            return model_kwargs

        def _get_all_kwargs(self, **kwargs: Any) -> dict[str, Any]:
            model_kwargs = super()._get_all_kwargs(**kwargs)
            # LlamaIndex's structured program supplies an OpenAI-style string
            # after the Anthropic adapter has prepared its tool choice.
            tool_choice = model_kwargs.get("tool_choice")
            if isinstance(tool_choice, str) and tool_choice in {
                "auto",
                "required",
                "none",
            }:
                model_kwargs["tool_choice"] = {
                    "type": "any" if tool_choice == "required" else tool_choice
                }
            extra_body = dict(model_kwargs.get("extra_body") or {})
            effective_model = extra_body.get(
                "model", model_kwargs.get("model", self.model)
            )
            if anthropic_model_omits_sampling_params(effective_model):
                for param in ANTHROPIC_UNSUPPORTED_SAMPLING_PARAMS:
                    model_kwargs.pop(param, None)
                    extra_body.pop(param, None)
            else:
                # Anthropic SDK 1.x removed named sampling arguments even for
                # older models whose HTTP API still supports them. Preserve
                # the SDK's extra_body precedence and older SDK signatures.
                parameters = inspect.signature(self._client.messages.create).parameters
                accepts_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
                )
                for param in ANTHROPIC_UNSUPPORTED_SAMPLING_PARAMS:
                    if (
                        param in model_kwargs
                        and param not in parameters
                        and not accepts_kwargs
                    ):
                        extra_body.setdefault(param, model_kwargs.pop(param))
            if extra_body or "extra_body" in model_kwargs:
                model_kwargs["extra_body"] = extra_body
            return model_kwargs

        @property
        def metadata(self) -> LLMMetadata:
            context_window = anthropic_model_context_window(self.model)
            if context_window is None:
                return super().metadata
            return LLMMetadata(
                context_window=context_window,
                num_output=self.max_tokens,
                is_chat_model=True,
                model_name=self.model,
                is_function_calling_model=True,
            )

    # The upstream adapter defaults to 512 output tokens, which is too small
    # for a manager response that includes a complete control result plus
    # reasoning metadata.  Keep an explicit caller override intact while
    # giving standard Anthropic usage a safe shared default.  OAuth uses its
    # own loader above and deliberately retains its separate output limit.
    if kwargs.get("max_tokens") is None:
        kwargs["max_tokens"] = 2048

    # Fable 5.1 currently rejects the tool_choice payload produced by the
    # locked Anthropic adapter for function-based Pydantic programs. Its text
    # path returns the same validated Pydantic result without that field.
    if kwargs.get("model") == ANTHROPIC_FABLE_5_1_MODEL:
        kwargs["pydantic_program_mode"] = PydanticProgramMode.LLM

    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    logger.debug(
        f"Initializing MobilerunAnthropic with kwargs: {list(filtered_kwargs.keys())}"
    )
    return MobilerunAnthropic(**filtered_kwargs)


def load_llm(provider_name: str, model: str | None = None, **kwargs: Any) -> LLM:
    """Load and initialize a configured LLM backend.

    Args:
        provider_name: Case-sensitive provider name (e.g. "OpenAIResponses", "Ollama").
        model: Model name (e.g. "gpt-5.5", "gemini-3.7-flash").
        **kwargs: Keyword arguments for the LLM class constructor.

    Returns:
        An initialized LLM instance.
    """
    if not provider_name:
        raise ValueError("provider_name cannot be empty.")

    provider_name = normalize_provider_name(provider_name)

    if model is not None:
        if provider_name == "OpenAIResponses":
            model = normalize_model_id_for_variant("openai", "api_key", model)
        elif provider_name == "XAI":
            model = normalize_grok_model_id(model)
        elif provider_name == "xai_oauth":
            model = normalize_model_id_for_variant("xai", "oauth", model)
        elif provider_name == "openai_oauth":
            model = normalize_model_id_for_variant("openai", "oauth", model)
        kwargs["model"] = model

    # --- OAuth providers ---
    if provider_name == "openai_oauth":
        from mobilerun.agent.utils.oauth.openai_oauth_llm import OpenAIOAuth

        _validate_openai_oauth_model(kwargs.get("model"))
        return OpenAIOAuth(**{k: v for k, v in kwargs.items() if v is not None})
    if provider_name == "anthropic_oauth":
        from mobilerun.agent.utils.oauth.anthropic_oauth_llm import AnthropicOAuthLLM

        return AnthropicOAuthLLM(**{k: v for k, v in kwargs.items() if v is not None})
    if provider_name == "gemini_oauth_code_assist":
        from mobilerun.agent.utils.oauth.gemini_oauth_code_assist_llm import (
            GeminiOAuthCodeAssistLLM,
        )

        _validate_gemini_oauth_model(kwargs.get("model"))
        # Drop removed/legacy params a stale config YAML might still carry, so
        # construction doesn't fail and the legacy credential slot can't be
        # selected.
        kwargs.pop("consumer_mode", None)
        kwargs.pop("project_id", None)
        kwargs.pop("credential_slot", None)
        return GeminiOAuthCodeAssistLLM(
            **{k: v for k, v in kwargs.items() if v is not None}
        )
    if provider_name == "xai_oauth":
        from mobilerun.agent.utils.oauth.grok_oauth_llm import GrokOAuth

        return GrokOAuth(**{k: v for k, v in kwargs.items() if v is not None})

    # Legacy aliases: MiniMax and DeepSeek route through OpenAILike.
    if provider_name == "MiniMax":
        import os

        api_key = kwargs.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            api_key = os.environ.get("MINIMAX_API_KEY")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError(
                "MiniMax requires an API key. Pass api_key explicitly or set "
                "MINIMAX_API_KEY."
            )

        provider_name = "OpenAILike"
        kwargs["api_key"] = api_key
        kwargs.setdefault("is_chat_model", True)
        base_url = kwargs.pop("base_url", None)
        if not kwargs.get("api_base"):
            kwargs["api_base"] = base_url or MINIMAX_GLOBAL_BASE_URL
        warn_if_legacy_minimax_endpoint(kwargs.get("api_base"))

    if provider_name == "ZAI":
        provider_name = "OpenAILike"
        kwargs.setdefault("is_chat_model", True)
        if "base_url" in kwargs and "api_base" not in kwargs:
            kwargs["api_base"] = kwargs.pop("base_url")
        kwargs.setdefault("api_base", ZAI_GLOBAL_API_BASE)

    if provider_name == "XAI":
        import os

        # Mobilerun's reasoning mode selects its agent architecture. It does
        # not opt Grok into a provider-specific reasoning effort.
        kwargs.pop("reasoning_options", None)
        api_key = kwargs.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            api_key = os.environ.get("XAI_API_KEY")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError(
                "XAI requires an API key. Pass api_key explicitly or set "
                "XAI_API_KEY."
            )

        kwargs["api_key"] = api_key
        # The lowercase runtime alias should be useful without a separately
        # generated profile. Keep its implicit model aligned with the catalog.
        kwargs.setdefault("model", GROK_DEFAULT_MODEL)
        # XAI_API_KEY must only ever be sent to xAI's pinned endpoint. Ignore
        # generic CLI/profile URL overrides rather than allowing a malicious
        # config to redirect the bearer credential to another host.
        kwargs.pop("base_url", None)
        kwargs["api_base"] = XAI_API_BASE
        # Grok's catalog context is provider metadata, not a caller-tunable
        # endpoint option. Keep hand-written runtime profiles aligned with the
        # first-class provider catalog as well as generated profiles.
        kwargs["context_window"] = GROK_CONTEXT_WINDOW
        return _load_openai_responses(grok=True, **kwargs)

    if provider_name == "DeepSeek":
        import os

        provider_name = "OpenAILike"
        kwargs.setdefault("api_key", os.environ.get("DEEPSEEK_API_KEY"))
        kwargs.setdefault("is_chat_model", True)
        kwargs.setdefault(
            "is_function_calling_model", kwargs.get("model") == "deepseek-chat"
        )
        kwargs.setdefault("context_window", 64000)
        if "base_url" in kwargs and "api_base" not in kwargs:
            kwargs["api_base"] = kwargs.pop("base_url")
        kwargs.setdefault("api_base", "https://api.deepseek.com")

    if provider_name == "Requesty":
        import os

        api_key = kwargs.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            api_key = os.environ.get(REQUESTY_API_KEY_ENV_VAR)
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError(
                "Requesty requires an API key. Pass api_key explicitly or set "
                f"{REQUESTY_API_KEY_ENV_VAR}."
            )

        provider_name = "OpenAILike"
        kwargs["api_key"] = api_key
        kwargs.setdefault("model", REQUESTY_DEFAULT_MODEL)
        kwargs.setdefault("is_chat_model", True)
        kwargs.setdefault("is_function_calling_model", True)
        base_url = kwargs.pop("base_url", None)
        if not kwargs.get("api_base"):
            kwargs["api_base"] = resolve_requesty_base_url(base_url)

    # --- Standard providers (inline dispatch) ---
    if provider_name == "OpenAIResponses":
        kwargs.setdefault("model", OPENAI_API_DEFAULT_MODEL)
        return _load_openai_responses(**kwargs)
    elif provider_name == "OpenAILike":
        from llama_index.llms.openai_like import OpenAILike

        llm_class = OpenAILike
        kwargs.setdefault("is_chat_model", True)
        if "base_url" in kwargs and "api_base" not in kwargs:
            kwargs["api_base"] = kwargs.pop("base_url")
    elif provider_name == "GoogleGenAI":
        kwargs.setdefault("model", GEMINI_API_DEFAULT_MODEL)
        return _load_google_genai(**kwargs)
    elif provider_name == "Ollama":
        from llama_index.llms.ollama import Ollama

        llm_class = Ollama
        kwargs = _prepare_ollama_kwargs(kwargs, Ollama)
    elif provider_name == "Anthropic":
        kwargs.setdefault("model", ANTHROPIC_API_DEFAULT_MODEL)
        return _load_anthropic(**kwargs)
    elif provider_name == "OpenRouter":
        from llama_index.llms.openrouter import OpenRouter

        llm_class = OpenRouter
    else:
        raise ValueError(
            f"Unsupported provider '{provider_name}'. "
            f"Supported: {sorted(SUPPORTED_PROVIDERS)}"
        )

    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    logger.debug(
        f"Initializing {llm_class.__name__} with kwargs: {list(filtered_kwargs.keys())}"
    )
    return llm_class(**filtered_kwargs)


def load_llms_from_profiles(
    profiles: dict[str, "LLMProfile"],
    profile_names: list[str] | None = None,
    **override_kwargs_per_profile,
) -> dict[str, LLM]:
    """
    Load multiple LLMs from LLMProfile objects.

    Args:
        profiles: Dict of profile_name -> LLMProfile objects
        profile_names: List of profile names to load. If None, loads all profiles
        **override_kwargs_per_profile: Dict of profile-specific overrides
            Example: manager={'temperature': 0.1}, executor={'max_tokens': 8000}

    Returns:
        Dict mapping profile names to initialized LLM instances

    Example:
        >>> config = MobileConfig.from_yaml("config.yaml")
        >>> llms = load_llms_from_profiles(config.llm_profiles)
        >>> manager_llm = llms['manager']

        >>> # Load specific profiles with overrides
        >>> llms = load_llms_from_profiles(
        ...     config.llm_profiles,
        ...     profile_names=['manager', 'executor'],
        ...     manager={'temperature': 0.1}
        ... )
    """
    if profile_names is None:
        profile_names = list(profiles.keys())

    llms = {}
    for profile_name in profile_names:
        logger.debug(f"Loading LLM for profile: {profile_name}")

        if profile_name not in profiles:
            raise KeyError(
                f"Profile '{profile_name}' not found. "
                f"Available profiles: {list(profiles.keys())}"
            )

        profile = profiles[profile_name]

        # Get base kwargs from profile
        kwargs = profile.to_load_llm_kwargs()

        # Apply profile-specific overrides if provided
        if profile_name in override_kwargs_per_profile:
            logger.debug(
                f"Applying overrides for {profile_name}: {override_kwargs_per_profile[profile_name]}"
            )
            kwargs.update(override_kwargs_per_profile[profile_name])

        # Load the LLM
        llms[profile_name] = load_llm(provider_name=profile.provider, **kwargs)
        logger.debug(
            f"Successfully loaded {profile_name} LLM: {profile.provider}/{profile.model}"
        )

    return llms


# --- Example Usage ---
if __name__ == "__main__":
    # Install the specific LLM integrations you want to test:
    # pip install \
    #   llama-index-llms-anthropic \
    #   llama-index-llms-gemini \
    #   llama-index-llms-openai

    from llama_index.core.base.llms.types import ChatMessage

    providers = [
        {
            "name": "Anthropic",
            "model": "claude-3-7-sonnet-latest",
        },
        {
            "name": "DeepSeek",
            "model": "deepseek-reasoner",
        },
        {
            "name": "GoogleGenAI",
            "model": "gemini-3.7-flash",
        },
        {
            "name": "OpenAIResponses",
            "model": "gpt-4",
        },
        {
            "name": "Ollama",
            "model": "llama3.2:1b",
            "base_url": "http://localhost:11434",
        },
    ]

    system_prompt = ChatMessage(
        role="system",
        content="You are a personal health and food coach. You are given a user's health and food preferences and you need to recommend a meal plan for them. only output the meal plan, no other text.",
    )

    user_prompt = ChatMessage(
        role="user",
        content="I am a 25 year old male. I am 5'10 and 180 pounds. I am a vegetarian. I am allergic to peanuts and tree nuts. I am allergic to shellfish. I am allergic to eggs. I am allergic to dairy. I am allergic to soy. I am allergic to wheat. I am allergic to corn. I am allergic to oats. I am allergic to rice. I am allergic to barley. I am allergic to rye. I am allergic to oats. I am allergic to rice. I am allergic to barley. I am allergic to rye.",
    )

    messages = [system_prompt, user_prompt]

    for provider in providers:
        print(f"\n{'#' * 35} Loading {provider['name']} {'#' * 35}")
        print("-" * 100)

        try:
            provider_name = provider.pop("name")
            llm = load_llm(provider_name, **provider)
            provider["name"] = provider_name
            print(f"Loaded LLM: {type(llm)}")
            print(f"Model: {llm.metadata}")
            print("-" * 100)

            tracker = track_usage(llm)
            print(f"Tracker: {type(tracker)}")
            print(f"Usage: {tracker.usage}")
            print("-" * 100)

            assert tracker.usage.requests == 0
            assert tracker.usage.request_tokens == 0
            assert tracker.usage.response_tokens == 0
            assert tracker.usage.total_tokens == 0

            res = llm.chat(messages)
            print(f"Response: {res.message.content}")
            print("-" * 100)
            print(f"Usage: {tracker.usage}")

            assert tracker.usage.requests == 1
            assert tracker.usage.request_tokens > 0
            assert tracker.usage.response_tokens > 0
            assert tracker.usage.total_tokens > tracker.usage.request_tokens
            assert tracker.usage.total_tokens > tracker.usage.response_tokens
        except Exception as e:
            print(f"Failed to load and track usage for {provider['name']}: {e}")
