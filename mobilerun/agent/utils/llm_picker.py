import logging
from typing import TYPE_CHECKING, Any

from llama_index.core.llms.llm import LLM

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
    "MiniMax",
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
}

ZAI_GLOBAL_API_BASE = "https://api.z.ai/api/paas/v4"
OPENAI_RESPONSES_MODELS_WITHOUT_SAMPLING_PARAMS = {"gpt-5.5"}
OPENAI_RESPONSES_UNSUPPORTED_SAMPLING_PARAMS = {"temperature", "top_p"}
ANTHROPIC_CURRENT_MODEL_CONTEXT_WINDOWS = {
    "claude-opus-4-8": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
}


def normalize_provider_name(provider_name: str) -> str:
    """Map user-facing provider names to Mobilerun runtime providers."""
    stripped = provider_name.strip()
    key = stripped.lower()
    return PROVIDER_ALIASES.get(key, stripped)


def _openai_responses_model_omits_sampling_params(model: object) -> bool:
    return str(model or "").strip() in OPENAI_RESPONSES_MODELS_WITHOUT_SAMPLING_PARAMS


def _anthropic_model_omits_temperature(model: object) -> bool:
    return str(model or "").strip().startswith("claude-opus-4")


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


def _load_openai_responses(**kwargs: Any) -> LLM:
    from llama_index.llms.openai.responses import OpenAIResponses

    class MobilerunOpenAIResponses(OpenAIResponses):
        def _get_model_kwargs(self, **kwargs: Any) -> dict[str, Any]:
            model_kwargs = super()._get_model_kwargs(**kwargs)
            if _openai_responses_model_omits_sampling_params(
                model_kwargs.get("model", self.model)
            ):
                for param in OPENAI_RESPONSES_UNSUPPORTED_SAMPLING_PARAMS:
                    model_kwargs.pop(param, None)
            return model_kwargs

    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    logger.debug(
        "Initializing MobilerunOpenAIResponses with kwargs: "
        f"{list(filtered_kwargs.keys())}"
    )
    return MobilerunOpenAIResponses(**filtered_kwargs)


def _load_anthropic(**kwargs: Any) -> LLM:
    from llama_index.core.base.llms.types import LLMMetadata
    from llama_index.llms.anthropic import Anthropic

    class MobilerunAnthropic(Anthropic):
        @property
        def _model_kwargs(self) -> dict[str, Any]:
            model_kwargs = super()._model_kwargs
            if _anthropic_model_omits_temperature(
                model_kwargs.get("model", self.model)
            ) and "temperature" not in (self.additional_kwargs or {}):
                model_kwargs.pop("temperature", None)
            return model_kwargs

        @property
        def metadata(self) -> LLMMetadata:
            try:
                return super().metadata
            except ValueError:
                context_window = ANTHROPIC_CURRENT_MODEL_CONTEXT_WINDOWS.get(self.model)
                if context_window is None:
                    raise
                return LLMMetadata(
                    context_window=context_window,
                    num_output=self.max_tokens,
                    is_chat_model=True,
                    model_name=self.model,
                    is_function_calling_model=True,
                )

    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    logger.debug(
        f"Initializing MobilerunAnthropic with kwargs: {list(filtered_kwargs.keys())}"
    )
    return MobilerunAnthropic(**filtered_kwargs)


def load_llm(provider_name: str, model: str | None = None, **kwargs: Any) -> LLM:
    """Load and initialize a configured LLM backend.

    Args:
        provider_name: Case-sensitive provider name (e.g. "OpenAIResponses", "Ollama").
        model: Model name (e.g. "gpt-4", "gemini-3.1-flash-lite").
        **kwargs: Keyword arguments for the LLM class constructor.

    Returns:
        An initialized LLM instance.
    """
    if not provider_name:
        raise ValueError("provider_name cannot be empty.")

    provider_name = normalize_provider_name(provider_name)

    if model is not None:
        kwargs["model"] = model

    # Legacy aliases: MiniMax and DeepSeek route through OpenAILike.
    if provider_name == "MiniMax":
        provider_name = "OpenAILike"
        kwargs.setdefault("is_chat_model", True)
        kwargs.setdefault("api_base", "https://api.minimaxi.chat/v1")
        if "base_url" in kwargs and "api_base" not in kwargs:
            kwargs["api_base"] = kwargs.pop("base_url")

    if provider_name == "ZAI":
        provider_name = "OpenAILike"
        kwargs.setdefault("is_chat_model", True)
        if "base_url" in kwargs and "api_base" not in kwargs:
            kwargs["api_base"] = kwargs.pop("base_url")
        kwargs.setdefault("api_base", ZAI_GLOBAL_API_BASE)

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

    # --- Standard providers (inline dispatch) ---
    if provider_name == "OpenAIResponses":
        return _load_openai_responses(**kwargs)
    elif provider_name == "OpenAILike":
        from llama_index.llms.openai_like import OpenAILike

        llm_class = OpenAILike
        kwargs.setdefault("is_chat_model", True)
        if "base_url" in kwargs and "api_base" not in kwargs:
            kwargs["api_base"] = kwargs.pop("base_url")
    elif provider_name == "GoogleGenAI":
        from llama_index.llms.google_genai import GoogleGenAI

        llm_class = GoogleGenAI
    elif provider_name == "Ollama":
        from llama_index.llms.ollama import Ollama

        llm_class = Ollama
        kwargs = _prepare_ollama_kwargs(kwargs, Ollama)
    elif provider_name == "Anthropic":
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
            "model": "gemini-3.1-flash-lite",
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
