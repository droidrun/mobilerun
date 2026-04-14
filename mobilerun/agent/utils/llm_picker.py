import logging
from typing import TYPE_CHECKING, Any

from llama_index.core.llms.llm import LLM

from droidrun.agent.usage import track_usage

if TYPE_CHECKING:
    from droidrun.config_manager.config_manager import LLMProfile

# Configure logging
logger = logging.getLogger("droidrun")


SUPPORTED_PROVIDERS = [
    "OpenAIResponses", "OpenAILike", "GoogleGenAI",
    "Ollama", "Anthropic", "DeepSeek", "OpenRouter", "MiniMax",
]


def load_llm(provider_name: str, model: str | None = None, **kwargs: Any) -> LLM:
    """Load and initialize a configured LLM backend.

    Args:
        provider_name: Case-sensitive provider name (e.g. "OpenAIResponses", "Ollama").
        model: Model name (e.g. "gpt-4", "gemini-3.1-flash-lite-preview").
        **kwargs: Keyword arguments for the LLM class constructor.

    Returns:
        An initialized LLM instance.
    """
    if not provider_name:
        raise ValueError("provider_name cannot be empty.")

    if model is not None:
        kwargs["model"] = model

    # --- OAuth providers ---
    if provider_name == "openai_oauth":
        from droidrun.agent.utils.oauth.openai_oauth_llm import OpenAIOAuth
        return OpenAIOAuth(**{k: v for k, v in kwargs.items() if v is not None})
    if provider_name == "anthropic_oauth":
        from droidrun.agent.utils.oauth.anthropic_oauth_llm import AnthropicOAuthLLM
        return AnthropicOAuthLLM(**{k: v for k, v in kwargs.items() if v is not None})
    if provider_name == "gemini_oauth_code_assist":
        from droidrun.agent.utils.oauth.gemini_oauth_code_assist_llm import GeminiOAuthCodeAssistLLM
        return GeminiOAuthCodeAssistLLM(**{k: v for k, v in kwargs.items() if v is not None})

    # Legacy alias: MiniMax configs now route through OpenAILike.
    if provider_name == "MiniMax":
        provider_name = "OpenAILike"
        kwargs.setdefault("is_chat_model", True)
        kwargs.setdefault("api_base", "https://api.minimaxi.chat/v1")
        if "base_url" in kwargs and "api_base" not in kwargs:
            kwargs["api_base"] = kwargs.pop("base_url")

    # --- Standard providers (inline dispatch) ---
    if provider_name == "OpenAIResponses":
        from llama_index.llms.openai.responses import OpenAIResponses
        llm_class = OpenAIResponses
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
    elif provider_name == "Anthropic":
        from llama_index.llms.anthropic import Anthropic
        llm_class = Anthropic
    elif provider_name == "DeepSeek":
        from llama_index.llms.deepseek import DeepSeek
        llm_class = DeepSeek
    elif provider_name == "OpenRouter":
        from llama_index.llms.openrouter import OpenRouter
        llm_class = OpenRouter
    else:
        raise ValueError(
            f"Unsupported provider '{provider_name}'. "
            f"Supported: {sorted(SUPPORTED_PROVIDERS)}"
        )

    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    logger.debug(f"Initializing {llm_class.__name__} with kwargs: {list(filtered_kwargs.keys())}")
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
        >>> config = DroidConfig.from_yaml("config.yaml")
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
    #   llama-index-llms-deepseek \
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
            "model": "gemini-3.1-flash-lite-preview",
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
