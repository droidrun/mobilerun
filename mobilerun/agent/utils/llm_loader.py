"""
LLM Loader - Centralized logic for loading agent-specific LLMs based on configuration.

This module determines which LLMs are needed based on the agent mode (reasoning vs direct execution)
and loads them from config profiles.
"""

import logging
from typing import Any, List, Type

from llama_index.core.llms.llm import LLM
from pydantic import BaseModel

from droidrun.agent.utils.llm_picker import load_llm, load_llms_from_profiles
from droidrun.config_manager.config_manager import DroidConfig

logger = logging.getLogger("droidrun")


def _get_required_profiles(
    config: DroidConfig, output_model: Type[BaseModel] | None = None
) -> List[str]:
    """
    Determine which LLM profiles are required based on agent configuration.

    Args:
        config: Droidrun configuration containing agent settings
        output_model: Optional Pydantic model for structured output extraction

    Returns:
        List of required profile names
    """
    if config.agent.reasoning:
        profiles = ["manager", "executor", "app_opener"]
    else:
        # Direct execution mode only needs FastAgent and helper agents
        profiles = ["fast_agent", "app_opener"]

    # Add structured_output if output_model is provided
    if output_model is not None:
        profiles.append("structured_output")

    return profiles


def validate_llm_dict(
    config: DroidConfig,
    llms: dict[str, LLM],
    output_model: Type[BaseModel] | None = None,
) -> List[str]:
    """
    Validate that required LLM profiles exist in the provided LLM dictionary.

    Args:
        config: Droidrun configuration containing LLM profiles
        llms: Dictionary containing LLM profiles
        output_model: Optional Pydantic model for structured output extraction

    Returns:
        List of required profile names

    Raises:
        ValueError: If any required profiles are missing from llms
    """
    required_profiles = _get_required_profiles(config, output_model)
    missing_profiles = [name for name in required_profiles if name not in llms]

    if missing_profiles:
        available_profiles = list(llms.keys())
        raise ValueError(
            f"Missing required LLM profiles: {', '.join(missing_profiles)}.\n"
            f"Available profiles: {', '.join(available_profiles)}.\n"
            f"Please add the missing profiles to your config."
        )

    return required_profiles


def validate_llm_profiles(
    config: DroidConfig, output_model: Type[BaseModel] | None = None
) -> List[str]:
    """
    Validate that required LLM profiles exist in the configuration.

    Args:
        config: Droidrun configuration containing LLM profiles
        output_model: Optional Pydantic model for structured output extraction

    Returns:
        List of required profile names

    Raises:
        ValueError: If any required profiles are missing from config
    """
    required_profiles = _get_required_profiles(config, output_model)
    missing_profiles = [
        name for name in required_profiles if name not in config.llm_profiles
    ]

    if missing_profiles:
        available_profiles = list(config.llm_profiles.keys())
        raise ValueError(
            f"Missing required LLM profiles in config: {', '.join(missing_profiles)}.\n"
            f"Available profiles: {', '.join(available_profiles)}.\n"
            f"Please add the missing profiles to your config."
        )

    return required_profiles


def load_agent_llms(
    config: DroidConfig,
    custom_provider: str | None = None,
    custom_model: str | None = None,
    temperature: float | None = None,
    output_model: Type[BaseModel] | None = None,
    **kwargs: Any,
) -> dict[str, LLM]:
    """
    Load LLMs required for DroidAgent based on reasoning mode and configuration.

    Args:
        config: Droidrun configuration containing LLM profiles
        custom_provider: Optional custom provider to use for all agents
        custom_model: Optional custom model to use for all agents
        temperature: Optional temperature override for all agents
        output_model: Optional Pydantic model for structured output extraction
        **kwargs: Additional kwargs to pass to LLM constructor (base_url, api_base, etc.)

    Returns:
        Dictionary mapping agent type to LLM instance:
        - If reasoning=True: {manager, executor, app_opener, structured_output?}
        - If reasoning=False: {fast_agent, app_opener, structured_output?}
        - structured_output is only included if output_model is provided

    Raises:
        ValueError: If required LLM profiles are missing from config
    """
    # Check if user wants custom LLM for all agents
    if custom_provider is not None or custom_model is not None:
        logger.debug("🔧 Using custom LLM for all agents")

        # Use provided values or fall back to first profile's defaults
        if custom_provider is None:
            custom_provider = list(config.llm_profiles.values())[0].provider
        if custom_model is None:
            custom_model = list(config.llm_profiles.values())[0].model

        # Build kwargs
        llm_kwargs = {}
        if temperature is not None:
            llm_kwargs["temperature"] = temperature
        else:
            llm_kwargs["temperature"] = kwargs.get("temperature", 0.3)

        # Add any additional kwargs (base_url, api_base, etc.)
        llm_kwargs.update(kwargs)

        # Load single LLM for all agents
        custom_llm = load_llm(
            provider_name=custom_provider, model=custom_model, **llm_kwargs
        )

        # Use same LLM for all agents
        llms = {
            "manager": custom_llm,
            "executor": custom_llm,
            "fast_agent": custom_llm,
            "app_opener": custom_llm,
        }

        # Add structured_output if output_model is provided
        if output_model is not None:
            llms["structured_output"] = custom_llm

        logger.debug(f"🧠 Custom LLM ready: {custom_provider}/{custom_model}")
        return llms

    # No custom provider/model - use profiles from config
    logger.debug("📋 Loading LLMs from config profiles...")

    # Determine which LLMs are needed and validate they exist
    profile_names = validate_llm_profiles(config, output_model)

    # Apply temperature override to all profiles if specified
    overrides = {}
    if temperature is not None:
        overrides = {name: {"temperature": temperature} for name in profile_names}

    # Add any additional kwargs to overrides
    if kwargs:
        for name in profile_names:
            if name not in overrides:
                overrides[name] = {}
            overrides[name].update(kwargs)

    # Load LLMs from profiles
    llms = load_llms_from_profiles(
        config.llm_profiles, profile_names=profile_names, **overrides
    )
    logger.debug(f"🧠 Loaded {len(llms)} agent-specific LLMs from profiles")

    return llms


def merge_llms_with_config(
    config: DroidConfig,
    llms: dict[str, LLM],
    output_model: Type[BaseModel] | None = None,
    **kwargs: Any,
) -> dict[str, LLM]:
    """
    Merge a user-provided partial llms dict with LLMs loaded from the
    configuration for any missing required profiles.
    Args:
        config: Droidrun configuration containing LLM profiles
        llms: Partial dictionary of LLM instances provided by user. Keys may be
            a subset of required agent names.
        output_model: Optional Pydantic model for structured output extraction
        **kwargs: Additional kwargs forwarded when loading missing profiles
    Returns:
        A dictionary containing the union of provided LLMs and any LLMs loaded
        from config for missing required profiles.
    Raises:
        ValueError: If required profiles are missing from both the provided
            llms dict and the config profiles.
    """
    required = _get_required_profiles(config, output_model)

    missing = [name for name in required if name not in llms]
    if not missing:
        return llms

    logger.debug(
        f"🔁 Merging provided LLMs with config for missing profiles: {', '.join(missing)}"
    )

    # Build overrides for missing profiles (temperature and other kwargs)
    overrides: dict[str, dict] = {}
    temperature = kwargs.get("temperature", None)
    if temperature is not None:
        overrides = {name: {"temperature": temperature} for name in missing}

    if kwargs:
        for name in missing:
            if name not in overrides:
                overrides[name] = {}
            overrides[name].update(kwargs)

    loaded = load_llms_from_profiles(
        config.llm_profiles, profile_names=missing, **overrides
    )

    merged = {**llms, **loaded}
    logger.debug(f"✅ Merged LLMs. Loaded: {', '.join(list(loaded.keys()))}")
    return merged
