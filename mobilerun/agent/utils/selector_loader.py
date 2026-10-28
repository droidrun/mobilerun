"""Build configured selector compositions using the existing model loaders."""

from typing import Any

from mobilerun.agent.selectors import (
    ActionSelector,
    DecisionSelector,
    DelegatingSelector,
    LlamaIndexLLMSelector,
)
from mobilerun.agent.utils.decision_loader import load_decision_models_from_profiles
from mobilerun.agent.utils.llm_picker import load_llms_from_profiles
from mobilerun.config_manager.config_manager import MobileConfig


def _load_selector(config: MobileConfig, profile: str, **llm_kwargs: Any) -> ActionSelector:
    # Match normal action-role resolution: a decision profile takes precedence.
    if profile in config.decision_profiles:
        models = load_decision_models_from_profiles(config.decision_profiles, [profile])
        return DecisionSelector(models[profile])
    if profile in config.llm_profiles:
        llms = load_llms_from_profiles(
            config.llm_profiles, profile_names=[profile], **{profile: llm_kwargs}
        )
        return LlamaIndexLLMSelector(llms[profile])
    raise KeyError(
        f"Selector profile '{profile}' not found. "
        f"Available decision profiles: {list(config.decision_profiles)}. "
        f"Available LLM profiles: {list(config.llm_profiles)}"
    )


def load_delegating_selector(config: MobileConfig, **llm_kwargs: Any) -> DelegatingSelector:
    """Resolve both sides from named profiles without changing helper LLM roles."""
    delegation = config.agent.delegation
    if delegation is None:
        raise ValueError("agent.delegation must be configured to load a DelegatingSelector")
    primary = delegation.primary or ("executor" if config.agent.reasoning else "fast_agent")
    return DelegatingSelector(
        primary=_load_selector(config, primary, **llm_kwargs),
        delegate=_load_selector(config, delegation.delegate, **llm_kwargs),
        description=delegation.description,
    )
