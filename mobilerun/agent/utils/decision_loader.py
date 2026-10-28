import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from mobilerun.agent.decision_models.typesafe import TypeSafeDecisionModel
from mobilerun.agent.selectors.decision import DecisionModel

if TYPE_CHECKING:
    from mobilerun.config_manager.config_manager import DecisionProfile, MobileConfig

logger = logging.getLogger("mobilerun")

_providers: dict[str, Callable[..., DecisionModel]] = {
    "typesafe": TypeSafeDecisionModel,
}


def register_decision_provider(name: str, factory: Callable[..., DecisionModel]) -> None:
    name = name.strip().lower()
    if not name:
        raise ValueError("Decision provider name cannot be empty")
    if not callable(factory):
        raise TypeError("Decision provider factory must be callable")
    if name in _providers:
        raise ValueError(f"Decision provider '{name}' is already registered")
    _providers[name] = factory


def load_decision_model(
    provider_name: str, model: str | None = None, **kwargs: Any
) -> DecisionModel:
    name = provider_name.strip().lower()
    if name not in _providers:
        raise ValueError(
            f"Unsupported decision provider '{provider_name}'. "
            f"Registered providers: {sorted(_providers)}"
        )
    if model is not None:
        kwargs["model"] = model
    logger.debug(f"🔄 Loading decision model from provider: {name}")
    instance = _providers[name](**kwargs)
    if not callable(getattr(instance, "decide", None)):
        raise TypeError("Decision provider must return a model implementing decide(request)")
    logger.debug(f"🧠 Decision model ready: {name}")
    return instance


def load_decision_models_from_profiles(
    profiles: dict[str, "DecisionProfile"],
    profile_names: list[str] | None = None,
    **override_kwargs_per_profile: Any,
) -> dict[str, DecisionModel]:
    """Load named decision profiles, just like load_llms_from_profiles.

    Per-profile overrides take precedence over the profile's constructor options.
    Provider defaults are preserved; LLM-only options are never injected.
    """
    names = list(profiles) if profile_names is None else profile_names
    missing = [name for name in names if name not in profiles]
    if missing:
        raise KeyError(
            f"Decision profiles {missing} not found. Available profiles: {list(profiles)}"
        )
    models = {}
    for name in names:
        profile = profiles[name]
        kwargs = profile.to_load_decision_kwargs()
        kwargs.update(override_kwargs_per_profile.get(name, {}))
        models[name] = load_decision_model(profile.provider, **kwargs)
    return models


def load_agent_decision_models(config: "MobileConfig") -> dict[str, DecisionModel]:
    """Load only the active action role; absent profiles keep the LLM path."""
    role = "executor" if config.agent.reasoning else "fast_agent"
    if role not in config.decision_profiles:
        return {}
    return load_decision_models_from_profiles(config.decision_profiles, [role])
