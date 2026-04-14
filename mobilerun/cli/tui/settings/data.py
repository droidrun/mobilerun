"""Settings data model for the TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from droidrun.config_manager.config_manager import DroidConfig

from droidrun.agent.providers.registry import VARIANT_ENV_KEY_SLOT
from droidrun.config_manager.env_keys import load_env_key_sources, save_env_keys

PROVIDERS = [
    "GoogleGenAI",
    "OpenAIResponses",
    "Anthropic",
    "Ollama",
    "OpenAILike",
]

AGENT_ROLES = ["manager", "executor", "fast_agent"]

# Maps TUI provider name to the provider_family used by the backend.
PROVIDER_FAMILY: dict[str, str] = {
    "GoogleGenAI": "gemini",
    "OpenAIResponses": "openai",
    "Anthropic": "anthropic",
    "Ollama": "ollama",
    "OpenAILike": "openai_like",
}

# Which fields are relevant per provider.
PROVIDER_FIELDS: dict[str, dict[str, Any]] = {
    "GoogleGenAI": {"api_key": True, "api_source": True, "base_url": False},
    "OpenAIResponses": {"api_key": True, "api_source": True, "base_url": False},
    "Anthropic": {"api_key": True, "api_source": True, "base_url": False},
    "Ollama": {"api_key": False, "api_source": False, "base_url": True},
    "OpenAILike": {"api_key": True, "api_source": False, "base_url": True},
}


@dataclass
class ProfileSettings:
    """Full LLM profile for one agent role."""

    provider: str = "GoogleGenAI"
    model: str = "gemini-3.1-flash-lite-preview"
    temperature: float = 0.2
    api_key: str = ""
    api_key_source: str = "auto"
    base_url: str = ""
    kwargs: dict[str, str] = field(default_factory=dict)


@dataclass
class SettingsData:
    """All TUI settings in one object."""

    # Per-agent LLM profiles (the real config, no fake global)
    profiles: dict[str, ProfileSettings] = field(
        default_factory=lambda: {role: ProfileSettings() for role in AGENT_ROLES}
    )

    # Per-agent custom prompt paths
    agent_prompts: dict[str, str] = field(
        default_factory=lambda: {role: "" for role in AGENT_ROLES}
    )

    # Agent
    manager_vision: bool = True
    executor_vision: bool = False
    fast_agent_vision: bool = False
    reasoning: bool = False
    max_steps: int = 15

    # Advanced
    use_tcp: bool = False
    debug: bool = False
    save_trajectory: bool = False
    trajectory_gifs: bool = True
    tracing_enabled: bool = False
    tracing_provider: str = "phoenix"

    # Langfuse
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_screenshots: bool = False

    # Timing
    after_sleep_action: float = 1.0
    wait_for_stable_ui: float = 0.3

    @classmethod
    def from_config(cls, config: DroidConfig) -> SettingsData:
        """Build settings from a loaded DroidConfig."""
        llm_profiles = config.llm_profiles or {}
        env_key_sources = load_env_key_sources()

        profiles: dict[str, ProfileSettings] = {}
        for role in AGENT_ROLES:
            lp = llm_profiles.get(role)
            if lp:
                # Determine API key source
                provider = lp.provider
                env_slot = VARIANT_ENV_KEY_SLOT.get(provider)
                if env_slot:
                    sources = env_key_sources.get(env_slot)
                    selected_source = getattr(lp, "api_key_source", "auto") or "auto"
                    if selected_source == "env":
                        api_key = sources.shell if sources else ""
                    elif selected_source == "file":
                        api_key = sources.saved if sources else ""
                    else:
                        api_key = ""
                        if sources:
                            api_key = sources.shell or sources.saved
                elif provider == "OpenAILike":
                    api_key = lp.kwargs.get("api_key", "stub")
                    selected_source = "auto"
                else:
                    api_key = lp.kwargs.get("api_key", "")
                    selected_source = "auto"

                # Build kwargs without api_key (shown separately)
                kwargs = {k: str(v) for k, v in lp.kwargs.items() if k != "api_key"}

                profiles[role] = ProfileSettings(
                    provider=provider,
                    model=lp.model,
                    temperature=lp.temperature,
                    api_key=api_key,
                    api_key_source=selected_source,
                    base_url=lp.base_url or lp.api_base or "",
                    kwargs=kwargs,
                )
            else:
                profiles[role] = ProfileSettings()

        agent_prompts = {
            "manager": config.agent.manager.system_prompt,
            "executor": config.agent.executor.system_prompt,
            "fast_agent": config.agent.fast_agent.system_prompt,
        }

        return cls(
            profiles=profiles,
            agent_prompts=agent_prompts,
            manager_vision=config.agent.manager.vision,
            executor_vision=config.agent.executor.vision,
            fast_agent_vision=config.agent.fast_agent.vision,
            reasoning=config.agent.reasoning,
            max_steps=config.agent.max_steps,
            use_tcp=config.device.use_tcp,
            debug=config.logging.debug,
            save_trajectory=config.logging.save_trajectory != "none",
            trajectory_gifs=config.logging.trajectory_gifs,
            tracing_enabled=config.tracing.enabled,
            tracing_provider=config.tracing.provider,
            langfuse_host=config.tracing.langfuse_host,
            langfuse_public_key=config.tracing.langfuse_public_key,
            langfuse_secret_key=config.tracing.langfuse_secret_key,
            langfuse_screenshots=config.tracing.langfuse_screenshots,
            after_sleep_action=config.agent.after_sleep_action,
            wait_for_stable_ui=config.agent.wait_for_stable_ui,
        )

    def save(self) -> None:
        """Persist all settings: API keys to .env and config to config.yaml."""
        from droidrun.config_manager.loader import ConfigLoader

        # Save env-based API keys for all cloud providers that have a key set
        env_keys: dict[str, str] = {}
        for role, profile in self.profiles.items():
            env_slot = VARIANT_ENV_KEY_SLOT.get(profile.provider)
            if env_slot and profile.api_key and profile.api_key_source != "env":
                env_keys[env_slot] = profile.api_key
        if env_keys:
            save_env_keys(env_keys)

        try:
            config = ConfigLoader.load()
        except Exception:
            from droidrun.config_manager.config_manager import DroidConfig

            config = DroidConfig()

        self.apply_to_config(config)
        ConfigLoader.save(config)

    @staticmethod
    def _build_kwargs(ps: ProfileSettings) -> dict[str, Any]:
        """Parse kwargs string values to typed values and inject api_key for OpenAILike."""
        parsed: dict[str, Any] = {}
        for k, v in ps.kwargs.items():
            if not k:
                continue
            try:
                parsed[k] = int(v)
            except ValueError:
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v
        if ps.provider == "OpenAILike":
            parsed["api_key"] = ps.api_key or "stub"
        return parsed

    @staticmethod
    def _apply_profile_to_llm(
        ps: ProfileSettings, cp: Any, update_model: bool = True
    ) -> None:
        """Write a ProfileSettings onto an LLMProfile config object."""
        cp.provider = ps.provider
        cp.provider_family = PROVIDER_FAMILY.get(ps.provider, "")
        cp.auth_mode = "api_key"
        cp.credential_path = None
        if update_model:
            cp.model = ps.model
        cp.temperature = ps.temperature
        cp.api_key_source = ps.api_key_source
        if ps.base_url:
            cp.base_url = ps.base_url
            if ps.provider == "OpenAILike":
                cp.api_base = ps.base_url
        else:
            cp.base_url = None
            cp.api_base = None
        cp.kwargs = SettingsData._build_kwargs(ps)

    def apply_to_config(self, config: DroidConfig) -> None:
        """Apply all TUI settings onto a DroidConfig, in place."""
        for role, ps in self.profiles.items():
            if role not in config.llm_profiles:
                continue
            self._apply_profile_to_llm(ps, config.llm_profiles[role])

        # Propagate fast_agent settings to hidden roles (app_opener, structured_output)
        # keeping their existing model (these are usually lighter models)
        fast_agent_ps = self.profiles.get("fast_agent")
        if fast_agent_ps:
            for hidden_role in ("app_opener", "structured_output"):
                if hidden_role in config.llm_profiles:
                    self._apply_profile_to_llm(
                        fast_agent_ps,
                        config.llm_profiles[hidden_role],
                        update_model=False,
                    )

        # Per-agent prompt paths
        prompt = self.agent_prompts.get("manager", "")
        if prompt:
            config.agent.manager.system_prompt = prompt
        prompt = self.agent_prompts.get("executor", "")
        if prompt:
            config.agent.executor.system_prompt = prompt
        prompt = self.agent_prompts.get("fast_agent", "")
        if prompt:
            config.agent.fast_agent.system_prompt = prompt
        # Agent
        config.agent.max_steps = self.max_steps
        config.agent.reasoning = self.reasoning
        config.agent.manager.vision = self.manager_vision
        config.agent.executor.vision = self.executor_vision
        config.agent.fast_agent.vision = self.fast_agent_vision

        # Device
        config.device.use_tcp = self.use_tcp

        # Logging
        config.logging.debug = self.debug
        config.logging.save_trajectory = "action" if self.save_trajectory else "none"
        config.logging.trajectory_gifs = self.trajectory_gifs

        # Tracing
        config.tracing.enabled = self.tracing_enabled
        config.tracing.provider = self.tracing_provider
        config.tracing.langfuse_host = self.langfuse_host
        config.tracing.langfuse_public_key = self.langfuse_public_key
        config.tracing.langfuse_secret_key = self.langfuse_secret_key
        config.tracing.langfuse_screenshots = self.langfuse_screenshots

        # Timing
        config.agent.after_sleep_action = self.after_sleep_action
        config.agent.wait_for_stable_ui = self.wait_for_stable_ui
