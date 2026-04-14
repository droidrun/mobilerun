from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

import yaml

from droidrun.agent.providers.registry import VARIANT_ENV_KEY_SLOT
from droidrun.config_manager.env_keys import API_KEY_ENV_VARS, load_env_key_sources
from droidrun.config_manager.path_resolver import PathResolver
from droidrun.mcp.config import MCPConfig, MCPServerConfig


# ---------- Config Schema ----------
@dataclass
class LLMProfile:
    """LLM profile configuration."""

    provider: str = "GoogleGenAI"
    model: str = "gemini-3.1-flash-lite-preview"
    temperature: float = 0.2
    api_key_source: Literal["auto", "env", "file"] = "auto"
    base_url: Optional[str] = None
    api_base: Optional[str] = None
    provider_family: Optional[str] = None
    auth_mode: Optional[str] = None
    credential_path: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_load_llm_kwargs(self) -> Dict[str, Any]:
        """Convert profile to kwargs for load_llm function."""
        result = {
            "model": self.model,
            "temperature": self.temperature,
        }
        # Add optional URL parameters
        if self.base_url:
            result["base_url"] = self.base_url
        if self.api_base:
            result["api_base"] = self.api_base
        if self.credential_path:
            result["credential_path"] = self.credential_path
        # Merge additional kwargs
        result.update(self.kwargs)
        # OAuth providers handle auth via credential files, not API keys.
        if self.auth_mode == "oauth":
            return result
        # Look up by provider name first (works for GoogleGenAI, Anthropic, etc.).
        # Fall back to provider_family for transport-wrapped providers like
        # MiniMax/ZAI that route through OpenAILike.
        env_slot = VARIANT_ENV_KEY_SLOT.get(self.provider)
        if env_slot is None and self.provider_family in API_KEY_ENV_VARS:
            env_slot = self.provider_family
        if env_slot and "api_key" not in result:
            sources = load_env_key_sources().get(env_slot)
            if sources is not None:
                if self.api_key_source == "env":
                    api_key = sources.shell
                elif self.api_key_source == "file":
                    api_key = sources.saved
                else:
                    api_key = sources.saved or sources.shell

                if api_key:
                    result["api_key"] = api_key
                else:
                    env_var = API_KEY_ENV_VARS.get(env_slot, env_slot.upper())
                    raise ValueError(
                        f"No API key found for provider '{self.provider}'. "
                        f"Set {env_var}, save a key in the env file, or switch "
                        f"api_key_source to 'env'/'file'."
                    )
        return result


@dataclass
class FastAgentConfig:
    vision: bool = False
    parallel_tools: bool = True
    system_prompt: str = "config/prompts/fast_agent/system.jinja2"
    user_prompt: str = "config/prompts/fast_agent/user.jinja2"


@dataclass
class ManagerConfig:
    vision: bool = False
    system_prompt: str = "config/prompts/manager/system.jinja2"
    stateless: bool = False


@dataclass
class ExecutorConfig:
    vision: bool = False
    system_prompt: str = "config/prompts/executor/system.jinja2"


@dataclass
class AppCardConfig:
    """App card configuration."""

    enabled: bool = True
    mode: str = "local"  # local | server | composite
    app_cards_dir: str = "config/app_cards"
    server_url: Optional[str] = None
    server_timeout: float = 2.0
    server_max_retries: int = 2


@dataclass
class AgentConfig:
    name: str = "droidrun"
    max_steps: int = 15
    reasoning: bool = False
    streaming: bool = True
    after_sleep_action: float = 1.0
    wait_for_stable_ui: float = 0.3
    use_normalized_coordinates: bool = False

    fast_agent: FastAgentConfig = field(default_factory=FastAgentConfig)
    manager: ManagerConfig = field(default_factory=ManagerConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    app_cards: AppCardConfig = field(default_factory=AppCardConfig)

    def get_fast_agent_system_prompt_path(self) -> str:
        return str(PathResolver.resolve(self.fast_agent.system_prompt, must_exist=True))

    def get_fast_agent_user_prompt_path(self) -> str:
        return str(PathResolver.resolve(self.fast_agent.user_prompt, must_exist=True))

    def get_manager_system_prompt_path(self) -> str:
        return str(PathResolver.resolve(self.manager.system_prompt, must_exist=True))

    def get_executor_system_prompt_path(self) -> str:
        return str(PathResolver.resolve(self.executor.system_prompt, must_exist=True))


@dataclass
class DeviceConfig:
    """Device-related configuration."""

    serial: Optional[str] = None
    use_tcp: bool = False
    platform: str = "android"  # "android" or "ios"
    auto_setup: bool = True  # auto-install/fix portal before each run


@dataclass
class TelemetryConfig:
    """Telemetry configuration."""

    enabled: bool = True


@dataclass
class TracingConfig:
    """Tracing configuration."""

    enabled: bool = False
    provider: str = "phoenix"  # "phoenix" or "langfuse"
    langfuse_screenshots: bool = False  # Upload screenshots to Langfuse (if enabled)
    langfuse_secret_key: str = ""  # Set as LANGFUSE_SECRET_KEY env var if not empty
    langfuse_public_key: str = ""  # Set as LANGFUSE_PUBLIC_KEY env var if not empty
    langfuse_host: str = ""  # Set as LANGFUSE_HOST env var if not empty
    langfuse_user_id: str = "anonymous"
    langfuse_session_id: str = (
        ""  # Empty = auto-generate UUID; set to custom value to persist across runs
    )


@dataclass
class LoggingConfig:
    """Logging configuration."""

    debug: bool = False
    save_trajectory: str = "none"
    trajectory_path: str = "trajectories"
    rich_text: bool = False
    trajectory_gifs: bool = True


def _default_disabled_tools() -> List[str]:
    return ["click_at", "click_area", "long_press_at"]


@dataclass
class ToolsConfig:
    """Tools configuration."""

    disabled_tools: List[str] = field(default_factory=_default_disabled_tools)
    stealth: bool = False


@dataclass
class CredentialsConfig:
    """Credentials configuration."""

    enabled: bool = False
    file_path: str = "config/credentials.yaml"


@dataclass
class DroidConfig:
    """Complete Droidrun configuration schema."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    llm_profiles: Dict[str, LLMProfile] = field(default_factory=dict)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    credentials: CredentialsConfig = field(default_factory=CredentialsConfig)
    external_agents: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    mcp: MCPConfig = field(default_factory=MCPConfig)

    def __post_init__(self):
        """Ensure default profiles exist."""
        if not self.llm_profiles:
            self.llm_profiles = self._default_profiles()

    @staticmethod
    def _default_profiles() -> Dict[str, LLMProfile]:
        """Get default agent specific LLM profiles."""
        return {
            "manager": LLMProfile(
                provider="GoogleGenAI",
                model="gemini-3.1-flash-lite-preview",
                temperature=0.2,
                kwargs={},
            ),
            "executor": LLMProfile(
                provider="GoogleGenAI",
                model="gemini-3.1-flash-lite-preview",
                temperature=0.1,
                kwargs={},
            ),
            "fast_agent": LLMProfile(
                provider="GoogleGenAI",
                model="gemini-3.1-flash-lite-preview",
                temperature=0.2,
                kwargs={},
            ),
            "app_opener": LLMProfile(
                provider="GoogleGenAI",
                model="gemini-3.1-flash-lite-preview",
                temperature=0.0,
                kwargs={},
            ),
            "structured_output": LLMProfile(
                provider="GoogleGenAI",
                model="gemini-3.1-flash-lite-preview",
                temperature=0.0,
                kwargs={},
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = asdict(self)
        # Convert LLMProfile objects to dicts
        result["llm_profiles"] = {
            name: asdict(profile) for name, profile in self.llm_profiles.items()
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DroidConfig":
        """Create config from dictionary."""
        # Parse LLM profiles
        llm_profiles = {}
        for name, profile_data in data.get("llm_profiles", {}).items():
            llm_profiles[name] = LLMProfile(**profile_data)

        # Parse agent config with sub-configs
        agent_data = data.get("agent", {})

        fast_agent_data = agent_data.get("fast_agent", {})
        fast_agent_config = (
            FastAgentConfig(**fast_agent_data) if fast_agent_data else FastAgentConfig()
        )

        manager_data = agent_data.get("manager", {})
        manager_config = (
            ManagerConfig(**manager_data) if manager_data else ManagerConfig()
        )

        executor_data = agent_data.get("executor", {})
        executor_config = (
            ExecutorConfig(**executor_data) if executor_data else ExecutorConfig()
        )

        app_cards_data = agent_data.get("app_cards", {})
        app_cards_config = (
            AppCardConfig(**app_cards_data) if app_cards_data else AppCardConfig()
        )

        agent_config = AgentConfig(
            name=agent_data.get("name", "droidrun"),
            max_steps=agent_data.get("max_steps", 15),
            reasoning=agent_data.get("reasoning", False),
            streaming=agent_data.get("streaming", False),
            after_sleep_action=agent_data.get("after_sleep_action", 1.0),
            wait_for_stable_ui=agent_data.get("wait_for_stable_ui", 0.3),
            use_normalized_coordinates=agent_data.get(
                "use_normalized_coordinates", False
            ),
            fast_agent=fast_agent_config,
            manager=manager_config,
            executor=executor_config,
            app_cards=app_cards_config,
        )

        # External agents config - just pass through as-is
        external_agents = data.get("external_agents", {})

        # Parse MCP config
        mcp_data = data.get("mcp", {}) or {}
        mcp_servers = {}
        servers_data = mcp_data.get("servers") or {}
        for server_name, server_data in servers_data.items():
            mcp_servers[server_name] = MCPServerConfig(
                command=server_data.get("command", ""),
                args=server_data.get("args", []),
                env=server_data.get("env", {}),
                prefix=server_data.get("prefix"),
                enabled=server_data.get("enabled", True),
                include_tools=server_data.get("include_tools"),
                exclude_tools=server_data.get("exclude_tools", []),
            )
        mcp_config = MCPConfig(
            enabled=mcp_data.get("enabled", False),
            servers=mcp_servers,
        )

        return cls(
            agent=agent_config,
            llm_profiles=llm_profiles,
            device=DeviceConfig(**data.get("device", {})),
            telemetry=TelemetryConfig(**data.get("telemetry", {})),
            tracing=TracingConfig(**data.get("tracing", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            tools=ToolsConfig(**data.get("tools", {})),
            credentials=CredentialsConfig(**data.get("credentials", {})),
            external_agents=external_agents,
            mcp=mcp_config,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "DroidConfig":
        """
        Load config from YAML file.

        Args:
            path: Path to config file (relative to CWD or absolute)

        Returns:
            DroidConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If file can't be parsed
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
