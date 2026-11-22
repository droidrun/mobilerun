"""Public config-manager facade for SDK 0.7 callers."""

from mobilerun._legacy.config import (
    AgentConfig,
    AppCardConfig,
    CredentialsConfig,
    DeviceConfig,
    DroidConfig,
    ExecutorConfig,
    FastAgentConfig,
    LLMProfile,
    LoggingConfig,
    ManagerConfig,
    MobileConfig,
    TelemetryConfig,
    ToolsConfig,
    TracingConfig,
)
from mobilerun.addons.agents import require_agents

with require_agents():
    from mobilerun.agents.config.loader import ConfigLoader, OutdatedConfigError
    from mobilerun.agents.utils.paths import PathResolver
from mobilerun.config.prompts import PromptLoader
