"""The old typed schema is the current agent schema, not the opaque base schema."""

from mobilerun.addons.agents import require_agents

with require_agents():
    from mobilerun.agents.config.user_config import (
        DEFAULT_DISABLED_TOOLS,
        AgentConfig,
        AppCardConfig,
        CredentialsConfig,
        DeviceConfig,
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

DroidConfig = MobileConfig
