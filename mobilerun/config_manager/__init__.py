from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mobilerun.config_manager.config_manager import (
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
    from mobilerun.config_manager.loader import ConfigLoader, OutdatedConfigError
    from mobilerun.config_manager.path_resolver import PathResolver
    from mobilerun.config_manager.prompt_loader import PromptLoader

_EXPORTS = {
    "AgentConfig": ("mobilerun.config_manager.config_manager", "AgentConfig"),
    "AppCardConfig": ("mobilerun.config_manager.config_manager", "AppCardConfig"),
    "CredentialsConfig": (
        "mobilerun.config_manager.config_manager",
        "CredentialsConfig",
    ),
    "DeviceConfig": ("mobilerun.config_manager.config_manager", "DeviceConfig"),
    "DroidConfig": ("mobilerun.config_manager.config_manager", "DroidConfig"),
    "ExecutorConfig": ("mobilerun.config_manager.config_manager", "ExecutorConfig"),
    "FastAgentConfig": ("mobilerun.config_manager.config_manager", "FastAgentConfig"),
    "LLMProfile": ("mobilerun.config_manager.config_manager", "LLMProfile"),
    "LoggingConfig": ("mobilerun.config_manager.config_manager", "LoggingConfig"),
    "ManagerConfig": ("mobilerun.config_manager.config_manager", "ManagerConfig"),
    "MobileConfig": ("mobilerun.config_manager.config_manager", "MobileConfig"),
    "TelemetryConfig": ("mobilerun.config_manager.config_manager", "TelemetryConfig"),
    "ToolsConfig": ("mobilerun.config_manager.config_manager", "ToolsConfig"),
    "TracingConfig": ("mobilerun.config_manager.config_manager", "TracingConfig"),
    "ConfigLoader": ("mobilerun.config_manager.loader", "ConfigLoader"),
    "OutdatedConfigError": ("mobilerun.config_manager.loader", "OutdatedConfigError"),
    "PathResolver": ("mobilerun.config_manager.path_resolver", "PathResolver"),
    "PromptLoader": ("mobilerun.config_manager.prompt_loader", "PromptLoader"),
}

__all__ = [
    "MobileConfig",
    "DroidConfig",
    "LLMProfile",
    "AgentConfig",
    "FastAgentConfig",
    "ManagerConfig",
    "ExecutorConfig",
    "AppCardConfig",
    "DeviceConfig",
    "TelemetryConfig",
    "TracingConfig",
    "LoggingConfig",
    "ToolsConfig",
    "CredentialsConfig",
    "ConfigLoader",
    "OutdatedConfigError",
    "PathResolver",
    "PromptLoader",
]


def __getattr__(name):
    if name in _EXPORTS:
        module, attribute = _EXPORTS[name]
        value = getattr(import_module(module), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
